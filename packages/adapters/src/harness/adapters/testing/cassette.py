"""HTTP record/replay for adapter tests ("cassettes"), so CI never calls a live API.

``CassetteTransport`` plugs into any SDK built on ``httpx2`` via its ``http_client``. In record
mode it forwards to the network and saves each interaction. In replay mode it serves them back
in order and fails on any request that differs from the recorded one. Only the method, path,
query and JSON body of a request are stored; **request headers (API keys) are never written**.
Response headers are reduced to ``content-type`` and ``request-id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx2
from pydantic import BaseModel, JsonValue

Mode = Literal["record", "replay"]
_KEPT_RESPONSE_HEADERS = ("content-type", "request-id")


class CassetteMismatch(AssertionError):
    """A replayed request differs from the recording (re-record with ``just record``)."""


class _Recorded(BaseModel):
    method: str
    path: str
    body: JsonValue
    status: int
    headers: dict[str, str]
    response: str


def _request_body(request: httpx2.Request) -> JsonValue:
    raw = request.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode(errors="replace")


def _path(request: httpx2.Request) -> str:
    return request.url.raw_path.decode()


class CassetteTransport(httpx2.AsyncBaseTransport):
    """Records or replays HTTP interactions stored as JSON at ``path``."""

    def __init__(
        self, path: Path, mode: Mode, *, network: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        """``network`` is the transport used when recording (default: a real HTTP transport)."""
        self.path = path
        self.mode = mode
        self._recorded: list[_Recorded] = []
        self._cursor = 0
        self._inner: httpx2.AsyncBaseTransport | None = None
        if mode == "replay":
            data = json.loads(path.read_text())
            self._recorded = [_Recorded.model_validate(item) for item in data]
        else:
            self._inner = network or httpx2.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        """Serve the next recorded response, or perform and record the real request."""
        body = _request_body(request)
        if self._inner is not None:
            response = await self._inner.handle_async_request(request)
            content = await response.aread()
            headers = {
                k: v for k in _KEPT_RESPONSE_HEADERS if (v := response.headers.get(k)) is not None
            }
            self._recorded.append(
                _Recorded(
                    method=request.method,
                    path=_path(request),
                    body=body,
                    status=response.status_code,
                    headers=headers,
                    response=content.decode(),
                )
            )
            return httpx2.Response(
                response.status_code, headers=headers, content=content, request=request
            )

        if self._cursor >= len(self._recorded):
            raise CassetteMismatch(
                f"{self.path.name}: unexpected extra request {request.method} {_path(request)}"
            )
        rec = self._recorded[self._cursor]
        self._cursor += 1
        actual = (request.method, _path(request), body)
        if actual != (rec.method, rec.path, rec.body):
            raise CassetteMismatch(
                f"{self.path.name}: request #{self._cursor} differs from the recording.\n"
                f"recorded: {rec.method} {rec.path} {json.dumps(rec.body, sort_keys=True)[:2000]}\n"
                f"actual:   {actual[0]} {actual[1]} {json.dumps(body, sort_keys=True)[:2000]}"
            )
        return httpx2.Response(
            rec.status, headers=rec.headers, content=rec.response.encode(), request=request
        )

    def save(self) -> None:
        """Write recorded interactions (record mode only)."""
        if self.mode != "record":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in self._recorded]
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    async def aclose(self) -> None:
        """Close the underlying network transport, if any."""
        if self._inner is not None:
            await self._inner.aclose()
