import json
from pathlib import Path

import httpx2
import pytest

from harness.adapters.testing import CassetteMismatch, CassetteTransport


def network(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={"content-type": "application/json", "request-id": "req_1", "set-cookie": "s=1"},
        json={"echo": json.loads(request.read())},
    )


async def post(transport: CassetteTransport, body: object) -> httpx2.Response:
    async with httpx2.AsyncClient(transport=transport) as client:
        return await client.post(
            "https://api.example.com/v1/messages?beta=true",
            json=body,
            headers={"x-api-key": "sk-secret-key"},
        )


async def test_record_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    recorder = CassetteTransport(path, "record", network=httpx2.MockTransport(network))
    recorded = await post(recorder, {"q": 1})
    recorder.save()

    stored = path.read_text()
    assert "sk-secret-key" not in stored
    assert "set-cookie" not in stored
    assert json.loads(stored)[0]["path"] == "/v1/messages?beta=true"

    replayed = await post(CassetteTransport(path, "replay"), {"q": 1})
    assert replayed.json() == recorded.json() == {"echo": {"q": 1}}
    assert replayed.headers["request-id"] == "req_1"


async def test_replay_rejects_different_or_extra_requests(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    recorder = CassetteTransport(path, "record", network=httpx2.MockTransport(network))
    await post(recorder, {"q": 1})
    recorder.save()

    with pytest.raises(CassetteMismatch, match="differs"):
        await post(CassetteTransport(path, "replay"), {"q": 2})

    replay = CassetteTransport(path, "replay")
    await post(replay, {"q": 1})
    with pytest.raises(CassetteMismatch, match="extra request"):
        await post(replay, {"q": 1})
