import pytest

from harness.core import HarnessError, LimitExceeded, ModelError, PermissionDenied, ToolError


def test_error_codes_are_stable() -> None:
    assert ModelError.code == "model_error"
    assert ToolError.code == "tool_error"
    assert LimitExceeded.code == "limit_exceeded"
    assert PermissionDenied.code == "permission_denied"
    assert HarnessError.code == "internal"


def test_model_error_retryable_metadata() -> None:
    err = ModelError("rate limited", retryable=True, status_code=429, provider="p")
    assert err.retryable
    assert err.status_code == 429
    assert isinstance(err, HarnessError)


def test_limit_exceeded_message() -> None:
    err = LimitExceeded("steps", limit_value=10, actual=11)
    assert err.limit == "steps"
    assert "steps limit exceeded" in str(err)
    with pytest.raises(HarnessError):
        raise err


def test_tool_error_carries_tool_name() -> None:
    assert ToolError("nope", tool_name="run_shell").tool_name == "run_shell"
