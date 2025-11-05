"""Integration tests for control plane evaluation requests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from finrl_pro.mlops.api_client import FinRLProAPIClient, FinRLProAPIError


def _mock_response(status_code: int, payload: dict[str, object]) -> Mock:
    response = Mock()
    response.status_code = status_code
    body = ""
    json_bytes = b""
    if payload:
        import json

        body = json.dumps(payload)
        json_bytes = body.encode("utf-8")
        response.json.return_value = payload
    else:
        response.json.return_value = {}
    response.content = json_bytes
    response.text = body
    return response


def test_create_evaluation_dispatches_request() -> None:
    """API client must call the evaluations endpoint and return JSON."""
    session = Mock()
    response = _mock_response(202, {"jobId": "123"})
    session.request.return_value = response

    client = FinRLProAPIClient(base_url="https://finrl-pro.dev/api", session=session)
    payload = {"fingerprintId": "abc", "benchmarkId": "bench", "walkForwardSplits": 3}

    result = client.create_evaluation(payload)
    session.request.assert_called_once_with(
        method="POST",
        url="https://finrl-pro.dev/api/evaluations",
        json=payload,
    )
    assert result["jobId"] == "123"


def test_create_evaluation_raises_on_error() -> None:
    """API client should raise an error on non-success responses."""
    session = Mock()
    response = _mock_response(500, {})
    response.text = "Server error"
    session.request.return_value = response

    client = FinRLProAPIClient(base_url="https://finrl-pro.dev/api", session=session)

    with pytest.raises(FinRLProAPIError):
        client.create_evaluation({"fingerprintId": "abc", "benchmarkId": "bench"})
