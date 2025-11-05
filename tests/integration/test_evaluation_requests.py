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


def test_risk_profile_operations() -> None:
    session = Mock()
    upsert_response = _mock_response(201, {"profileId": "abc"})
    get_response = _mock_response(200, {"profileId": "abc", "name": "Baseline"})
    session.request.side_effect = [upsert_response, get_response]

    client = FinRLProAPIClient(base_url="https://finrl-pro.dev/api", session=session)
    payload = {
        "profileId": "abc",
        "name": "Baseline",
        "maxCapitalAtRisk": 0.05,
        "maxDrawdownPct": 0.1,
        "leverageCap": 2.0,
        "sandboxRequired": False,
        "approvedBy": "risk_officer",
        "effectiveDate": "2025-01-01",
    }

    upsert = client.upsert_risk_profile(payload)
    assert upsert["profileId"] == "abc"

    profile = client.get_risk_profile("abc")
    assert profile["name"] == "Baseline"

    session.request.assert_any_call(
        method="POST",
        url="https://finrl-pro.dev/api/risk-profiles",
        json=payload,
    )
    session.request.assert_any_call(
        method="GET",
        url="https://finrl-pro.dev/api/risk-profiles/abc",
        json=None,
    )


def test_report_endpoints() -> None:
    session = Mock()
    create_response = _mock_response(201, {"reportId": "rep-1"})
    fetch_response = _mock_response(200, {"reportId": "rep-1", "approvalStatus": "Pending"})
    session.request.side_effect = [create_response, fetch_response]

    client = FinRLProAPIClient(base_url="https://finrl-pro.dev/api", session=session)
    payload = {
        "fingerprintId": "fp-123",
        "summaryLocation": "s3://reports/fp-123/report.json",
        "metrics": {"sharpe_ratio": 1.1},
        "statisticalTests": [],
    }

    created = client.create_report(payload)
    assert created["reportId"] == "rep-1"

    report = client.get_report("rep-1")
    assert report["approvalStatus"] == "Pending"

    session.request.assert_any_call(
        method="POST",
        url="https://finrl-pro.dev/api/reports",
        json=payload,
    )
    session.request.assert_any_call(
        method="GET",
        url="https://finrl-pro.dev/api/reports/rep-1",
        json=None,
    )
