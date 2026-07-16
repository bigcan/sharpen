"""Unit tests for the shared FinMind HTTP helper ``_finmind_get`` (transient-network resilience).

All offline — ``requests.get`` and ``time.sleep`` are mocked, no token, no network. Guards the
S553-cont-133 robustness fix: a requests-level ``ConnectionError``/``Timeout`` is raised BEFORE any
HTTP status, so it must be retried with capped exponential backoff and, on final give-up, surface as
a ``RuntimeError`` (so callers that ``except RuntimeError`` — the per-name skip paths in
``fetch_taiwan_fundamentals_finmind`` and the panel loader — degrade gracefully instead of crashing a
multi-hour fetch). A 402 quota block is terminal and must NOT be retried.
"""
from __future__ import annotations

from unittest import mock

import pytest
import requests

from scripts.data import fetch_taiwan_finmind as finmind

MOD = "scripts.data.fetch_taiwan_finmind"


class _FakeResp:
    """Minimal stand-in for a ``requests.Response`` (only what ``_finmind_get`` reads)."""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_finmind_get_retries_transient_network_error_then_succeeds():
    """ConnectionError twice, then a 200 → returns the frame; backs off between the two failures."""
    ok = _FakeResp(200, {"status": 200, "data": [{"date": "2020-01-02", "open": 1.0, "close": 2.0}]})
    with mock.patch(
        f"{MOD}.requests.get",
        side_effect=[
            requests.exceptions.ConnectionError("WinError 10054: connection reset by peer"),
            requests.exceptions.ConnectionError("WinError 10054: connection reset by peer"),
            ok,
        ],
    ) as m, mock.patch(f"{MOD}.time.sleep") as sleep:
        df = finmind._finmind_get("TaiwanStockPrice", "2330", "2020-01-01", None, "")

    assert m.call_count == 3          # two transient failures + one success
    assert sleep.call_count == 2      # capped exponential backoff after each failure (not the success)
    assert list(df["open"]) == [1.0]  # the successful payload flowed straight through


def test_finmind_get_402_quota_is_terminal_not_retried():
    """A 402 quota block raises immediately — never retried (neither status nor network retry)."""
    with mock.patch(f"{MOD}.requests.get", return_value=_FakeResp(402, text="quota")) as m, \
            mock.patch(f"{MOD}.time.sleep") as sleep:
        with pytest.raises(RuntimeError, match="402"):
            finmind._finmind_get("TaiwanStockPrice", "2330", "2020-01-01", None, "")

    assert m.call_count == 1
    assert sleep.call_count == 0


def test_finmind_get_persistent_network_error_becomes_runtimeerror():
    """A network error that never clears is capped at ``net_retries`` and surfaces as RuntimeError.

    RuntimeError (not the raw ConnectionError) is the contract: callers' ``except RuntimeError`` skip
    paths then drop the one id/channel instead of crashing the whole run.
    """
    with mock.patch(
        f"{MOD}.requests.get",
        side_effect=requests.exceptions.Timeout("read timed out"),
    ) as m, mock.patch(f"{MOD}.time.sleep") as sleep:
        with pytest.raises(RuntimeError, match="network error"):
            finmind._finmind_get(
                "TaiwanStockMonthRevenue", "2330", "2020-01-01", None, "", net_retries=4)

    assert m.call_count == 4          # four attempts, then give up
    assert sleep.call_count == 3      # backoff between attempts, none after the final failure
