"""TWSE T86 three-institutional-investors connector — Crucible v2.8.

Fixture field names/shape mirror the LIVE response captured 2026-07-04 (session S553-cont-115):
one ``fields`` array of 19 Chinese column names, one row per security, no per-row date (the whole
response describes one queried calendar day), comma-formatted (possibly negative) number strings.
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.crucible.data import assert_asof_join_causal, validate_series
from finrl_pro_ds.crucible.data.twse_institutional import TwseInstitutionalConnector

_FIELDS = [
    "證券代號", "證券名稱",
    "外陸資買進股數(不含外資自營商)", "外陸資賣出股數(不含外資自營商)", "外陸資買賣超股數(不含外資自營商)",
    "外資自營商買進股數", "外資自營商賣出股數", "外資自營商買賣超股數",
    "投信買進股數", "投信賣出股數", "投信買賣超股數",
    "自營商買賣超股數",
    "自營商買進股數(自行買賣)", "自營商賣出股數(自行買賣)", "自營商買賣超股數(自行買賣)",
    "自營商買進股數(避險)", "自營商賣出股數(避險)", "自營商買賣超股數(避險)",
    "三大法人買賣超股數",
]

# A daily calendar the connector's series must join to causally.
_BARS = np.arange(np.datetime64("2026-06-01"), np.datetime64("2026-07-15"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


def _row(ticker: str, name: str, foreign_net: str, trust_net: str, dealer_net: str,
         total_net: str) -> list[str]:
    """Build a full 19-column T86 row; only the 5 fields this connector reads carry real values,
    the rest are placeholder zeros (never read by the parser, which looks up columns by name)."""
    row = ["0"] * len(_FIELDS)
    row[0], row[1] = ticker, name
    row[4] = foreign_net       # 外陸資買賣超股數(不含外資自營商)
    row[10] = trust_net        # 投信買賣超股數
    row[11] = dealer_net       # 自營商買賣超股數
    row[18] = total_net        # 三大法人買賣超股數
    return row


def _t86_payload(date: str, rows: list[list[str]]) -> dict:
    return {"stat": "OK", "date": date, "title": "115年 test 三大法人買賣超日報",
            "hints": "單位：股", "fields": list(_FIELDS), "data": rows}


def _transport_for(days: dict[str, dict]):
    calls: list[str] = []

    def transport(url: str) -> dict:
        calls.append(url)
        date = url.split("date=")[1].split("&")[0]
        if date not in days:
            raise RuntimeError(f"no fixture for {date}")
        return days[date]
    transport.calls = calls   # type: ignore[attr-defined]
    return transport


def test_parses_named_columns_and_stamps_release_lag() -> None:
    days = {"20260701": _t86_payload("20260701", [
        _row("0050", "元大台灣50       ", "12,345,678", "-1,000,000", "500,000", "11,845,678"),
    ])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0)
    refs = {r.series_id: r for r in conn.discover()}
    assert set(refs) == {"0050:foreign_net", "0050:trust_net", "0050:dealer_net", "0050:total_net"}

    data = conn.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-01")
    assert data.n_obs == 1
    assert data.value[0] == 12_345_678.0
    assert data.reference_period[0] == np.datetime64("2026-07-01", "ns")
    assert data.release_timestamp[0] == np.datetime64("2026-07-02", "ns")   # +1 calendar day
    assert validate_series(data).passed
    assert_asof_join_causal(data, _BARS)


def test_parses_comma_and_negative_numbers() -> None:
    days = {"20260701": _t86_payload("20260701", [
        _row("0050", "x", "1,234,567", "-445,000", "-2,500", "787,067"),
    ])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0)
    refs = {r.series_id: r for r in conn.discover()}
    trust = conn.fetch(refs["0050:trust_net"], "2026-07-01", "2026-07-01")
    dealer = conn.fetch(refs["0050:dealer_net"], "2026-07-01", "2026-07-01")
    assert trust.value[0] == -445_000.0
    assert dealer.value[0] == -2_500.0


def test_day_cache_shares_one_http_call_across_series() -> None:
    days = {"20260701": _t86_payload("20260701", [
        _row("0050", "x", "1", "2", "3", "6"),
    ])}
    transport = _transport_for(days)
    conn = TwseInstitutionalConnector(transport=transport, tickers=("0050",), sleep_seconds=0)
    refs = {r.series_id: r for r in conn.discover()}
    conn.fetch(refs["0050:foreign_net"], "2026-07-01", "2026-07-01")
    conn.fetch(refs["0050:trust_net"], "2026-07-01", "2026-07-01")
    conn.fetch(refs["0050:dealer_net"], "2026-07-01", "2026-07-01")
    conn.fetch(refs["0050:total_net"], "2026-07-01", "2026-07-01")
    assert len(transport.calls) == 1        # 4 series, same day -> exactly 1 HTTP call


def test_missing_ticker_returns_empty_series_not_a_crash() -> None:
    days = {"20260701": _t86_payload("20260701", [_row("2330", "x", "1", "1", "1", "3")])}
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0)
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-07-01", "2026-07-01")
    assert data.n_obs == 0


def test_non_ok_response_treated_as_no_data() -> None:
    days = {"20260704": {"stat": "很抱歉，沒有符合條件的資料!"}}     # a non-trading-day-shaped response
    conn = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                      sleep_seconds=0)
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-07-04", "2026-07-04")
    assert data.n_obs == 0


def test_respects_max_lookback_days_cap() -> None:
    conn = TwseInstitutionalConnector(transport=_transport_for({}), tickers=("0050",),
                                      max_lookback_days=5, sleep_seconds=0)
    ref = conn.discover()[0]
    # requested a 30-day window but every day will 404/raise in the fixture (empty `days` dict) —
    # the cap only needs to be verified via `meta`, not via how many days actually had data.
    data = conn.fetch(ref, "2026-06-01", "2026-07-01")
    assert data.meta["n_days_queried"] == 6     # capped to max_lookback_days + 1 (inclusive span)


def test_negative_lag_misconfiguration_is_caught_by_the_shared_pit_gate() -> None:
    """CAUS-05-style negative tripwire: this connector has no revision/holiday-lag model to break the
    way COT's did, so the meaningful mutation is a sign/arithmetic error in `release_lag_days` (an
    author mistake, not a real-world assumption). Proves the shared `validate_series` PIT-sanity check
    (release must not predate its own reference period) actually fires against THIS connector's
    output, not just in the abstract."""
    days = {"20260701": _t86_payload("20260701", [_row("0050", "x", "1", "1", "1", "3")])}
    broken = TwseInstitutionalConnector(transport=_transport_for(days), tickers=("0050",),
                                        release_lag_days=-1, sleep_seconds=0)   # sign-flipped bug
    ref = broken.discover()[0]
    data = broken.fetch(ref, "2026-07-01", "2026-07-01")
    report = validate_series(data)
    assert not report.passed
    assert any("PIT violation" in r for r in report.reasons)
