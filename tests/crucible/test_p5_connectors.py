"""Crucible P5 — breadth connectors (Stooq / GDELT / SEC EDGAR), spec §4.2.

Each connector is driven entirely from fixtures (injected ``transport`` — NO network, NO key/UA), and
each must clear the same load-bearing gate P1b established: the **as-of-join reconstruction tripwire**
(release-time join, never reference-period). EDGAR is the cleanest true-PIT case — the filing
``filed`` date IS the release, and an amendment (same period, later filing) must replay as a revision.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.crucible.data import (
    DataConnector,
    EdgarConnector,
    GdeltConnector,
    SeriesRef,
    StooqConnector,
    asof_join,
    assert_asof_join_causal,
    validate_series,
)

# A daily calendar the connectors' series must join to causally.
_BARS = np.arange(np.datetime64("2019-06-01"), np.datetime64("2020-06-01"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


# ------------------------------------------------------------------ Stooq ----

def _stooq_csv() -> str:
    rows = "\n".join(f"2020-01-{d:02d},10,11,9,{100 + d},0" for d in range(1, 29))
    return "Date,Open,High,Low,Close,Volume\n" + rows


def test_stooq_parses_close_and_stamps_release_lag() -> None:
    conn = StooqConnector(transport=lambda url: _stooq_csv(), symbols=(("^spx", "S&P 500"),))
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2020-01-01", "2020-04-01")
    assert data.n_obs == 28
    assert data.value[0] == 101.0                                   # close of 2020-01-01
    assert data.reference_period[0] == np.datetime64("2020-01-01", "ns")
    assert data.release_timestamp[0] == np.datetime64("2020-01-02", "ns")   # +1d (post-session)
    assert validate_series(data).passed
    assert_asof_join_causal(data, _BARS)                            # PIT tripwire


def test_stooq_out_of_range_rows_dropped_and_asof_vintage() -> None:
    conn = StooqConnector(transport=lambda url: _stooq_csv(), symbols=(("^spx", "x"),))
    ref = conn.discover()[0]
    # as_of before some releases: only rows whose release (ref+1d) <= as_of survive.
    data = conn.fetch(ref, "2020-01-01", "2020-04-01", as_of="2020-01-10")
    assert data.reference_period.max() <= np.datetime64("2020-01-09", "ns")


# ------------------------------------------------------------------ GDELT ----

def _gdelt_payload() -> dict:
    return {"timeline": [{"series": "Average Tone", "data": [
        {"date": f"202001{d:02d}T000000Z", "value": -2.0 + 0.1 * d} for d in range(1, 29)]}]}


def test_gdelt_parses_daily_tone_and_is_causal() -> None:
    conn = GdeltConnector(transport=lambda url: _gdelt_payload(), themes=(("gold", "gold tone"),))
    ref = conn.discover()[0]
    assert ref.terminal == "gdelt:gold"                             # lexer-safe single-token theme
    data = conn.fetch(ref, "2020-01-01", "2020-04-01")
    assert data.n_obs == 28
    assert data.value[0] == pytest.approx(-1.9)
    assert data.release_timestamp[0] == np.datetime64("2020-01-02", "ns")   # tone settles next day
    assert validate_series(data).passed
    assert_asof_join_causal(data, _BARS)


def test_gdelt_handles_empty_timeline() -> None:
    conn = GdeltConnector(transport=lambda url: {"timeline": []}, themes=(("gold", "x"),))
    data = conn.fetch(conn.discover()[0], "2020-01-01", "2020-04-01")
    assert data.n_obs == 0


# ------------------------------------------------------------------ SEC EDGAR ----

def _edgar_payload() -> dict:
    # Q3 (filed Nov), FY (filed Feb), and an AMENDMENT of the FY period (filed Mar, restates the value).
    return {"units": {"USD": [
        {"end": "2019-09-30", "val": 100.0, "filed": "2019-11-01", "form": "10-Q"},
        {"end": "2019-12-31", "val": 200.0, "filed": "2020-02-15", "form": "10-K"},
        {"end": "2019-12-31", "val": 205.0, "filed": "2020-03-10", "form": "10-K/A"},
    ]}}


def test_edgar_uses_filing_acceptance_as_release_and_replays_amendments() -> None:
    conn = EdgarConnector(transport=lambda url: _edgar_payload(),
                          concepts=(("320193", "Revenues", "AAPL revenue"),))
    ref = conn.discover()[0]
    assert ref.series_id == "0000320193:Revenues"                  # CIK zero-padded to 10 digits
    data = conn.fetch(ref, "2019-01-01", "2020-12-31")
    assert data.n_obs == 3
    assert data.meta["unit"] == "USD"
    # release_timestamp is the actual `filed` date — NOT reference+lag (the true-PIT differentiator).
    fy = data.reference_period == np.datetime64("2019-12-31", "ns")
    assert set(data.release_timestamp[fy].astype("datetime64[D]").astype(str)) == {
        "2020-02-15", "2020-03-10"}
    assert_asof_join_causal(data, _BARS)

    # The FY value is 200 from its Feb filing, then 205 once the March amendment is public (revision).
    j = asof_join(data, _BARS)
    i_feb = int(np.where(_BARS == np.datetime64("2020-02-20T00:00:00", "ns"))[0][0])
    i_mar = int(np.where(_BARS == np.datetime64("2020-03-15T00:00:00", "ns"))[0][0])
    assert j[i_feb] == 200.0
    assert j[i_mar] == 205.0


def _edgar_mixed_frames_payload() -> dict:
    # The SAME concept reported under four overlapping frames: standalone Q1 (~90d), H1 cumulative
    # (~181d), full-year (~365d), and standalone Q3 (~91d). Only the two standalone quarters are a
    # consistent flow; the cumulative + annual are the sawtooth-inducing frames the filter must drop.
    return {"units": {"USD": [
        {"start": "2024-01-01", "end": "2024-03-31", "val": 90.0, "filed": "2024-05-01", "form": "10-Q"},
        {"start": "2024-01-01", "end": "2024-06-30", "val": 190.0, "filed": "2024-08-01", "form": "10-Q"},
        {"start": "2024-01-01", "end": "2024-12-31", "val": 400.0, "filed": "2025-02-01", "form": "10-K"},
        {"start": "2024-07-01", "end": "2024-09-30", "val": 95.0, "filed": "2024-11-01", "form": "10-Q"},
    ]}}


def test_edgar_frame_filter_keeps_only_standalone_quarters() -> None:
    ref = SeriesRef("edgar", "0000320193:RevenueFromContractWithCustomerExcludingAssessedTax",
                    "fundamental")
    # Default (period_days=(80,100)): only the two ~90d standalone quarters survive — a clean flow.
    conn = EdgarConnector(transport=lambda url: _edgar_mixed_frames_payload())
    data = conn.fetch(ref, "2024-01-01", "2025-12-31")
    assert data.n_obs == 2
    assert sorted(data.value.tolist()) == [90.0, 95.0]           # NOT the 190 YTD / 400 annual frames
    # Negative tripwire: disabling the filter reintroduces the mixed-frame sawtooth (all 4 rows).
    conn_all = EdgarConnector(transport=lambda url: _edgar_mixed_frames_payload(), period_days=None)
    assert conn_all.fetch(ref, "2024-01-01", "2025-12-31").n_obs == 4


def test_edgar_live_without_user_agent_fails_closed() -> None:
    conn = EdgarConnector(user_agent="")                           # no UA, no transport
    with pytest.raises(RuntimeError, match="User-Agent"):
        conn.fetch(conn.discover()[0], "2019-01-01", "2020-12-31")


def test_edgar_bad_series_id_rejected() -> None:
    conn = EdgarConnector(transport=lambda url: {"units": {}})
    with pytest.raises(ValueError, match="<cik>:<concept>"):
        conn.fetch(SeriesRef("edgar", "no-colon", "fundamental"), "2019-01-01", "2020-12-31")


# ------------------------------------------------------------------ protocol ----

def test_p5_connectors_satisfy_protocol() -> None:
    assert isinstance(StooqConnector(transport=lambda u: ""), DataConnector)
    assert isinstance(GdeltConnector(transport=lambda u: {}), DataConnector)
    assert isinstance(EdgarConnector(transport=lambda u: {}), DataConnector)
