"""TAIFEX large-trader OI connector — Crucible v2.8.

Fixture field names/shape mirror the LIVE response captured 2026-07-04 (session S553-cont-115):
flat list of dicts, every (contract, settlement-month, trader-type) combination present, an
all-months sentinel ``SettlementMonth == "999912"``, ``TypeOfTraders`` "0"=all / "1"=subset, and a
poll-only endpoint that ignores any date parameter (this connector accumulates locally instead).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sharpen.crucible.data import assert_asof_join_causal, validate_series
from sharpen.crucible.data.taifex_positioning import TaifexPositioningConnector

_BARS = np.arange(np.datetime64("2026-06-01"), np.datetime64("2026-07-15"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


def _row(contract: str, date: str, month: str, ttype: str, t5b, t5s, t10b, t10s, oi) -> dict:
    return {"Date": date, "Contract": contract, "ContractName": f"{contract} futures",
            "SettlementMonth": month, "TypeOfTraders": ttype,
            "Top5Buy": str(t5b), "Top5Sell": str(t5s), "Top10Buy": str(t10b), "Top10Sell": str(t10s),
            "OIOfMarket": str(oi)}


def _snapshot(date: str) -> list[dict]:
    """One live-shaped snapshot: TX with a per-month row (should be ignored), the all-months
    sentinel for both trader-type rows (only type "0" should be used), plus TE."""
    return [
        _row("TX", date, "202607", "0", 66117, 58550, 75149, 78829, 110141),   # per-month: ignored
        _row("TX", date, "999912", "0", 67030, 60376, 76062, 80987, 117292),   # used
        _row("TX", date, "999912", "1", 63283, 60376, 72314, 80987, 117292),   # wrong trader-type: ignored
        _row("TE", date, "999912", "0", 236, 331, 313, 410, 514),              # used
    ]


def _transport_for(*snapshots: list[dict]):
    """Returns a transport that yields each snapshot in order on successive calls (simulating
    successive scheduled polls on successive real days), repeating the last one thereafter."""
    calls = {"n": 0}

    def transport(url: str) -> list[dict]:
        i = min(calls["n"], len(snapshots) - 1)
        calls["n"] += 1
        return snapshots[i]
    return transport


def test_first_poll_accumulates_and_computes_concentration(tmp_path) -> None:
    store = tmp_path / "store.json"
    conn = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260703")),
                                      contracts=("TX", "TE"), store_path=store)
    refs = {r.series_id: r for r in conn.discover()}
    data = conn.fetch(refs["TX:top5_net_pct_oi"], "2026-06-01", "2026-07-15")
    assert data.n_obs == 1
    expected = (67030 - 60376) / 117292
    assert data.value[0] == pytest.approx(expected)
    assert data.reference_period[0] == np.datetime64("2026-07-03", "ns")
    assert data.release_timestamp[0] == np.datetime64("2026-07-04", "ns")     # +1 calendar day
    assert validate_series(data).passed
    assert_asof_join_causal(data, _BARS)
    assert store.exists()


def test_only_all_months_sentinel_and_all_traders_row_used(tmp_path) -> None:
    conn = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260703")),
                                      contracts=("TX",), store_path=tmp_path / "s.json")
    refs = {r.series_id: r for r in conn.discover()}
    data = conn.fetch(refs["TX:top10_net_pct_oi"], "2026-06-01", "2026-07-15")
    # from the 999912/"0" row (76062-80987)/117292, NOT the per-month row or the "1" trader-type row
    assert data.value[0] == pytest.approx((76062 - 80987) / 117292)


def test_polling_the_same_day_twice_is_idempotent_not_overwritten(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    # Pre-seed the store as if a prior instance already recorded 2026-07-03 with a KNOWN value.
    store_path.write_text(json.dumps({"TX|20260703": {"top5_net_pct_oi": 0.111,
                                                       "top10_net_pct_oi": 0.222}}),
                          encoding="utf-8")
    # A fresh poll returns the SAME day but with DIFFERENT numbers -- must NOT overwrite.
    conn = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260703")),
                                      contracts=("TX",), store_path=store_path)
    refs = {r.series_id: r for r in conn.discover()}
    data = conn.fetch(refs["TX:top5_net_pct_oi"], "2026-06-01", "2026-07-15")
    assert data.n_obs == 1
    assert data.value[0] == pytest.approx(0.111)          # the pre-seeded value survives, untouched


def test_second_instance_sees_prior_accumulation_plus_a_new_day(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    conn_day1 = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260702")),
                                           contracts=("TX",), store_path=store_path)
    ref = conn_day1.discover()[0]
    conn_day1.fetch(ref, "2026-06-01", "2026-07-15")       # triggers the poll -> writes day 1

    # A brand-new connector instance (as a fresh scheduled-tick process would construct), same store.
    conn_day2 = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260703")),
                                           contracts=("TX",), store_path=store_path)
    ref2 = conn_day2.discover()[0]
    data = conn_day2.fetch(ref2, "2026-06-01", "2026-07-15")
    assert data.n_obs == 2                                  # day1 (from disk) + day2 (freshly polled)
    assert sorted(np.datetime_as_string(data.reference_period, unit="D")) == ["2026-07-02", "2026-07-03"]


def test_poll_failure_degrades_to_empty_store_not_a_crash(tmp_path) -> None:
    def failing_transport(url: str) -> list[dict]:
        raise RuntimeError("network down")
    conn = TaifexPositioningConnector(transport=failing_transport, contracts=("TX",),
                                      store_path=tmp_path / "store.json")
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-06-01", "2026-07-15")
    assert data.n_obs == 0


def test_negative_lag_misconfiguration_is_caught_by_the_shared_pit_gate(tmp_path) -> None:
    """CAUS-05-style negative tripwire (mirrors test_twse_institutional.py's): proves the shared
    `validate_series` PIT-sanity check fires against THIS connector's output if `release_lag_days`
    were ever sign-flipped by an author mistake -- the meaningful mutation for a poll-and-accumulate
    connector with no revision/holiday-lag model to break."""
    conn = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260703")),
                                      contracts=("TX",), release_lag_days=-1,
                                      store_path=tmp_path / "store.json")
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-06-01", "2026-07-15")
    report = validate_series(data)
    assert not report.passed
    assert any("PIT violation" in r for r in report.reasons)


def test_corrupted_store_file_recovers_to_empty_not_a_crash(tmp_path) -> None:
    """A crash mid-write (Task Scheduler force-kill, power loss) could leave the JSON store
    truncated/invalid. `_AccumulationStore.load()` must degrade to empty, never raise -- accumulated
    history is lost (a real, disclosed limitation of a non-atomic write), but the tick must not fail."""
    store_path = tmp_path / "store.json"
    store_path.write_text("{not valid json truncated mid-wri", encoding="utf-8")
    conn = TaifexPositioningConnector(transport=_transport_for(_snapshot("20260703")),
                                      contracts=("TX",), store_path=store_path)
    ref = conn.discover()[0]
    data = conn.fetch(ref, "2026-06-01", "2026-07-15")   # must not raise
    assert data.n_obs == 1                                # the fresh poll still merges in cleanly


def test_poll_happens_at_most_once_per_instance(tmp_path) -> None:
    calls = {"n": 0}

    def counting_transport(url: str) -> list[dict]:
        calls["n"] += 1
        return _snapshot("20260703")

    conn = TaifexPositioningConnector(transport=counting_transport, contracts=("TX", "TE", "TF"),
                                      store_path=tmp_path / "store.json")
    refs = conn.discover()          # 3 contracts x 2 fields = 6 series
    for ref in refs:
        conn.fetch(ref, "2026-06-01", "2026-07-15")
    assert calls["n"] == 1          # one poll shared across all 6 fetch() calls on this instance
