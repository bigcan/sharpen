"""crucible-v16.0: POWER-LORD-01 wired — the substrate-power guard reads the calibration curve at the
LORD++ level the tick will actually spend, not at a fresh account's first level.

Pre-v16 the stamp was always the fresh reading, so a substrate whose persistent account had decayed far
below the fresh level was still judged powered. Each tripwire below FAILS on that behaviour.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from sharpen.crucible import OrchestratorStore, run_orchestrator_tick
from sharpen.crucible.orchestrator import orchestrator as orch_mod
from sharpen.crucible.orchestrator.fdr import OnlineFDR
from sharpen.crucible.orchestrator.substrate import (
    LORD_BEYOND_GRID,
    PowerGuard,
    _power_holdout_bars,
    lord_depth_for_level,
    stamp_substrate_power,
)

ROOT = Path(__file__).resolve().parents[2]


def _level_after(k: int) -> float:
    acct = OnlineFDR(alpha=0.10)
    for _ in range(k):
        acct.observe(is_discovery=False)
    return acct.next_level()


def _sweep(holdout_bars: int, mde_fresh: float = 0.4, mde_deep: float = 0.9) -> dict:
    return {"cross_sectional": {"mde_sweep": {
        "candidate_type": "cross_sectional",
        "rows": [{"holdout_bars": holdout_bars, "mde_realized_delta_sr": mde_fresh,
                  "mde_by_lord_tests": {"0": mde_fresh, "8": mde_deep}}],
        "lord_tests_grid": [0, 8],
        "lord_levels": {"0": _level_after(0), "8": _level_after(8)}}}}


# ------------------------------------------------------------------ selection by level
def test_depth_is_the_loosest_measured_level_not_looser_than_the_live_one() -> None:
    sw = _sweep(100)["cross_sectional"]
    assert lord_depth_for_level(sw, _level_after(0)) == 0          # exactly the fresh level
    assert lord_depth_for_level(sw, 1.0) == 0                      # replenished above fresh: still 0
    assert lord_depth_for_level(sw, _level_after(3)) == 8          # between: round to the TIGHTER row
    assert lord_depth_for_level(sw, _level_after(8)) == 8
    assert lord_depth_for_level(sw, _level_after(20)) == LORD_BEYOND_GRID
    assert lord_depth_for_level({"mde_sweep": {"rows": []}}, 0.01) is None


def test_live_stamp_reads_the_row_for_the_live_level_and_refuses_beyond_the_grid() -> None:
    hb = _power_holdout_bars(700, 0.25)
    sw = _sweep(hb)
    fresh = stamp_substrate_power(700, 0.25, sw, "h", candidate_types=("cross_sectional",))
    live = stamp_substrate_power(700, 0.25, sw, "h", candidate_types=("cross_sectional",),
                                 live_level=_level_after(5))
    deep = stamp_substrate_power(700, 0.25, sw, "h", candidate_types=("cross_sectional",),
                                 live_level=_level_after(50))
    assert fresh.implied_mde_delta_sr == pytest.approx(0.4)
    assert live.implied_mde_delta_sr == pytest.approx(0.9) and live.interp_mode.endswith(":lord8")
    assert math.isinf(deep.implied_mde_delta_sr) and deep.interp_mode == "unmeasured_lord_beyond_grid"


def test_a_sweep_without_a_level_family_degrades_loudly_to_the_fresh_reading() -> None:
    hb = _power_holdout_bars(700, 0.25)
    sw = {"overlay": {"mde_sweep": {"candidate_type": "overlay",
                                    "rows": [{"holdout_bars": hb, "mde_realized_delta_sr": 0.4}]}}}
    st = stamp_substrate_power(700, 0.25, sw, "h", candidate_types=("overlay",),
                               live_level=_level_after(30))
    assert st.implied_mde_delta_sr == pytest.approx(0.4) and st.interp_mode.endswith(":lord_unmeasured")


# ------------------------------------------------------------------ the guard decides at the live level
def _tick(tmp_path, *, tests_already: int, monkeypatch=None):
    from test_v15_0_fixes import GATES, T, _FixedProposer, _overlay, _substrate

    hb = _power_holdout_bars(T, 0.25)
    sweep = _sweep(hb)
    props = [_overlay("a", "macro:regime"), _overlay("b", "delta(macro:regime, 20)"),
             _overlay("c", "ts_mean(macro:regime, 40)")]
    sub = _substrate(tmp_path, _FixedProposer(props))
    fresh = stamp_substrate_power(T, 0.25, sweep, "h", candidate_types=("cross_sectional",))
    inner = sub.prepare
    sub.prepare = lambda: orch_mod._dc_replace(inner(), power=fresh)
    store = OrchestratorStore(tmp_path / "orch.db")
    if tests_already:
        store.save_fdr("syn", OnlineFDR(alpha=sub.fdr_alpha, w0=sub.fdr_w0, num_tests=tests_already))
    guard = PowerGuard(enabled=True, ceiling=0.5, action="refuse", sweep=sweep, sweep_hash="h")
    res = run_orchestrator_tick(substrates=[sub], store=store, gates_path=GATES,
                                tick_ts="2026-09-30T00:00:00+00:00", power_gate=guard)
    return res.outcomes[0]


def test_a_decayed_account_is_refused_although_the_fresh_stamp_passes(tmp_path) -> None:
    o = _tick(tmp_path, tests_already=8)
    assert not o.mined and "UNDERPOWERED" in o.reason


def test_a_fresh_account_still_mines(tmp_path) -> None:
    o = _tick(tmp_path, tests_already=0)
    assert o.mined


def test_the_rejection_class_reads_the_batch_tightest_level(tmp_path, monkeypatch) -> None:
    """3 fresh specs on a fresh account: the guard reads test 1's level (MDE 0.4, mines), but the last
    spec of the batch is tested at test 3's level, which only the deep row covers (MDE 0.9)."""
    seen: dict = {}
    real = orch_mod.run_hypothesis_loop

    def spy(**kw):
        seen["substrate_mde"] = kw["substrate_mde"]
        return real(**kw)

    monkeypatch.setattr(orch_mod, "run_hypothesis_loop", spy)
    o = _tick(tmp_path, tests_already=0)
    assert o.mined
    assert seen["substrate_mde"] == pytest.approx(0.9)
