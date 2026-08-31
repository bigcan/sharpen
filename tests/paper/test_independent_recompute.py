"""STEP-4 KEYSTONE: the independent live-recompute parity variant makes
``weight_l1_drift`` LOAD-BEARING (Tier-2 audit 2026-06-14: P3-01 / P10-01 / P8-03 / P10-08).

The default :meth:`ParityHarness.run` feeds the replay the sim oracle's OWN weights, so
``weight_l1_drift`` is 0 *by construction* — it validates ACCOUNTING (book == env on
identical weights), NOT the forward DATA/WEIGHT path. :meth:`run_independent_recompute`
reconstructs the held-conviction series INDEPENDENTLY on a growing live-fetched window (the
live scheduler's view) and drives the same env with it. These tests pin both halves of the
claim the audit demands before rung-2 capital:

  1. a CORRECT forward assembly reproduces the oracle exactly (drift 0 is legitimate), and
  2. the ESCAPE exists — a forward-path look-ahead / P2-01 in-progress-tail trap that is
     INVISIBLE to ``run`` is CAUGHT as non-zero ``weight_l1_drift`` by the variant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sharpen.envs.allocator_factory import drive_with_conviction, monthly_rebal_conviction
from sharpen.paper import ParityHarness, evaluate_paper_soak_gates

ROOT = Path(__file__).resolve().parents[2]
OHLCV_CACHE = ROOT / "results" / "xsec_momentum" / "ohlcv_daily.parquet"


def _lookahead_fn(arrays, t):
    """A deliberately-broken forward reader: at the month-end rebalance it peeks ONE bar
    into the future (``conv[t+1]``) instead of using the confirmed month-end ``conv[t]``.
    A genuine LEAK-2 look-ahead in the growing-window assembly — must be caught."""
    conv = np.asarray(arrays["conviction_ary"], dtype=np.float64)
    return conv[min(t + 1, len(conv) - 1)]


def _mid_month_index(ts: np.ndarray, start: int = 200) -> int:
    """First index >= ``start`` that is NOT a true month-end (an in-progress mid-month bar)."""
    me = {int(x) for x in ParityHarness._true_month_end_ts(ts)}
    for i in range(start, len(ts)):
        if int(ts[i]) not in me:
            return i
    raise AssertionError("no mid-month bar found")


# --------------------------------------------------------------------------- #
def test_independent_recompute_safe_reproduces_oracle(cfg, arrays_18):
    """SAFE growing-window assembly == the batch oracle, byte-for-byte. Proves the
    forward weight-derivation path is correct AND that a drift of 0 here is meaningful
    (not tautological): it came from an INDEPENDENT recompute, not a self-feed."""
    h = ParityHarness(cfg)
    live, sim = h.run_independent_recompute(arrays_18)
    rep = h.compare(live, sim)

    assert rep.weight_l1_drift_max < 1e-9, f"weight drift {rep.weight_l1_drift_max}"
    assert rep.daily_return_te_bps_max < 1e-6, f"te_bps {rep.daily_return_te_bps_max}"
    assert rep.missed_rebalances == 0
    # end-to-end byte fidelity: independently-assembled weights + equity == the oracle.
    np.testing.assert_array_equal(live.weights, sim["weights"])
    np.testing.assert_allclose(live.equity_curve, sim["equity_curve"], atol=1e-6, rtol=0)


def test_assembled_conviction_equals_batch_monthly_rebal(cfg, arrays_18):
    """The independent growing-window conviction assembly reproduces the batch
    ``monthly_rebal_conviction`` element-for-element when the SAFE reader is used — the
    invariant that makes the safe drift legitimately 0."""
    h = ParityHarness(cfg)
    conv_live = h._assemble_forward_conviction(arrays_18)
    conv_batch = monthly_rebal_conviction(arrays_18["timestamps"], arrays_18["conviction_ary"])
    np.testing.assert_array_equal(conv_live, conv_batch)


def test_independent_variant_catches_what_run_cannot(cfg, arrays_18):
    """THE audit point (P3-01/P10-01): a forward-path look-ahead bug is INVISIBLE to the
    self-feeding ``run`` (drift 0 regardless) but the independent recompute SURFACES it as
    non-zero ``weight_l1_drift`` — i.e. the metric is now load-bearing."""
    h = ParityHarness(cfg)

    # run() self-feeds the oracle's own weights -> drift 0 no matter the forward path.
    live_run, sim = h.run(arrays_18)
    assert h.compare(live_run, sim).weight_l1_drift_max < 1e-9

    # the SAME machinery with an independent look-ahead reader catches the bug.
    live_bug, sim2 = h.run_independent_recompute(arrays_18, conviction_fn=_lookahead_fn)
    rep_bug = h.compare(live_bug, sim2)
    assert rep_bug.weight_l1_drift_max > 1e-3, (
        f"look-ahead forward bug not caught (drift {rep_bug.weight_l1_drift_max})")


def test_p201_in_progress_tail_trap_is_caught(cfg, arrays_18):
    """The P2-01 X2-class landmine: a daily-incremental reader that takes
    ``monthly_rebal_conviction(window)[-1]`` rebalances on the in-progress bar every day,
    i.e. feeds RAW daily conviction (a different, unvalidated strategy). Driving the same
    env with it diverges from the monthly oracle -> caught by weight drift."""
    h = ParityHarness(cfg)
    sim = h.sim_oracle(arrays_18)
    ts = np.asarray(arrays_18["timestamps"], dtype=np.int64)
    conv = np.asarray(arrays_18["conviction_ary"], dtype=np.float64)

    # Mechanism: at a MID-MONTH bar the truncated-window tail [-1] is the raw in-progress
    # conviction (the contamination), NOT the held month-end value the oracle uses.
    m = _mid_month_index(ts)
    tail = monthly_rebal_conviction(ts[: m + 1], conv[: m + 1])[-1]
    np.testing.assert_array_equal(tail, conv[m])                      # tail == raw daily (trap)
    held_monthly = monthly_rebal_conviction(ts, conv)[m]
    assert not np.array_equal(conv[m], held_monthly), "fixture has no mid-month divergence"

    # Effect: feeding the daily-tail series (== raw daily conv) into the SAME drive moves
    # the weights vs the monthly oracle -> the variant catches it.
    w_trap = drive_with_conviction(arrays_18, cfg, conv)["weights"]
    drift = np.abs(w_trap - np.asarray(sim["weights"], dtype=np.float64)).sum(axis=1).max()
    assert drift > 1e-3, f"daily in-progress-tail trap not caught (drift {drift})"


def test_independent_recompute_parity_gate_passes_on_safe_path(cfg, gates_cfg, arrays_18):
    """The pre-registered ``paper_soak.parity`` group still PASSES on the load-bearing
    safe forward path — so the gate is meaningful (it can fail, per the bug tests above)
    yet green when the forward assembly is correct."""
    h = ParityHarness(cfg)
    live, sim = h.run_independent_recompute(arrays_18)
    rep = h.compare(live, sim)
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)
    assert verdict["groups"]["parity"]["status"] == "PASS", verdict["groups"]["parity"]


@pytest.mark.skipif(not OHLCV_CACHE.exists(),
                    reason="cached real ETF OHLCV absent (run cross_asset_loader)")
def test_independent_recompute_real_etf_growing_window(cfg, gates_cfg):
    """The independent growing-window recompute reproduces the oracle on the REAL 18-asset
    validated universe (5138 bars) — the 'growing live-fetched window' the audit names,
    exercising NaN/warmup/zero-price/extreme-participation paths the synthetic data misses.
    A drift of 0 here is load-bearing: the conviction was re-assembled independently, not
    fed from the oracle."""
    try:
        from sharpen.data import cross_asset_loader as loader
        data = loader.load_cross_asset_data(cfg)             # offline: cached clean parquet
        close = data["close"]
        arrays = loader.build_allocator_arrays(
            data["signals"], close, data["volume"], data["assets"],
            close.index[0], close.index[-1], lookbacks=data["lookbacks"],
        )
    except Exception as e:                                    # pragma: no cover - data/env dependent
        pytest.skip(f"could not build real-ETF arrays from cache: {e}")

    h = ParityHarness(cfg)
    live, sim = h.run_independent_recompute(arrays)
    rep = h.compare(live, sim)
    assert rep.weight_l1_drift_max < 1e-9, f"weight drift {rep.weight_l1_drift_max}"
    assert rep.daily_return_te_bps_max < 1e-6, f"te_bps {rep.daily_return_te_bps_max}"
    assert rep.missed_rebalances == 0
    np.testing.assert_allclose(live.equity_curve, sim["equity_curve"], atol=1e-5, rtol=0)
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)
    assert verdict["groups"]["parity"]["status"] == "PASS", verdict["groups"]["parity"]
