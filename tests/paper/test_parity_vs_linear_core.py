"""KEYSTONE: the rung-1 forward path reproduces evaluate_linear_core (parity ≈ 0).

The paper soak's whole value is sim↔live fidelity (ADR-7). The executor shadows
``evaluate_linear_core`` VERBATIM (operator decision 2026-06-13: daily vol-rescale),
so replaying historical bars through SimFillEngine + PaperState must reproduce the
env's weights and equity curve to ≈0. Any non-zero ``weight_l1_drift`` /
``daily_return_te_bps`` here is a real forward-path bug (data, calendar, look-ahead,
scheduler), NOT an accounting artifact — exactly the failure mode the soak exists to
catch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sharpen.envs.allocator_factory import linear_core_weights
from sharpen.paper import (
    ParityHarness,
    evaluate_paper_soak_gates,
    serialize_verdict,
)

ROOT = Path(__file__).resolve().parents[2]
OHLCV_CACHE = ROOT / "results" / "xsec_momentum" / "ohlcv_daily.parquet"


def _assert_parity_zero(rep) -> None:
    assert rep.weight_l1_drift_max < 1e-9, f"weight drift {rep.weight_l1_drift_max}"
    assert rep.daily_return_te_bps_max < 1e-6, f"te_bps {rep.daily_return_te_bps_max}"
    assert rep.missed_rebalances == 0
    assert abs(rep.cost_drift_ratio - 1.0) < 1e-9, f"cost_drift {rep.cost_drift_ratio}"


def test_parity_synthetic_18_asset(cfg, gates_cfg, arrays_18):
    """Parity ≈ 0 on realistic-shaped 18-asset data; soak gates evaluate to PASS."""
    h = ParityHarness(cfg)
    live, sim = h.run(arrays_18)
    rep = h.compare(live, sim)

    _assert_parity_zero(rep)
    # the full equity curve must be byte-faithful to the env, not just aggregates.
    np.testing.assert_allclose(live.equity_curve, sim["equity_curve"], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(live.weights, sim["weights"])

    # THE keystone claim is the PARITY group (forward path == oracle). risk/drift on
    # synthetic random-sign conviction are data-dependent and not the point here.
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)
    assert verdict["groups"]["parity"]["status"] == "PASS"
    assert verdict["overall_status"] in {"PASS", "FAIL", "REVIEW", "UNKNOWN_INSUFFICIENT_DATA"}
    assert verdict["rebalance_cadence"] == "monthly"
    # the full-span month-end calendar is genuinely satisfied in a full replay ⇒ 0 missed.
    assert rep.missed_rebalances == 0


def test_replay_handles_short_weights_gracefully(cfg, gates_cfg, arrays_18):
    """A circuit-broken / early-terminated oracle yields FEWER weight rows than bars; the
    replay must degrade (cover the prefix + flag coverage_incomplete), NOT crash on the old
    hard ``assert W.shape == (T-1, N)`` (P10-03). compare()/gates stay well-defined."""
    h = ParityHarness(cfg)
    sim = h.sim_oracle(arrays_18)
    full = np.asarray(sim["weights"], dtype=np.float64)
    short = full[: len(full) // 2]                       # simulate an early circuit-break
    live = h._replay(arrays_18, short, fill_engine=None)
    assert live.coverage_incomplete is True
    assert live.n_steps == len(short)
    # compare() takes the min length — no crash, parity still computable over the prefix.
    rep = h.compare(live, {**sim, "weights": full})
    assert rep.n_steps == len(short)
    v = evaluate_paper_soak_gates(live, rep, gates_cfg)
    assert v["summary"]["coverage_incomplete"] is True
    # A full-length replay is (still) full coverage.
    live_full = h._replay(arrays_18, full, fill_engine=None)
    assert live_full.coverage_incomplete is False


def test_parity_verdict_serializes(tmp_path, cfg, gates_cfg, arrays_18):
    """The step-3 gate is 'evaluate AND serialize' — the verdict must round-trip JSON."""
    h = ParityHarness(cfg)
    live, sim = h.run(arrays_18)
    rep = h.compare(live, sim)
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)

    out = serialize_verdict(verdict, tmp_path / "paper_soak_verdict.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["overall_status"] in {"PASS", "FAIL", "REVIEW", "UNKNOWN_INSUFFICIENT_DATA"}
    assert loaded["groups"]["parity"]["checks"]["weight_l1_drift"]["status"] == "PASS"
    assert loaded["groups"]["parity"]["checks"]["daily_return_te_bps"]["status"] == "PASS"


def test_forward_prefix_consistency(cfg, arrays_18):
    """LEAK-2 corollary: the frozen-core weights on a window truncated at bar m equal
    the full-window weights over [0, m). This is what lets the live incremental executor
    recompute on a growing window and still match the backtest (no boundary look-ahead).
    """
    W_full = linear_core_weights(arrays_18, cfg)
    T = len(arrays_18["timestamps"])
    for m in (260, 455, 700):                       # mixed mid-month / boundary cutoffs
        trunc = {k: (v if k == "assets" else v[:m + 1]) for k, v in arrays_18.items()}
        W_trunc = linear_core_weights(trunc, cfg)   # (m, N)
        assert W_trunc.shape == (m, arrays_18["price_ary"].shape[1])
        np.testing.assert_array_equal(W_trunc, W_full[:m])
    del T


def test_missed_rebalances_detects_a_dropped_month_end():
    """With the full-span month-end calendar (not the self-derived subset), dropping a true
    month-end from the processed set is now DETECTABLE — the HARD gate can actually fail
    (P3-02/P8-02/P10-04; previously it was structurally pinned at 0)."""
    import pandas as pd

    ts = (pd.bdate_range("2020-01-01", periods=170).asi8 // 10**9).astype("int64")  # ~8 months
    expected = ParityHarness._true_month_end_ts(ts)
    assert len(expected) >= 6
    # processed every bar (incl. every true month-end) ⇒ 0 missed.
    assert ParityHarness._missed_rebalances(ts, expected_ts=expected) == 0
    # drop one true month-end from the processed set ⇒ exactly 1 missed (the gate CAN fail).
    dropped = ts[ts != expected[3]]
    assert ParityHarness._missed_rebalances(dropped, expected_ts=expected) == 1


@pytest.mark.skipif(not OHLCV_CACHE.exists(),
                    reason="cached real ETF OHLCV absent (run cross_asset_loader)")
def test_parity_real_etf_data(cfg, gates_cfg):
    """Parity ≈ 0 on the real validated ETF universe (exercises NaN/warmup/zero-price
    and extreme-participation paths the synthetic data does not)."""
    try:
        from sharpen.data import cross_asset_loader as loader
        data = loader.load_cross_asset_data(cfg)            # offline: uses the cached clean parquet
        close = data["close"]
        arrays = loader.build_allocator_arrays(
            data["signals"], close, data["volume"], data["assets"],
            close.index[0], close.index[-1], lookbacks=data["lookbacks"],
        )
    except Exception as e:                                  # pragma: no cover - data/env dependent
        pytest.skip(f"could not build real-ETF arrays from cache: {e}")

    h = ParityHarness(cfg)
    live, sim = h.run(arrays)
    rep = h.compare(live, sim)
    _assert_parity_zero(rep)
    np.testing.assert_allclose(live.equity_curve, sim["equity_curve"], atol=1e-5, rtol=0)
    # the pre-registered parity gate must also pass on real data.
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)
    assert verdict["groups"]["parity"]["status"] == "PASS", verdict["groups"]["parity"]
