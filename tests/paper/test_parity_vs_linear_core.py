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

from finrl_pro_ds.envs.allocator_factory import linear_core_weights
from finrl_pro_ds.paper import (
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
    assert verdict["overall_status"] in {"PASS", "FAIL", "REVIEW"}
    assert verdict["rebalance_cadence"] == "monthly"


def test_parity_verdict_serializes(tmp_path, cfg, gates_cfg, arrays_18):
    """The step-3 gate is 'evaluate AND serialize' — the verdict must round-trip JSON."""
    h = ParityHarness(cfg)
    live, sim = h.run(arrays_18)
    rep = h.compare(live, sim)
    verdict = evaluate_paper_soak_gates(live, rep, gates_cfg)

    out = serialize_verdict(verdict, tmp_path / "paper_soak_verdict.json")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["overall_status"] in {"PASS", "FAIL", "REVIEW"}
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


@pytest.mark.skipif(not OHLCV_CACHE.exists(),
                    reason="cached real ETF OHLCV absent (run cross_asset_loader)")
def test_parity_real_etf_data(cfg, gates_cfg):
    """Parity ≈ 0 on the real validated ETF universe (exercises NaN/warmup/zero-price
    and extreme-participation paths the synthetic data does not)."""
    try:
        from finrl_pro_ds.data import cross_asset_loader as loader
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
