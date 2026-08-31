"""TAILWIND (momentum + defensive/BAB) end-to-end executor wiring tests.

Proves the signal-dispatched two-sleeve loader + TwoSleeveExecutor run the TAILWIND
composition (TSMOM engine + within-class BAB hedge) the same way they run the validated
momentum + rates-carry book: fund-of-funds drive -> risk-parity combine -> union replay,
rung-1 parity ~0 by construction. Synthetic arrays only (no network).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.paper import TwoSleeveExecutor, evaluate_paper_soak_gates

ROOT = Path(__file__).resolve().parents[2]
ANN = 252

_NAMES = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD",
          "GLD", "SLV", "DBC", "USO", "DBA", "UUP", "FXE", "FXY", "FXB", "FXA"]


def _causal_vol(price: np.ndarray, vol_window: int = 63) -> np.ndarray:
    T, n = price.shape
    rets = np.full((T, n), np.nan)
    rets[1:] = price[1:] / price[:-1] - 1.0
    vol = np.zeros((T, n))
    for t in range(T):
        if t - 1 >= vol_window:
            vol[t] = rets[t - vol_window:t].std(axis=0, ddof=1) * np.sqrt(ANN)
    return vol


def _tailwind_bundle(*, T: int = 780, seed: int = 21, volume: float = 5e8) -> dict:
    """momentum(18) + defensive(18) + union(18), one shared price matrix (single-fetch
    alignment). Both sleeves trade the full union; the defensive conviction is a synthetic
    within-class BAB signal in [-1,1] (values are not read by the linear drive's obs)."""
    rng = np.random.default_rng(seed)
    ts = (pd.bdate_range("2014-01-01", periods=T).asi8 // 10**9).astype(np.int64)
    n = len(_NAMES)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.010, (T, n)), axis=0)
    vol = _causal_vol(price)
    dvol = np.full((T, n), float(volume), dtype=np.float64)

    momentum = {
        "price_ary": price.copy(),
        "tech_ary": rng.normal(0, 1, (T, n * 7)).astype(np.float32),
        "vol_ary": vol.copy(),
        "carry_ary": np.zeros((T, n), dtype=np.float64),
        "volume_ary": dvol.copy(),
        "timestamps": ts,
        "conviction_ary": np.sign(rng.normal(0, 1, (T, n))),
        "assets": list(_NAMES),
    }
    conv_def = np.tanh(rng.normal(0, 1, (T, n)))
    defensive = {
        "price_ary": price.copy(),
        "tech_ary": conv_def.astype(np.float32),        # (T,n) tech_dim=1 placeholder
        "vol_ary": vol.copy(),
        "carry_ary": np.zeros((T, n), dtype=np.float64),
        "volume_ary": dvol.copy(),
        "timestamps": ts,
        "conviction_ary": conv_def,
        "assets": list(_NAMES),
    }
    union = {
        "price_ary": price.copy(),
        "volume_ary": dvol.copy(),
        "carry_ary": np.zeros((T, n), dtype=np.float64),
        "timestamps": ts,
        "assets": list(_NAMES),
    }
    return {"momentum": momentum, "defensive": defensive, "union": union}


@pytest.fixture(scope="module")
def tailwind_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "tailwind_v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def tw_bundle() -> dict:
    return _tailwind_bundle()


def test_executor_selects_momentum_and_defensive(tailwind_cfg):
    """The executor resolves its allocator sleeves from the tailwind config: the ENGINE
    (momentum) + the HEDGE (defensive) — NOT rates_carry."""
    ex = TwoSleeveExecutor(tailwind_cfg)
    assert set(ex.sleeve_names) == {"momentum", "defensive"}


def test_tailwind_batch_parity_is_zero(tailwind_cfg, tw_bundle):
    """Rung-1 batch book: the combined momentum+defensive weights replay to parity ~0
    (accounting tautology), exactly like the momentum+rates-carry book."""
    ex = TwoSleeveExecutor(tailwind_cfg)
    live, oracle = ex.run(tw_bundle)
    parity = ex.compare(live, oracle)
    assert parity.weight_l1_drift_max < 1e-9
    assert parity.daily_return_te_bps_max < 1e-6
    assert parity.missed_rebalances == 0
    assert live.weights.shape[1] == len(_NAMES)             # booked over the 18-asset union


def test_tailwind_forward_recompute_parity_is_zero(tailwind_cfg, tw_bundle):
    """The load-bearing forward-path: independent per-sleeve recompute reproduces the oracle
    on the safe path (drift ~0) for BOTH the tsmom and the defensive sleeve."""
    ex = TwoSleeveExecutor(tailwind_cfg)
    live, oracle = ex.run_independent_recompute(tw_bundle)
    parity = ex.compare(live, oracle)
    assert parity.weight_l1_drift_max < 1e-9
    assert parity.daily_return_te_bps_max < 1e-6


def test_tailwind_risk_calibration_cross_check_matches(tailwind_cfg, tw_bundle):
    """The paper_soak risk group's calibrated_for_sleeves must equal the executor's actual
    composition, so a mis-scoped kill cannot be silently trusted (P8-07 cross-check)."""
    gates = yaml.safe_load(
        (ROOT / "configs" / "tailwind_v1.gates.yaml").read_text(encoding="utf-8"))
    assert gates["paper_soak"]["risk"]["calibrated_for_sleeves"] == ["momentum", "defensive"]
    ex = TwoSleeveExecutor(tailwind_cfg)
    live, _ = ex.run_independent_recompute(tw_bundle)
    parity = ex.compare(live, ex.sim_oracle(tw_bundle)[0])
    verdict = evaluate_paper_soak_gates(
        live, parity, gates,
        executor_sleeves=sorted(tailwind_cfg["sleeves"].keys()))
    # The risk group must NOT fail on a composition mismatch (it may still flag on other axes).
    risk = verdict["groups"].get("risk", {})
    assert "composition" not in str(risk.get("detail", "")).lower() or risk.get("status") != "FAIL"
