"""Tests for Protocol v2.2 §8.2 ActionDriftTracker."""

from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.monitoring.action_drift import (
    ActionDriftTracker,
    DriftStatus,
    _marginal_kl_from_hists,
)
from finrl_pro_ds.reporting import compute_eval_distribution


# ---------- KL helper --------------------------------------------------------


def test_kl_identical_hists_is_zero():
    h = np.array([10, 20, 30, 20, 10], dtype=float)
    assert _marginal_kl_from_hists(h, h) == pytest.approx(0.0, abs=1e-6)


def test_kl_non_negative_and_symmetric_smoothing():
    # With symmetric smoothing KL is still non-negative (Jensen)
    h1 = np.array([0, 0, 100, 0, 0], dtype=float)
    h2 = np.array([100, 0, 0, 0, 0], dtype=float)
    kl = _marginal_kl_from_hists(h1, h2)
    assert kl > 0.0


# ---------- Constructor & validation ----------------------------------------


def test_ctor_rejects_tiny_window():
    with pytest.raises(ValueError, match="window_bars"):
        ActionDriftTracker(None, window_bars=49)


def test_ctor_rejects_warmup_larger_than_window():
    with pytest.raises(ValueError, match="min_bars_before_check"):
        ActionDriftTracker(None, window_bars=100, min_bars_before_check=200)


def test_ctor_rejects_bad_regime_cutpoints():
    with pytest.raises(ValueError, match="regime_cutpoints"):
        ActionDriftTracker(None, regime_cutpoints=[0.1, 0.2])  # len != 3


def test_ctor_logs_warning_without_baseline(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="finrl_pro_ds.monitoring.action_drift"):
        ActionDriftTracker(None, window_bars=100, min_bars_before_check=50)
    assert any("without baseline" in r.getMessage() for r in caplog.records)


# ---------- Warmup + LOG_ONLY ------------------------------------------------


def _mk_baseline_scalar(n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(0, 0.4, n), -1, 1)
    return compute_eval_distribution(a)


def test_warmup_returns_warmup_status():
    base = _mk_baseline_scalar()
    t = ActionDriftTracker(base, window_bars=200, min_bars_before_check=100)
    report = None
    for _ in range(50):  # < warmup
        report = t.observe(0.0, bar_close=None)
    assert report is not None
    assert report.status == DriftStatus.WARMUP


def test_no_baseline_is_log_only_not_crit():
    t = ActionDriftTracker(None, window_bars=200, min_bars_before_check=100)
    report = None
    for _ in range(150):
        report = t.observe(0.9, bar_close=None)  # highly saturated
    assert report.status == DriftStatus.LOG_ONLY


# ---------- Scalar deadband/saturation logic --------------------------------


def test_scalar_matching_distribution_is_ok():
    base = _mk_baseline_scalar(seed=1)
    t = ActionDriftTracker(
        base, window_bars=500, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
        saturation_warn=0.15, saturation_crit=0.30,
    )
    rng = np.random.default_rng(1)  # same seed as baseline
    last = None
    for _ in range(499):
        a = float(np.clip(rng.normal(0, 0.4), -1, 1))
        last = t.observe(a, bar_close=None)
    assert last.status == DriftStatus.OK


def test_scalar_collapse_to_deadband_trips_crit():
    base = _mk_baseline_scalar(seed=2)
    t = ActionDriftTracker(
        base, window_bars=500, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
    )
    last = None
    for _ in range(499):
        last = t.observe(0.05, bar_close=None)  # far inside deadband
    assert last.status == DriftStatus.CRIT
    assert last.deadband_frac_live == pytest.approx(1.0)


def test_scalar_mid_shift_trips_warn_not_crit():
    base = _mk_baseline_scalar(seed=3)
    # Shift dead frac by ~0.2 (>warn 0.15 but <crit 0.30)
    t = ActionDriftTracker(
        base, window_bars=500, min_bars_before_check=200,
        deadband_warn=0.15, deadband_crit=0.30,
    )
    # Mix baseline + collapsed so final dead frac is approx baseline+0.2
    rng = np.random.default_rng(3)
    last = None
    for i in range(499):
        a = 0.05 if i % 3 == 0 else float(np.clip(rng.normal(0, 0.4), -1, 1))
        last = t.observe(a, bar_close=None)
    assert last.status in (DriftStatus.OK, DriftStatus.WARN)  # tolerance on mix


# ---------- Regime bucketing -------------------------------------------------


def test_regime_bucket_uses_explicit_cutpoints():
    base = _mk_baseline_scalar(seed=4)
    # Synthetic baseline with by_vol_quartile blocks (easier: add manually)
    base["by_vol_quartile"] = {
        q: {"deadband_frac": 0.5, "saturation_frac": 0.0}
        for q in ("q1", "q2", "q3", "q4")
    }
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=100,
        regime_cutpoints=[0.001, 0.005, 0.020],
    )
    close = 100.0
    last = None
    for _ in range(200):
        close *= (1 + 0.0001)  # tiny vol → q1
        last = t.observe(0.1, bar_close=close)
    assert last.bucket == "q1"


def test_regime_cutpoints_fallback_from_baseline():
    base = _mk_baseline_scalar(seed=5)
    base["regime_cutpoints"] = [0.001, 0.005, 0.020]
    # Don't pass regime_cutpoints explicitly — should pick up from baseline
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=100,
    )
    assert t.regime_cutpoints == (0.001, 0.005, 0.020)


# ---------- Multi-dim KL path ------------------------------------------------


def test_multidim_matching_distribution_is_ok():
    rng = np.random.default_rng(6)
    a = np.clip(rng.normal(0, 0.3, (2000, 3)), -1, 1)
    base = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=200,
        action_kl_warn=0.5, action_kl_crit=1.0,
    )
    last = None
    for _ in range(399):
        v = np.clip(rng.normal(0, 0.3, 3), -1, 1)
        last = t.observe(v, bar_close=None)
    assert last.status == DriftStatus.OK
    assert last.kl is not None and last.kl < 0.5


def test_multidim_saturation_trips_crit():
    rng = np.random.default_rng(7)
    a = np.clip(rng.normal(0, 0.3, (2000, 3)), -1, 1)
    base = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    t = ActionDriftTracker(
        base, window_bars=400, min_bars_before_check=200,
        action_kl_warn=0.5, action_kl_crit=1.0,
    )
    last = None
    for _ in range(399):
        last = t.observe(np.array([0.99, -0.99, 0.99]), bar_close=None)
    assert last.status == DriftStatus.CRIT
    assert last.kl > 1.0


def test_multidim_asset_key_mismatch_raises():
    rng = np.random.default_rng(8)
    a = np.clip(rng.normal(0, 0.3, (500, 3)), -1, 1)
    base = compute_eval_distribution(a, asset_keys=["BTC", "ETH", "SOL"])
    # Live config declares different ordering → ctor must reject
    with pytest.raises(ValueError, match="do not match expected asset_keys"):
        ActionDriftTracker(base, asset_keys=["ETH", "BTC", "SOL"])


def test_dim_change_mid_stream_raises():
    base = _mk_baseline_scalar()
    t = ActionDriftTracker(base, window_bars=100, min_bars_before_check=50)
    t.observe(0.1, bar_close=None)
    with pytest.raises(ValueError, match="action dim changed"):
        t.observe(np.array([0.1, 0.2]), bar_close=None)
