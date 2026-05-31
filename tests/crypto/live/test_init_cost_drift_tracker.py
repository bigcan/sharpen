"""Tests for the live-engine factory `_init_cost_drift_tracker`.

Covers the gates.retrain → CostDriftTracker wiring (Protocol v2 §4.5 #6):
config-cost denominator resolution (trading → env fallback, optional
slippage), threshold/window pass-through, and the no-block / no-cost
degradation paths.
"""
from __future__ import annotations

import pytest

from finrl_pro_ds.crypto.live.live_engine import _init_cost_drift_tracker


def test_no_retrain_block_returns_none():
    # No gates.retrain → nothing to gate on → tracker disabled.
    assert _init_cost_drift_tracker({"gates": {"drift": {}}}) is None
    assert _init_cost_drift_tracker({}) is None


def test_taker_only_config_cost():
    cfg = {
        "gates": {"retrain": {"cost_drift_ratio": 1.20, "cost_drift_window_trades": 100}},
        "trading": {"taker_fee": 0.0005},
    }
    t = _init_cost_drift_tracker(cfg)
    assert t is not None
    assert t.config_cost_frac == pytest.approx(0.0005)
    assert t.cost_drift_ratio == pytest.approx(1.20)
    assert t.window_trades == 100


def test_taker_plus_slippage_config_cost():
    cfg = {
        "gates": {"retrain": {"cost_drift_ratio": 1.20, "cost_drift_window_trades": 50}},
        "trading": {"taker_fee": 0.00055, "slippage_base_bps": 5.0},
    }
    t = _init_cost_drift_tracker(cfg)
    # 0.00055 + 5.0/1e4 = 0.00055 + 0.0005 = 0.00105
    assert t.config_cost_frac == pytest.approx(0.00105)
    assert t.window_trades == 50


def test_env_taker_fallback():
    cfg = {
        "gates": {"retrain": {"cost_drift_ratio": 1.20}},
        "env": {"taker_fee": 0.0007},
    }
    t = _init_cost_drift_tracker(cfg)
    assert t.config_cost_frac == pytest.approx(0.0007)
    # defaults applied when window not specified
    assert t.window_trades == 100


def test_no_cost_anywhere_is_log_only():
    cfg = {"gates": {"retrain": {"cost_drift_ratio": 1.20}}}
    t = _init_cost_drift_tracker(cfg)
    assert t is not None  # block present → tracker constructed
    assert t.config_cost_frac is None  # but LOG_ONLY (no denominator)


def test_custom_ratio_and_window_pass_through():
    cfg = {
        "gates": {"retrain": {"cost_drift_ratio": 1.35, "cost_drift_window_trades": 200}},
        "trading": {"taker_fee": 0.001},
    }
    t = _init_cost_drift_tracker(cfg)
    assert t.cost_drift_ratio == pytest.approx(1.35)
    assert t.window_trades == 200


def test_sg1_btc_live_scenario():
    # The actual sg1-btc live config shape: stale 5.0 bps taker (no slippage
    # key), default 1.20x / 100-trade gates.retrain. config_cost_frac = 0.0005
    # so the realized fee+slippage of ~10.5 bps would read ~2.1x → FIRED
    # (P10-04: the trigger that would have caught the sim->live gap).
    cfg = {
        "gates": {"retrain": {"cost_drift_ratio": 1.20, "cost_drift_window_trades": 100}},
        "trading": {"initial_balance": 50000.0, "taker_fee": 0.0005},
    }
    t = _init_cost_drift_tracker(cfg)
    assert t.config_cost_frac == pytest.approx(0.0005)
    assert t.cost_drift_ratio == pytest.approx(1.20)
