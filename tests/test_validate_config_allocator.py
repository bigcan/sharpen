"""Validator tests for the cross-asset allocator (Phase 3, S553-cont-33).

Covers ADR-4 (env-type-conditional HPO objective allow-list), the yfinance
runtime-fetch data branch, the allocator gross-exposure safety bound, and that
the shipped config passes both --stage hpo and --stage wf while single-instrument
configs remain strictly bound to profit_factor (BUG-01).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    _training_budget_multiplicity,
    check_data_manifest,
    check_hpo,
    check_max_leverage_bounds,
    validate,
)

ALLOCATOR_CFG = ROOT / "configs" / "cross_asset_momentum.yaml"
ALLOCATOR_RETRY_CFG = ROOT / "configs" / "cross_asset_momentum_retry.yaml"


def _hpo_cfg(env_type, objective):
    return {
        "env": ({"type": env_type} if env_type else {}),
        "hpo": {"objective": objective, "trials": 40, "steps_per_trial": 500000},
        "gates": {"entropy_floor": 0.01, "q_div_max": 100.0, "action_sat_max": 0.9},
    }


# --------------------------------------------------------------------------- #
# ADR-4: env-type-conditional objective allow-list
# --------------------------------------------------------------------------- #

def test_allocator_objective_sharpe_passes():
    r = ValidationResult()
    check_hpo(_hpo_cfg("multi_asset_allocator", "sharpe"), r)
    assert not r.failures, r.failures


def test_allocator_objective_sortino_passes():
    r = ValidationResult()
    check_hpo(_hpo_cfg("multi_asset_allocator", "sortino"), r)
    assert not r.failures, r.failures


def test_allocator_objective_profit_factor_fails():
    """The allocator must NOT use profit_factor (low discrimination on a daily book)."""
    r = ValidationResult()
    check_hpo(_hpo_cfg("multi_asset_allocator", "profit_factor"), r)
    assert any("multi_asset_allocator must use" in f for f in r.failures), r.failures


def test_single_instrument_objective_sharpe_still_fails_bug01():
    """BUG-01 preserved: non-allocator configs are strictly profit_factor."""
    r = ValidationResult()
    check_hpo(_hpo_cfg(None, "sharpe"), r)
    assert any("must be 'profit_factor'" in f for f in r.failures), r.failures


def test_single_instrument_profit_factor_passes():
    r = ValidationResult()
    check_hpo(_hpo_cfg("v7", "profit_factor"), r)
    assert not r.failures, r.failures


# --------------------------------------------------------------------------- #
# Allocator gross-exposure safety bound (MARGIN-CFG)
# --------------------------------------------------------------------------- #

def test_allocator_gross_exposure_in_bounds_ok():
    r = ValidationResult()
    check_max_leverage_bounds({"env": {"type": "multi_asset_allocator", "max_gross_exposure": 3.0}}, r)
    assert not r.failures, r.failures


def test_allocator_gross_exposure_out_of_bounds_fails():
    r = ValidationResult()
    check_max_leverage_bounds({"env": {"type": "multi_asset_allocator", "max_gross_exposure": 9.0}}, r)
    assert any("max_gross_exposure" in f and "out of bounds" in f for f in r.failures), r.failures


def test_gross_exposure_bound_only_for_allocator():
    """A non-allocator config with a large max_gross_exposure is NOT bound-checked
    (single-instrument configs use max_leverage, not this key)."""
    r = ValidationResult()
    check_max_leverage_bounds({"env": {"max_gross_exposure": 9.0}}, r)
    assert not any("max_gross_exposure" in f for f in r.failures), r.failures


# --------------------------------------------------------------------------- #
# yfinance runtime-fetch data branch
# --------------------------------------------------------------------------- #

def test_yfinance_source_skips_manifest():
    r = ValidationResult()
    check_data_manifest({"data": {"source": "yfinance_etf", "frequency": "1d"}}, "hpo", r)
    assert not r.failures, r.failures
    assert any("runtime fetch" in p for p in r.passed)


def test_yfinance_source_requires_frequency():
    r = ValidationResult()
    check_data_manifest({"data": {"source": "yfinance_etf"}}, "hpo", r)
    assert any("frequency missing" in f for f in r.failures), r.failures


# --------------------------------------------------------------------------- #
# Training-budget multiplicity (Protocol v2.5.1 §3.5) — allocator branch.
# The gate used to return None for non-crypto, so the 500k/1260-bar = 397x
# Stage-1 overfit slipped validation (S553-cont-34). These are the regression
# tests that keep that hole closed.
# --------------------------------------------------------------------------- #

def _alloc_mult_cfg(train_bars=1260):
    return {
        "env": {"type": "multi_asset_allocator"},
        "walk_forward": {"train_bars": train_bars},
        "data": {"frequency": "1d"},
    }


def test_allocator_multiplicity_uses_walk_forward_train_bars():
    m = _training_budget_multiplicity(_alloc_mult_cfg(), 500000)
    assert m is not None
    mult, bars, _basis = m
    assert bars == 1260
    assert abs(mult - 500000 / 1260) < 1e-9


def test_allocator_multiplicity_missing_train_bars_skips():
    cfg = {"env": {"type": "multi_asset_allocator"}, "walk_forward": {}}
    assert _training_budget_multiplicity(cfg, 500000) is None


def test_non_crypto_non_allocator_still_skips():
    """Session-bound assets without a modeled calendar stay skipped (no false mult)."""
    cfg = {"env": {"type": "v7"}, "features": {"asset_class": "gold"}}
    assert _training_budget_multiplicity(cfg, 500000) is None


def test_original_allocator_config_fails_hpo_multiplicity():
    """The exact config that slipped (500k steps / 1260 bars = 397x) must now
    FAIL the 50x REJECT cliff at --stage hpo."""
    r = validate(ALLOCATOR_CFG, "hpo")
    assert any("REJECT cliff" in f for f in r.failures), r.failures


def test_retry_allocator_config_passes_hpo_in_band():
    """The anti-overfit retry (50k steps = ~39.7x) sits inside [15,40] and passes."""
    r = validate(ALLOCATOR_RETRY_CFG, "hpo")
    assert not r.failures, r.failures
    assert any("budget multiplicity" in p and "productive band" in p for p in r.passed), r.passed


# --------------------------------------------------------------------------- #
# End-to-end: the active (retry) config validates at both gated stages
# --------------------------------------------------------------------------- #

def test_retry_allocator_config_passes_wf():
    r = validate(ALLOCATOR_RETRY_CFG, "wf")
    assert not r.failures, r.failures


def test_shipped_allocator_config_passes_wf():
    """wf stage does not re-check the per-trial HPO budget; the original config
    remains valid for WF-stage replay."""
    r = validate(ALLOCATOR_CFG, "wf")
    assert not r.failures, r.failures
