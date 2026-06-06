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
    check_data_manifest,
    check_hpo,
    check_max_leverage_bounds,
    validate,
)

ALLOCATOR_CFG = ROOT / "configs" / "cross_asset_momentum.yaml"


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
# End-to-end: the shipped config validates at both gated stages
# --------------------------------------------------------------------------- #

def test_shipped_allocator_config_passes_hpo():
    r = validate(ALLOCATOR_CFG, "hpo")
    assert not r.failures, r.failures


def test_shipped_allocator_config_passes_wf():
    r = validate(ALLOCATOR_CFG, "wf")
    assert not r.failures, r.failures
