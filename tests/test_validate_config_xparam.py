"""Tests for XPARAM-01..12 cross-parameter invariants in validate_config.py.

Ported 2026-07-29 from the /audit skill addendum, where these lived only as a
checklist. Per the audit skill's own rule, a load-bearing correctness property
with no executing test IS the bug, and each needs a negative tripwire: every
invariant below has a violation case that must fail, so reintroducing the bug
breaks the suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_xparam,
)


def _run(cfg: dict) -> ValidationResult:
    r = ValidationResult()
    check_xparam(cfg, r)
    return r


def _failed(r: ValidationResult, xid: str) -> bool:
    return any(f"XPARAM-{xid}" in f for f in r.failures)


def _warned(r: ValidationResult, xid: str) -> bool:
    return any(f"XPARAM-{xid}" in w for w in r.warnings)


# A minimal config that satisfies every applicable invariant. Key paths mirror
# configs/gmgp1_sac_gc_15min.yaml (the GMGP1 V7 reference).
def _good() -> dict:
    return {
        "features": {"scales": [15, 60, 240], "features_per_scale": 8},
        "env": {
            "mdp_version": "v7",
            "features_per_scale": 8,
            "deadband_threshold": 0.25,
            "scales": [15, 60, 240],
            "reward": {"dsr_eta": 0.001, "dsr_scale": 1.0},
        },
        "network": {"private_dim": 5, "scale_encoder": {"input_size": 8}},
        "agents": {"sac": {"buffer_size": 500_000, "learning_starts": 5_000, "update_interval": 4}},
        "training": {"total_timesteps": 3_000_000, "num_envs": 20},
        "hpo": {"steps_per_trial": 500_000},
    }


# ---------------------------------------------------------------------------
# Baseline: the reference-shaped config is clean
# ---------------------------------------------------------------------------

def test_reference_config_shape_passes():
    r = _run(_good())
    assert r.failures == [], r.failures
    assert r.warnings == [], r.warnings
    assert r.status == "PASS"


def test_empty_config_is_silent_not_passing():
    """Absent blocks SKIP. Silence must never be reported as a pass."""
    r = _run({})
    assert r.failures == []
    assert r.warnings == []
    assert r.passed == []


# ---------------------------------------------------------------------------
# Negative tripwires — one per invariant
# ---------------------------------------------------------------------------

def test_xparam_01_insufficient_post_warmup_budget():
    cfg = _good()
    cfg["hpo"]["steps_per_trial"] = 150_000  # 150k - 5k < 200k floor for SAC
    assert _failed(_run(cfg), "01")


def test_xparam_01_skipped_when_hpo_explicitly_disabled():
    cfg = _good()
    cfg["hpo"] = {"enabled": False, "steps_per_trial": 1_000}
    assert not _failed(_run(cfg), "01")


def test_xparam_01_iqn_uses_lower_floor():
    """IQN floor is 100k, so 150k-5k passes where SAC would fail."""
    cfg = _good()
    cfg["agents"] = {"iqn": {"learning_starts": 5_000, "n_step": 3}}
    cfg["hpo"]["steps_per_trial"] = 150_000
    assert not _failed(_run(cfg), "01")


def test_xparam_02_learning_starts_exceeds_total_timesteps():
    # Stay above the production floor so the guard doesn't mask the violation.
    cfg = _good()
    cfg["training"]["total_timesteps"] = 200_000
    cfg["agents"]["sac"]["learning_starts"] = 250_000
    assert _failed(_run(cfg), "02")


def test_xparam_03_learning_starts_exceeds_buffer():
    cfg = _good()
    cfg["agents"]["sac"]["buffer_size"] = 1_000
    assert _failed(_run(cfg), "03")


def test_xparam_04_fee_ramp_outlasts_run():
    cfg = _good()
    cfg["env"]["fee_schedule"] = [
        {"step": 0, "taker_fee": 0.0},
        {"step": 500_000, "ramp_to": 0.0005, "ramp_end_step": 9_000_000},
    ]
    assert _failed(_run(cfg), "04")


def test_xparam_04_fee_ramp_within_run_passes():
    cfg = _good()
    cfg["env"]["fee_schedule"] = [
        {"step": 0, "taker_fee": 0.0},
        {"step": 500_000, "ramp_to": 0.0005, "ramp_end_step": 2_000_000},
    ]
    assert not _failed(_run(cfg), "04")


def test_xparam_05_deadband_too_wide():
    cfg = _good()
    cfg["env"]["deadband_threshold"] = 0.75
    assert _failed(_run(cfg), "05")


@pytest.mark.parametrize("eta", [0.00001, 0.5])
def test_xparam_06_dsr_eta_out_of_range(eta):
    cfg = _good()
    cfg["env"]["reward"]["dsr_eta"] = eta
    assert _warned(_run(cfg), "06")


@pytest.mark.parametrize("scale", [0.01, 50.0])
def test_xparam_07_dsr_scale_out_of_range(scale):
    cfg = _good()
    cfg["env"]["reward"]["dsr_scale"] = scale
    assert _warned(_run(cfg), "07")


def test_xparam_07_reads_flat_key_fallback():
    """Some configs put dsr_* directly under env rather than env.reward."""
    cfg = _good()
    del cfg["env"]["reward"]
    cfg["env"]["dsr_scale"] = 99.0
    assert _warned(_run(cfg), "07")


def test_xparam_08_update_interval_times_envs_too_high():
    cfg = _good()
    cfg["agents"]["sac"]["update_interval"] = 16  # 16 * 20 = 320 > 200
    assert _warned(_run(cfg), "08")


def test_xparam_09_features_per_scale_mismatch():
    cfg = _good()
    cfg["network"]["scale_encoder"]["input_size"] = 12
    assert _failed(_run(cfg), "09")


def test_xparam_09_detects_two_way_mismatch_when_third_absent():
    cfg = _good()
    del cfg["network"]["scale_encoder"]
    cfg["env"]["features_per_scale"] = 12
    assert _failed(_run(cfg), "09")


def test_xparam_10_scale_lists_diverge():
    cfg = _good()
    cfg["env"]["scales"] = [15, 60]
    assert _failed(_run(cfg), "10")


@pytest.mark.parametrize(
    "mdp,private_dim,should_fail",
    [("v7", 4, True), ("v7", 5, False), ("v6", 5, True), ("v6", 4, False)],
)
def test_xparam_11_private_dim_bound_to_mdp_version(mdp, private_dim, should_fail):
    cfg = _good()
    cfg["env"]["mdp_version"] = mdp
    cfg["network"]["private_dim"] = private_dim
    assert _failed(_run(cfg), "11") is should_fail


def test_xparam_11_unknown_mdp_version_is_skipped():
    cfg = _good()
    cfg["env"]["mdp_version"] = "v9"
    assert not _failed(_run(cfg), "11")


def test_xparam_12_iqn_n_step_too_shallow():
    cfg = _good()
    cfg["agents"] = {"iqn": {"learning_starts": 5_000, "n_step": 1}}
    assert _failed(_run(cfg), "12")


# ---------------------------------------------------------------------------
# Scope guards — these exist because the first version of this check produced
# 18 false positives across the 149 real configs. Each guard gets a tripwire so
# the scoping cannot silently regress in either direction.
# ---------------------------------------------------------------------------

def test_live_inference_config_is_skipped_entirely():
    """Live configs (exchange/bar_clock, no `training`) use another schema.

    Their vestigial `buffer_size: 100` and checkpoint-shaped `network` block
    are not violations. Mirrors configs/live_gmgp1_xauusd_ctrader.yaml.
    """
    cfg = {
        "exchange": {"name": "ctrader"},
        "bar_clock": {"interval": "15m"},
        "agents": {"sac": {"buffer_size": 100, "learning_starts": 5_000}},
        "network": {"private_dim": 5, "scale_encoder": {"input_size": 8}},
        "features": {"features_per_scale": 19},
    }
    r = _run(cfg)
    assert r.failures == [] and r.warnings == [] and r.passed == []


def test_backtest_config_skips_budget_checks():
    """total_timesteps: 0 is a backtest, not a broken training run."""
    cfg = _good()
    cfg["training"]["total_timesteps"] = 0
    cfg["hpo"] = {"enabled": False}
    r = _run(cfg)
    assert not _failed(r, "01")
    assert not _failed(r, "02")
    assert not _failed(r, "03")


def test_backtest_config_still_enforces_dimensional_checks():
    """A backtest with the wrong private_dim IS broken — don't skip those."""
    cfg = _good()
    cfg["training"]["total_timesteps"] = 0
    cfg["network"]["private_dim"] = 4  # wrong for v7
    assert _failed(_run(cfg), "11")


def test_smoke_config_does_not_trip_budget_floor():
    """Mirrors configs/_smoke_leverage_axis.yaml — 1K steps, 1 trial."""
    cfg = _good()
    cfg["training"]["total_timesteps"] = 1_000
    cfg["hpo"] = {"enabled": True, "trials": 1, "steps_per_trial": 1_000}
    assert not _failed(_run(cfg), "01")


def test_production_scale_still_trips_budget_floor():
    """The guard must not disarm the check for real runs."""
    cfg = _good()
    cfg["training"]["total_timesteps"] = 3_000_000
    cfg["hpo"] = {"enabled": True, "trials": 50, "steps_per_trial": 150_000}
    assert _failed(_run(cfg), "01")


# ---------------------------------------------------------------------------
# Wiring: the check must actually run inside validate(), not just standalone
# ---------------------------------------------------------------------------

def test_check_xparam_is_wired_into_validate():
    """Declared-but-unwired is the failure mode this port exists to close."""
    import inspect

    from scripts.validate_config import validate

    assert "check_xparam" in inspect.getsource(validate)
