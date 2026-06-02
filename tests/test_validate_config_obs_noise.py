"""Validator tests for `check_obs_noise_gate` (Protocol v2.7-B, S553-cont-25).

Mirrors the Phase α / Phase β operator-decision behavior of
`check_sensitivity_audit`, locked in
`.agent/artifacts/protocol_v27_b_obs_noise_stage_3_5_architecture.md` (IC-4, ADR-9):

  Phase α (protocol_version unset or < "2.7"):
    - legacy config → no-op (exempt), regardless of prop-firm tag / missing keys
  Phase β (protocol_version >= "2.7"):
    - prop-firm config missing a required key → FAIL; research-only → WARN
    - sanity-bound and type checks on floors / buffers / n_seeds / required
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_obs_noise_gate,
)


def _cfg(*, protocol_version: str = "2.7", prop_firm: bool = True,
         **gates_overrides) -> dict:
    """v2.7 config with a full Stage 3.5 obs-noise gate block.

    Override via gates_overrides; pass ``key=None`` to delete a key.
    """
    tags = ["velotrade", "prop-firm"] if prop_firm else ["research"]
    gates = {
        "obs_noise_sigma_levels": {"10bps": 0.001, "50bps": 0.005},
        "obs_noise_n_seeds": 10,
        "obs_noise_pf_floor_10bps": 0.85,
        "obs_noise_pf_floor_50bps": 0.70,
        "obs_noise_mdd_buffer_pp_10bps": 0.3,
        "obs_noise_mdd_buffer_pp_50bps": 0.7,
        "obs_noise_required_min_folds": 4,
        "obs_noise_required": False,
    }
    for k, v in gates_overrides.items():
        if v is None:
            gates.pop(k, None)
        else:
            gates[k] = v
    return {
        "protocol_version": protocol_version,
        "wandb": {"tags": tags},
        "env": {"deadband_threshold": 0.25, "max_leverage": 1.0},
        "gates": gates,
    }


# --- Phase α exemption ------------------------------------------------------


def test_phase_alpha_pre27_propfirm_exempt():
    """A prop-firm config below 2.7 is a no-op even with every key missing."""
    r = ValidationResult()
    cfg = _cfg(protocol_version="2.6", obs_noise_sigma_levels=None,
               obs_noise_pf_floor_10bps=None, obs_noise_required=None)
    check_obs_noise_gate(cfg, r)
    assert r.failures == [] and r.warnings == [], (r.failures, r.warnings)


def test_phase_alpha_missing_protocol_version_exempt():
    """No protocol_version → defaults to "2.6" → exempt."""
    cfg = _cfg()
    cfg.pop("protocol_version", None)
    cfg["gates"].pop("obs_noise_sigma_levels", None)
    r = ValidationResult()
    check_obs_noise_gate(cfg, r)
    assert r.failures == [] and r.warnings == []


def test_phase_alpha_v26_sensitivity_era_exempt():
    """A v2.6 (sensitivity-audit-era) config is not yet subject to Stage 3.5."""
    r = ValidationResult()
    check_obs_noise_gate(_cfg(protocol_version="2.6"), r)
    assert r.failures == [] and r.warnings == []


# --- Phase β baseline -------------------------------------------------------


def test_phase_beta_v27_baseline_propfirm_passes():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(), r)
    assert r.failures == [], f"unexpected failures: {r.failures}"


# --- sigma_levels -----------------------------------------------------------


def test_sigma_levels_missing_propfirm_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_sigma_levels=None), r)
    assert any("obs_noise_sigma_levels" in f for f in r.failures)


def test_sigma_levels_missing_research_warns():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(prop_firm=False, obs_noise_sigma_levels=None), r)
    assert r.failures == []
    assert any("obs_noise_sigma_levels" in w for w in r.warnings)


def test_sigma_level_negative_value_fails():
    r = ValidationResult()
    check_obs_noise_gate(
        _cfg(obs_noise_sigma_levels={"10bps": -0.001, "50bps": 0.005}), r,
    )
    assert any("must be >= 0" in f for f in r.failures)


# --- pf_floor / mdd_buffer per sigma ---------------------------------------


def test_pf_floor_missing_propfirm_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_pf_floor_50bps=None), r)
    assert any("obs_noise_pf_floor_50bps" in f and "not set" in f
               for f in r.failures)


def test_pf_floor_missing_research_warns():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(prop_firm=False, obs_noise_pf_floor_50bps=None), r)
    assert r.failures == []
    assert any("obs_noise_pf_floor_50bps" in w for w in r.warnings)


def test_mdd_buffer_missing_propfirm_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_mdd_buffer_pp_10bps=None), r)
    assert any("obs_noise_mdd_buffer_pp_10bps" in f and "not set" in f
               for f in r.failures)


def test_pf_floor_out_of_range_warns():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_pf_floor_10bps=0.40), r)  # below 0.50
    assert any("obs_noise_pf_floor_10bps" in w and "sanity bound" in w
               for w in r.warnings)
    assert r.failures == []


def test_mdd_buffer_out_of_range_warns():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_mdd_buffer_pp_50bps=9.0), r)  # above 5.0
    assert any("obs_noise_mdd_buffer_pp_50bps" in w and "sanity bound" in w
               for w in r.warnings)


def test_pf_floor_non_numeric_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_pf_floor_10bps="high"), r)
    assert any("obs_noise_pf_floor_10bps" in f and "not numeric" in f
               for f in r.failures)


# --- n_seeds ----------------------------------------------------------------


def test_n_seeds_too_low_warns():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_n_seeds=3), r)
    assert any("obs_noise_n_seeds" in w and "< 5" in w for w in r.warnings)


def test_n_seeds_missing_propfirm_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_n_seeds=None), r)
    assert any("obs_noise_n_seeds" in f and "not set" in f for f in r.failures)


# --- obs_noise_required -----------------------------------------------------


def test_required_missing_propfirm_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_required=None), r)
    assert any("obs_noise_required" in f and "not set" in f for f in r.failures)


def test_required_non_boolean_fails():
    r = ValidationResult()
    check_obs_noise_gate(_cfg(obs_noise_required="yes"), r)
    assert any("obs_noise_required" in f and "boolean" in f for f in r.failures)


# --- overlay: real shipped gates files load the keys (B3 integration) -------


def test_v27_overlay_loads_real_gates_file_no_fail():
    """A v2.7 prop-firm config delegating to a shipped <ws>_ensemble.gates.yaml
    must resolve the obs_noise_* keys via _load_ensemble_gates_overlay and pass
    (the B3 yaml edits are well-formed and complete). Runs from the repo root."""
    r = ValidationResult()
    cfg = {
        "protocol_version": "2.7",
        "wandb": {"tags": ["velotrade", "prop-firm"]},
        "env": {"deadband_threshold": 0.25, "max_leverage": 1.0},
        "ensemble": {"gates_file": "configs/gmgp1_btc_velotrade_ensemble.gates.yaml"},
        "gates": {},
    }
    check_obs_noise_gate(cfg, r)
    assert r.failures == [], f"shipped gates file should satisfy the gate: {r.failures}"
