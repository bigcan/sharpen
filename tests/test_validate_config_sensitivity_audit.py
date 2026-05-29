"""Validator tests for `check_sensitivity_audit` (Protocol v2.6, S553).

Covers the Phase α / Phase β operator-decision behavior locked in
`.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md` ADR-3:

  Phase α (legacy / protocol_version != "2.6"):
    1. Legacy v2.5 config → no-op (exempt), regardless of prop-firm tag
    2. Missing protocol_version field → treated as "2.5" → exempt

  Phase β (protocol_version == "2.6"):
    3. Fully-populated v2.6 prop-firm config → passes cleanly
    4. edge_stability_pf_ratio_floor missing, prop-firm → FAIL with hint
    5. edge_stability_pf_ratio_floor missing, research-only → WARN
    6. edge_stability_pf_ratio_floor outside [0.50, 0.95] → WARN (sanity)
    7. edge_stability_pf_ratio_floor non-numeric → FAIL
    8. sensitivity_deadband_grid not 3-element list, prop-firm → FAIL
    9. sensitivity_max_leverage_mults not 3-element list, prop-firm → FAIL
   10. SENS-1: deadband grid center != env.deadband_threshold → FAIL
   11. SENS-1: max_leverage_mults center != 1.0 → FAIL
   12. sensitivity_deployable_max_leverage_cap negative → FAIL
   13. sensitivity_audit_required not set, prop-firm → FAIL
   14. sensitivity_audit_required non-boolean → FAIL
   15. Phase β: required=true + prop-firm + all keys present → OK emitted
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_sensitivity_audit,
)


def _baseline_cfg(
    *,
    protocol_version: str = "2.6",
    prop_firm: bool = True,
    deployed_deadband: float = 0.25,
    **gates_overrides,
) -> dict:
    """v2.6 prop-firm L1 multiseed config with full sensitivity audit block.

    Defaults to Phase β-enforceable shape (audit_required=true). Override
    via gates_overrides to test specific failure modes; pass `key=None` to
    delete a key.
    """
    tags = ["velotrade", "prop-firm"] if prop_firm else ["research"]
    gates = {
        "edge_stability_pf_ratio_floor": 0.70,
        "sensitivity_deadband_grid": [0.20, 0.25, 0.30],
        "sensitivity_max_leverage_mults": [0.5, 1.0, 1.5],
        "sensitivity_deployable_max_leverage_cap": 1.0,
        "sensitivity_audit_required_min_deployable_neighbors": 4,
        "sensitivity_audit_required": True,
    }
    for k, v in gates_overrides.items():
        if v is None:
            gates.pop(k, None)
        else:
            gates[k] = v
    return {
        "protocol_version": protocol_version,
        "wandb": {"tags": tags},
        "env": {"deadband_threshold": deployed_deadband, "max_leverage": 1.0},
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Phase α back-compat: legacy v2.5 configs exempt.
# ---------------------------------------------------------------------------

def test_phase_alpha_legacy_v25_propfirm_exempt():
    """A prop-firm config without protocol_version is treated as legacy 2.5 →
    no-op even if every sensitivity gate is missing (Phase α back-compat)."""
    r = ValidationResult()
    cfg = _baseline_cfg(
        protocol_version="2.5",
        edge_stability_pf_ratio_floor=None,
        sensitivity_deadband_grid=None,
        sensitivity_max_leverage_mults=None,
        sensitivity_audit_required=None,
    )
    check_sensitivity_audit(cfg, r)
    assert r.failures == [], f"legacy v2.5 must be exempt; got {r.failures}"
    assert r.warnings == [], f"legacy v2.5 must be silent; got {r.warnings}"


def test_phase_alpha_missing_protocol_version_exempt():
    """A config without `protocol_version` field defaults to "2.5" → exempt."""
    cfg = _baseline_cfg(protocol_version="2.5")
    cfg.pop("protocol_version", None)
    cfg["gates"].pop("edge_stability_pf_ratio_floor", None)
    r = ValidationResult()
    check_sensitivity_audit(cfg, r)
    assert r.failures == []
    assert r.warnings == []


# ---------------------------------------------------------------------------
# Phase β: baseline v2.6 prop-firm config passes cleanly.
# ---------------------------------------------------------------------------

def test_phase_beta_baseline_propfirm_v26_passes():
    r = ValidationResult()
    check_sensitivity_audit(_baseline_cfg(), r)
    assert r.failures == [], f"unexpected failures: {r.failures}"
    # Should emit OK at the bottom because required=true + prop-firm + clean.
    assert any("enforcement active" in p for p in r.passed)


# ---------------------------------------------------------------------------
# Rule 4: floor missing → FAIL prop-firm, WARN research.
# ---------------------------------------------------------------------------

def test_floor_missing_propfirm_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(edge_stability_pf_ratio_floor=None)
    check_sensitivity_audit(cfg, r)
    assert any(
        "edge_stability_pf_ratio_floor" in f for f in r.failures
    ), f"expected FAIL on missing floor; got {r.failures}"
    # Hint text must mention sister yamls so operators can copy-paste.
    assert any("sister" in f.lower() or "Phase" in f for f in r.failures)


def test_floor_missing_research_only_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(prop_firm=False, edge_stability_pf_ratio_floor=None)
    check_sensitivity_audit(cfg, r)
    assert r.failures == [], (
        f"research-only config must not FAIL on missing floor; got {r.failures}"
    )
    assert any(
        "edge_stability_pf_ratio_floor" in w for w in r.warnings
    ), f"expected WARN on missing floor; got {r.warnings}"


# ---------------------------------------------------------------------------
# Rule 6: sanity bound on the floor.
# ---------------------------------------------------------------------------

def test_floor_below_sanity_bound_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(edge_stability_pf_ratio_floor=0.30)
    check_sensitivity_audit(cfg, r)
    assert any("sanity bound" in w for w in r.warnings), (
        f"expected WARN on floor outside [0.50, 0.95]; got {r.warnings}"
    )


def test_floor_above_sanity_bound_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(edge_stability_pf_ratio_floor=0.98)
    check_sensitivity_audit(cfg, r)
    assert any("sanity bound" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Rule 7: non-numeric floor → FAIL.
# ---------------------------------------------------------------------------

def test_floor_non_numeric_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(edge_stability_pf_ratio_floor="not_a_number")
    check_sensitivity_audit(cfg, r)
    assert any("is not numeric" in f for f in r.failures)


# ---------------------------------------------------------------------------
# Rules 8-9: grid shape requirements.
# ---------------------------------------------------------------------------

def test_deadband_grid_wrong_length_propfirm_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_deadband_grid=[0.20, 0.25])  # only 2
    check_sensitivity_audit(cfg, r)
    assert any(
        "sensitivity_deadband_grid" in f and "3-element" in f
        for f in r.failures
    )


def test_max_leverage_mults_wrong_length_propfirm_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(
        sensitivity_max_leverage_mults=[0.5, 1.0, 1.5, 2.0],  # 4 elements
    )
    check_sensitivity_audit(cfg, r)
    assert any(
        "sensitivity_max_leverage_mults" in f and "3-element" in f
        for f in r.failures
    )


# ---------------------------------------------------------------------------
# Rule 10-11: SENS-1 invariant — center must match deployed config.
# ---------------------------------------------------------------------------

def test_sens1_deadband_center_mismatch_fails():
    """env.deadband_threshold=0.25 but grid center is 0.30 → FAIL."""
    r = ValidationResult()
    cfg = _baseline_cfg(
        deployed_deadband=0.25,
        sensitivity_deadband_grid=[0.25, 0.30, 0.35],  # center is 0.30, not 0.25
    )
    check_sensitivity_audit(cfg, r)
    assert any(
        "SENS-1" in f or "non-centered" in f for f in r.failures
    ), f"expected SENS-1 FAIL on center mismatch; got {r.failures}"


def test_sens1_deadband_center_match_passes():
    """env.deadband_threshold=0.30 + grid center=0.30 → passes SENS-1."""
    r = ValidationResult()
    cfg = _baseline_cfg(
        deployed_deadband=0.30,
        sensitivity_deadband_grid=[0.25, 0.30, 0.35],
    )
    check_sensitivity_audit(cfg, r)
    sens1_failures = [f for f in r.failures if "SENS-1" in f or "non-centered" in f]
    assert sens1_failures == [], f"unexpected SENS-1 fails: {sens1_failures}"


def test_sens1_max_leverage_mults_center_not_one_fails():
    """sensitivity_max_leverage_mults center must equal 1.0 (the deployed cap)."""
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_max_leverage_mults=[0.5, 1.2, 1.5])
    check_sensitivity_audit(cfg, r)
    assert any(
        "must equal 1.0" in f for f in r.failures
    ), f"expected FAIL on mults center != 1.0; got {r.failures}"


# ---------------------------------------------------------------------------
# Rule 12: deployable cap must be positive.
# ---------------------------------------------------------------------------

def test_deployable_cap_negative_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_deployable_max_leverage_cap=-0.5)
    check_sensitivity_audit(cfg, r)
    assert any(
        "must be positive" in f for f in r.failures
    ), f"expected FAIL on negative cap; got {r.failures}"


def test_deployable_cap_omitted_does_not_fail():
    """Cap is optional; absence is fine (defaults applied at runtime)."""
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_deployable_max_leverage_cap=None)
    check_sensitivity_audit(cfg, r)
    cap_failures = [
        f for f in r.failures if "sensitivity_deployable_max_leverage_cap" in f
    ]
    assert cap_failures == [], f"cap omission should not FAIL; got {cap_failures}"


# ---------------------------------------------------------------------------
# Rule 13-14: sensitivity_audit_required field semantics.
# ---------------------------------------------------------------------------

def test_audit_required_missing_propfirm_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_audit_required=None)
    check_sensitivity_audit(cfg, r)
    assert any(
        "sensitivity_audit_required" in f and "not set" in f for f in r.failures
    )


def test_audit_required_non_boolean_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_audit_required="yes")  # string, not bool
    check_sensitivity_audit(cfg, r)
    assert any(
        "must be boolean" in f for f in r.failures
    ), f"expected FAIL on non-boolean; got {r.failures}"


def test_audit_required_false_phase_alpha_state_passes_propfirm():
    """Phase α: gates declare audit_required=false. Validator still fires (we're
    at protocol_version 2.6 in this test), keys are present and well-formed, so
    no FAIL; OK message only emitted when required=true."""
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_audit_required=False)
    check_sensitivity_audit(cfg, r)
    assert r.failures == [], (
        f"required=false should still pass validator if all other keys are good; "
        f"got {r.failures}"
    )
    # No OK because required=false → enforcement is not "active".
    assert not any("enforcement active" in p for p in r.passed)


# ---------------------------------------------------------------------------
# Edge case: min_deployable_neighbors validation.
# ---------------------------------------------------------------------------

def test_min_neighbors_out_of_range_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(sensitivity_audit_required_min_deployable_neighbors=10)
    check_sensitivity_audit(cfg, r)
    assert any(
        "outside [1, 8]" in w for w in r.warnings
    ), f"expected WARN on neighbors > 8; got {r.warnings}"


def test_min_neighbors_non_integer_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(
        sensitivity_audit_required_min_deployable_neighbors="four",
    )
    check_sensitivity_audit(cfg, r)
    assert any("is not integer" in f for f in r.failures)
