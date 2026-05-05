"""Validator tests for `check_ensemble_confirm` (Protocol v2.5 promotion).

Covers the 7 rule-coverage cases in C-2 of the v2.5 architecture plan
(.agent/artifacts/stage_2_5_bootstrap_primary_architecture.md):

  1. ensemble_bootstrap_p_pf_promote missing, prop-firm  → FAIL with hint
  2. ensemble_bootstrap_p_mdd_promote missing, prop-firm → FAIL with hint
  3. ensemble_bootstrap_resamples missing                → WARN (not FAIL)
  4. ensemble_uplift_min missing, prop-firm              → WARN (demoted in v2.5)
  5. ensemble_uplift_min < 1.05                          → WARN (unchanged)
  6. ensemble_bootstrap_p_pf_promote ∉ [0.80, 0.99]      → WARN (sanity bound)
  7. ensemble_bootstrap_p_pf_ambiguous ≥ promote         → FAIL (ordering invariant)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_ensemble_confirm,
)


def _baseline_cfg(*, prop_firm: bool = True, **gates_overrides) -> dict:
    """Minimal Stage 2.5 config with the canonical bootstrap block."""
    tags = ["velotrade", "prop-firm"] if prop_firm else ["research"]
    gates = {
        "ensemble_bootstrap_p_pf_promote": 0.90,
        "ensemble_bootstrap_p_mdd_promote": 0.90,
        "ensemble_bootstrap_p_pf_ambiguous": 0.75,
        "ensemble_bootstrap_resamples": 10000,
        "ensemble_uplift_min": 1.10,
        "ensemble_ambiguous_min": 1.05,
    }
    for k, v in gates_overrides.items():
        if v is None:
            gates.pop(k, None)
        else:
            gates[k] = v
    return {
        "wandb": {"tags": tags},
        "ensemble": {"seeds": [42, 2025, 3141], "rule": "ens_agreement"},
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Baseline: a fully-populated v2.5 prop-firm config passes cleanly.
# ---------------------------------------------------------------------------

def test_baseline_propfirm_v25_config_passes():
    r = ValidationResult()
    check_ensemble_confirm(_baseline_cfg(), r)
    assert r.failures == [], f"unexpected failures: {r.failures}"


# ---------------------------------------------------------------------------
# Rule 1: ensemble_bootstrap_p_pf_promote missing — prop-firm FAIL.
# ---------------------------------------------------------------------------

def test_rule_1_p_pf_promote_missing_propfirm_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_bootstrap_p_pf_promote=None)
    check_ensemble_confirm(cfg, r)
    assert any(
        "ensemble_bootstrap_p_pf_promote" in f for f in r.failures
    ), f"expected FAIL on missing p_pf_promote; got {r.failures}"
    # Hint text must point at sister gates yamls so operators can copy-paste.
    assert any("sister" in f.lower() or "gates.yaml" in f for f in r.failures)


def test_rule_1_p_pf_promote_missing_research_only_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(prop_firm=False, ensemble_bootstrap_p_pf_promote=None)
    check_ensemble_confirm(cfg, r)
    assert r.failures == []
    assert any("ensemble_bootstrap_p_pf_promote" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Rule 2: ensemble_bootstrap_p_mdd_promote missing — prop-firm FAIL.
# ---------------------------------------------------------------------------

def test_rule_2_p_mdd_promote_missing_propfirm_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_bootstrap_p_mdd_promote=None)
    check_ensemble_confirm(cfg, r)
    assert any("ensemble_bootstrap_p_mdd_promote" in f for f in r.failures)


# ---------------------------------------------------------------------------
# Rule 3: ensemble_bootstrap_resamples missing — WARN, not FAIL.
# ---------------------------------------------------------------------------

def test_rule_3_resamples_missing_warns_only():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_bootstrap_resamples=None)
    check_ensemble_confirm(cfg, r)
    assert r.failures == [], f"resamples missing should not FAIL; got {r.failures}"
    assert any("ensemble_bootstrap_resamples" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Rule 4: ensemble_uplift_min missing — WARN (demoted in v2.5).
# ---------------------------------------------------------------------------

def test_rule_4_uplift_min_missing_propfirm_warns_not_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_uplift_min=None)
    check_ensemble_confirm(cfg, r)
    # In v2.3 this was a FAIL for prop-firm; v2.5 demotes to WARN.
    assert all(
        "ensemble_uplift_min" not in f for f in r.failures
    ), f"v2.5 must not FAIL on missing uplift_min; got {r.failures}"
    assert any("ensemble_uplift_min" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Rule 5: ensemble_uplift_min < 1.05 — WARN.
# ---------------------------------------------------------------------------

def test_rule_5_uplift_min_below_ambiguous_floor_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_uplift_min=1.02)
    check_ensemble_confirm(cfg, r)
    assert any("< 1.05" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Rule 6: ensemble_bootstrap_p_pf_promote outside [0.80, 0.99] — WARN.
# ---------------------------------------------------------------------------

def test_rule_6_p_pf_promote_below_sanity_bound_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_bootstrap_p_pf_promote=0.70)
    check_ensemble_confirm(cfg, r)
    assert any(
        "p_pf_promote" in w and ("0.80" in w or "sanity" in w.lower())
        for w in r.warnings
    ), f"expected sanity-bound WARN; got {r.warnings}"


def test_rule_6_p_pf_promote_above_sanity_bound_warns():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_bootstrap_p_pf_promote=0.999)
    check_ensemble_confirm(cfg, r)
    assert any(
        "p_pf_promote" in w and ("0.99" in w or "sanity" in w.lower())
        for w in r.warnings
    )


def test_rule_6_p_pf_promote_inside_sanity_bound_no_warn():
    r = ValidationResult()
    cfg = _baseline_cfg(ensemble_bootstrap_p_pf_promote=0.85)
    check_ensemble_confirm(cfg, r)
    assert not any(
        "sanity" in w.lower() and "p_pf_promote" in w for w in r.warnings
    )


# ---------------------------------------------------------------------------
# Rule 7: ensemble_bootstrap_p_pf_ambiguous >= promote — FAIL.
# ---------------------------------------------------------------------------

def test_rule_7_ambiguous_at_promote_fails_ordering_invariant():
    r = ValidationResult()
    cfg = _baseline_cfg(
        ensemble_bootstrap_p_pf_promote=0.90,
        ensemble_bootstrap_p_pf_ambiguous=0.90,  # equals → empty band
    )
    check_ensemble_confirm(cfg, r)
    assert any(
        "ambiguous" in f.lower() and "promote" in f.lower() for f in r.failures
    ), f"expected ordering-invariant FAIL; got {r.failures}"


def test_rule_7_ambiguous_above_promote_fails():
    r = ValidationResult()
    cfg = _baseline_cfg(
        ensemble_bootstrap_p_pf_promote=0.85,
        ensemble_bootstrap_p_pf_ambiguous=0.92,
    )
    check_ensemble_confirm(cfg, r)
    assert any("ambiguous" in f.lower() for f in r.failures)
