"""Tests for scripts/validate_config.py (subset covering rev-2 additions)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_challenge_block,
    check_legacy_prop_firm_block,
    check_static_peak_consistency,
)


# ---------------------------------------------------------------------------
# check_legacy_prop_firm_block (rev-2, prop-firm decoupling)
# ---------------------------------------------------------------------------

def test_no_prop_firm_block_is_silent():
    """Configs without env.prop_firm: emit neither warn nor fail."""
    r = ValidationResult()
    check_legacy_prop_firm_block({"env": {"risk": {"enabled": True}}}, "hpo", r)
    assert r.warnings == []
    assert r.failures == []


def test_prop_firm_block_warns_at_hpo():
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True, "profit_target_pct": 0.10}}}
    check_legacy_prop_firm_block(cfg, "hpo", r)
    assert any("env.prop_firm" in w for w in r.warnings)
    assert r.failures == []


def test_prop_firm_block_warns_at_l1_multiseed():
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True}}}
    check_legacy_prop_firm_block(cfg, "l1-multiseed", r)
    assert r.warnings
    assert r.failures == []


def test_prop_firm_block_warns_at_wf():
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True}}}
    check_legacy_prop_firm_block(cfg, "wf", r)
    assert r.warnings
    assert r.failures == []


def test_prop_firm_block_fails_at_paper_deploy():
    """Paper-deploy is the hard gate: migrated deploys must use env.risk:."""
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True, "profit_target_pct": 0.10}}}
    check_legacy_prop_firm_block(cfg, "paper-deploy", r)
    assert r.warnings == []
    assert any("env.prop_firm" in f for f in r.failures)


def test_migration_message_is_actionable():
    """Warning/failure message should name the new keys."""
    r = ValidationResult()
    check_legacy_prop_firm_block(
        {"env": {"prop_firm": {}}}, "paper-deploy", r
    )
    msg = r.failures[0] if r.failures else ""
    assert "env.risk" in msg
    assert "challenge" in msg


# ---------------------------------------------------------------------------
# check_challenge_block (S495-cont)
# ---------------------------------------------------------------------------

def test_challenge_absent_is_silent():
    r = ValidationResult()
    check_challenge_block({}, r)
    assert r.warnings == [] and r.failures == []


def test_challenge_disabled_block_is_silent():
    r = ValidationResult()
    check_challenge_block({"challenge": {"enabled": False}}, r)
    assert r.failures == []


def test_challenge_phase_must_be_known():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "bogus", "advance_rule": "manual_ack"}}
    check_challenge_block(cfg, r)
    assert any("phase" in f for f in r.failures)


def test_challenge_advance_rule_auto_rejected():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "step1",
                          "profit_target_pct": 0.10,
                          "advance_rule": "auto"}}
    check_challenge_block(cfg, r)
    assert any("advance_rule" in f for f in r.failures)


def test_challenge_funded_must_null_target():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "funded",
                          "profit_target_pct": 0.05,
                          "advance_rule": "manual_ack"}}
    check_challenge_block(cfg, r)
    assert any("funded" in f for f in r.failures)


def test_challenge_step1_requires_positive_target():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "step1",
                          "profit_target_pct": 0.0,
                          "advance_rule": "manual_ack"}}
    check_challenge_block(cfg, r)
    assert any("profit_target_pct" in f for f in r.failures)


def test_challenge_n_confirm_must_be_positive_int():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "step1",
                          "profit_target_pct": 0.10,
                          "advance_rule": "manual_ack",
                          "n_confirm": 0}}
    check_challenge_block(cfg, r)
    assert any("n_confirm" in f for f in r.failures)


def test_challenge_smoothing_window_must_be_positive_int():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "step1",
                          "profit_target_pct": 0.10,
                          "advance_rule": "manual_ack",
                          "smoothing_window": "three"}}
    check_challenge_block(cfg, r)
    assert any("smoothing_window" in f for f in r.failures)


def test_challenge_happy_path_passes():
    r = ValidationResult()
    cfg = {"challenge": {"enabled": True, "phase": "step1",
                          "profit_target_pct": 0.10,
                          "next_phase": "step2",
                          "advance_rule": "manual_ack",
                          "n_confirm": 2,
                          "smoothing_window": 3}}
    check_challenge_block(cfg, r)
    assert r.failures == []


# ---------------------------------------------------------------------------
# check_static_peak_consistency (S495-cont)
# ---------------------------------------------------------------------------

def test_static_peak_consistency_both_agree():
    r = ValidationResult()
    cfg = {"env": {"risk": {"static_peak": True}},
           "risk": {"static_peak": True}}
    check_static_peak_consistency(cfg, r)
    assert r.failures == []


def test_static_peak_consistency_mismatch_fails():
    r = ValidationResult()
    cfg = {"env": {"risk": {"static_peak": True}},
           "risk": {"static_peak": False}}
    check_static_peak_consistency(cfg, r)
    assert any("static_peak" in f for f in r.failures)


def test_static_peak_consistency_one_side_unset_is_ok():
    """Only env.risk set, live risk missing — accepted (default-compatible)."""
    r = ValidationResult()
    cfg = {"env": {"risk": {"static_peak": True}}}
    check_static_peak_consistency(cfg, r)
    assert r.failures == []


def test_static_peak_consistency_neither_set_is_silent():
    r = ValidationResult()
    check_static_peak_consistency({}, r)
    assert r.failures == []
