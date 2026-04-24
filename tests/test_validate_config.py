"""Tests for scripts/validate_config.py (subset covering rev-2 additions)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_legacy_prop_firm_block,
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
