"""Tests for CROSSQ-01..03 pre-launch checks in validate_config.py.

The agent raises on every one of these at construction; these checks exist so
the failure lands in the pre-flight command instead of after a GPU has been
rented. Each has a negative case.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_crossq,
)


def _run(sac: dict, training: dict | None = None) -> ValidationResult:
    r = ValidationResult()
    check_crossq(
        {"agents": {"sac": sac}, "training": training or {"total_timesteps": 3_000_000}},
        r,
    )
    return r


def _failed(r: ValidationResult, cid: str) -> bool:
    return any(f"CROSSQ-{cid}" in f for f in r.failures)


def _warned(r: ValidationResult, cid: str) -> bool:
    return any(f"CROSSQ-{cid}" in w for w in r.warnings)


class TestSilentWhenNotApplicable:

    def test_no_crossq_key_is_silent(self):
        r = _run({"tau": 0.005, "update_interval": 4})
        assert not r.failures and not r.warnings and not r.passed

    def test_crossq_false_is_silent(self):
        r = _run({"crossq": False})
        assert not r.failures and not r.warnings

    def test_live_inference_config_is_skipped(self):
        r = ValidationResult()
        check_crossq({"exchange": "bybit", "agents": {"sac": {"crossq": {"enabled": True, "bad": 1}}}}, r)
        assert not r.failures


class TestCrossQChecks:

    def test_clean_config_passes(self):
        r = _run({"crossq": True, "update_interval": 4})
        assert not r.failures
        assert any("CROSSQ enabled" in p for p in r.passed)

    def test_malformed_config_fails(self):
        r = _run({"crossq": {"enabled": True, "bn_warmup_step": 10}})
        assert _failed(r, "01")

    def test_wrong_type_fails(self):
        r = _run({"crossq": "yes"})
        assert _failed(r, "01")

    def test_distributional_combination_fails(self):
        r = _run({"crossq": True, "distributional": True})
        assert _failed(r, "02")

    def test_distributional_alone_is_fine(self):
        r = _run({"crossq": False, "distributional": True})
        assert not r.failures

    def test_warmup_longer_than_the_run_warns(self):
        r = _run(
            {"crossq": {"enabled": True, "bn_warmup_steps": 100_000}, "update_interval": 1},
            training={"total_timesteps": 50_000},
        )
        assert _warned(r, "03")

    def test_warmup_inside_the_run_does_not_warn(self):
        r = _run(
            {"crossq": {"enabled": True, "bn_warmup_steps": 100_000}, "update_interval": 4},
            training={"total_timesteps": 3_000_000},
        )
        assert not _warned(r, "03")

    def test_tau_is_flagged_as_inert(self):
        r = _run({"crossq": True, "tau": 0.005})
        assert any("tau is inert" in w for w in r.warnings)
