"""validate_config: declared-but-unconsumed controls, unread keys, and the §3.5 demotion
(Protocol v2 audit 2026-09-29 §6 item 10; TAILWIND Tier-2 N3 / T6-09).

A safeguard declared in a config that no code on the book's run path reads is false assurance,
and the old validator reported such keys as a PASS. Pinned both ways, including a grep of the
forward runner so the consumer table cannot drift from the code it describes.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.validate_config as vc  # noqa: E402

RUNNER = (ROOT / "sharpen" / "paper" / "forward_runner.py").read_text(encoding="utf-8")


def _linear(**extra) -> dict:
    cfg = {"env": {"type": "multi_asset_allocator"}, "execution": {"shadow": "linear_core_tailwind"}}
    for k, v in extra.items():
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    return cfg


def _run(check, cfg) -> vc.ValidationResult:
    r = vc.ValidationResult()
    check(cfg, r)
    return r


@pytest.mark.parametrize("key,val", [
    ("risk.static_peak", True),
    ("safety.flatten_on_kill_file", False),
    ("gates.safe_mode.crit_repeat_window_hours", 24),
    ("gates.safe_mode.crit_repeat_count_before_lockout", 2),
    ("gates.drift.feature_variance_veto", {"enabled": True}),
])
def test_unconsumed_control_fails_on_a_linear_book(key, val):
    r = _run(vc.check_linear_controls_consumed, _linear(**{key: val}))
    assert r.status == "FAIL" and key in r.failures[0]


@pytest.mark.parametrize("key,val", [("risk.static_peak", False), ("safety.flatten_on_kill_file", True)])
def test_values_matching_runner_behaviour_pass(key, val):
    assert _run(vc.check_linear_controls_consumed, _linear(**{key: val})).status == "PASS"


def test_non_linear_book_is_not_checked():
    cfg = {"env": {"type": "continuous_swing"}, "risk": {"static_peak": True}}
    r = _run(vc.check_linear_controls_consumed, cfg)
    assert not r.failures and not r.passed


def test_consumer_table_matches_the_runner():
    """Both directions: every key the table calls UNCONSUMED is absent from the runner, and the
    consumed siblings are present. Wiring a consumer must update the table (and vice versa)."""
    for key in vc._LINEAR_PATH_UNCONSUMED:
        leaf = key.rsplit(".", 1)[1]
        if leaf == "static_peak":
            assert "static_peak" not in RUNNER and "peak_equity" in RUNNER     # trailing peak
        else:
            assert leaf not in RUNNER, f"runner now reads {leaf}: update _LINEAR_PATH_UNCONSUMED"
    for consumed in ("kill_file", "crit_triggers_flatten", "window_bars", "min_bars_before_check",
                     "deadband_frac_warn", "deadband_frac_crit", "saturation_frac_warn",
                     "saturation_frac_crit", "action_kl_warn", "action_kl_crit", "baseline_path"):
        assert consumed in RUNNER, f"runner no longer reads {consumed}"


def test_linear_presence_rule_does_not_deadlock():
    """A linear book needs only crit_triggers_flatten of the safe-mode keys; an RL prop-firm book
    still needs all three."""
    drift = {k: 0.1 for k in vc._V22_DRIFT_GATE_KEYS}
    lin = _linear(**{"gates.drift": dict(drift), "gates.safe_mode": {"crit_triggers_flatten": True}})
    r = _run(vc.check_drift_safemode_gates, lin)
    assert not any("safe_mode missing" in f for f in r.failures + r.warnings)
    rl = {"env": {"type": "continuous_swing"}, "gates": {"drift": dict(drift),
                                                         "safe_mode": {"crit_triggers_flatten": True}}}
    r = _run(vc.check_drift_safemode_gates, rl)
    assert any("safe_mode missing" in m for m in r.failures + r.warnings)


def test_tailwind_challenge_is_blocked_at_paper_deploy_for_its_unconsumed_controls():
    cfg = yaml.safe_load((ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    r = _run(vc.check_linear_controls_consumed, cfg)
    assert r.status == "FAIL"
    msg = r.failures[0]
    for key in ("risk.static_peak", "crit_repeat_window_hours", "feature_variance_veto"):
        assert key in msg
    fixed = copy.deepcopy(cfg)
    fixed["risk"].pop("static_peak")
    for k in ("crit_repeat_window_hours", "crit_repeat_count_before_lockout"):
        fixed["gates"]["safe_mode"].pop(k)
    fixed["gates"]["drift"].pop("feature_variance_veto")
    assert _run(vc.check_linear_controls_consumed, fixed).status == "PASS"    # removable: no deadlock


def test_linear_book_is_not_asked_for_static_peak_at_paper_deploy():
    cfg = yaml.safe_load((ROOT / "configs" / "tailwind_v1_challenge.yaml").read_text(encoding="utf-8"))
    cfg["risk"].pop("static_peak")
    r = _run(vc.check_paper_deploy, cfg)
    assert not any("static_peak must be true" in f for f in r.failures)


def test_multiplicity_above_50x_is_advisory(monkeypatch):
    monkeypatch.setattr(vc, "_training_budget_multiplicity", lambda cfg, steps: (60.0, 1000.0, "test"))
    r = vc.ValidationResult()
    vc._check_multiplicity({}, 60_000, "L1", r)
    assert not r.failures and any("unvalidated" in w for w in r.warnings)


def test_unread_keys_warn_wherever_they_appear():
    cfg = {"gates": {"wf_pf_floor": 1.1, "nested": {"hpo_pf_floor": 1.2}}, "recent_oos_days": 60}
    r = _run(vc.check_unread_keys, cfg)
    assert len(r.warnings) == 3 and not r.failures
    assert all("NOT ENFORCED" in w for w in r.warnings)
    assert _run(vc.check_unread_keys, {"gates": {"min_dsr": 0.95}}).warnings == []


def test_new_checks_are_registered():
    assert vc.check_linear_controls_consumed in vc.STAGE_CHECKS["paper-deploy"]
    assert all(vc.check_unread_keys in checks for checks in vc.STAGE_CHECKS.values())
