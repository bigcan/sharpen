"""Tests for scripts/validate_config.py (subset covering rev-2 additions)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Ensure scripts/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_config import (  # noqa: E402
    ValidationResult,
    check_challenge_block,
    check_drift_baseline_manifest_schema,
    check_gates_block,
    check_legacy_prop_firm_block,
    check_retrain_gate,
    check_static_peak_consistency,
    check_v23_agreement_decay_gates,
    check_v23_swap_handshake,
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


def test_prop_firm_block_fails_at_hpo():
    """S504: training stages now FAIL (escalated from WARN)."""
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True, "profit_target_pct": 0.10}}}
    check_legacy_prop_firm_block(cfg, "hpo", r)
    assert r.warnings == []
    assert any("env.prop_firm" in f for f in r.failures)


def test_prop_firm_block_fails_at_l1_multiseed():
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True}}}
    check_legacy_prop_firm_block(cfg, "l1-multiseed", r)
    assert r.warnings == []
    assert r.failures


def test_prop_firm_block_fails_at_wf():
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True}}}
    check_legacy_prop_firm_block(cfg, "wf", r)
    assert r.warnings == []
    assert r.failures


def test_prop_firm_block_fails_at_paper_deploy():
    """Paper-deploy is the hard gate: migrated deploys must use env.risk:."""
    r = ValidationResult()
    cfg = {"env": {"prop_firm": {"enabled": True, "profit_target_pct": 0.10}}}
    check_legacy_prop_firm_block(cfg, "paper-deploy", r)
    assert r.warnings == []
    assert any("env.prop_firm" in f for f in r.failures)


def test_migration_message_is_actionable():
    """Failure message should name the new keys."""
    r = ValidationResult()
    check_legacy_prop_firm_block(
        {"env": {"prop_firm": {}}}, "paper-deploy", r
    )
    msg = r.failures[0] if r.failures else ""
    assert "env.risk" in msg
    assert "challenge" in msg


def test_prop_firm_block_with_intentional_marker_is_silent(tmp_path):
    """S504: the 3 A/B-control configs carry a DO-NOT-MIGRATE marker that
    suppresses the FAIL — the validator must honor it on all stages."""
    cfg_path = tmp_path / "intentional_legacy.yaml"
    cfg_path.write_text(
        "# INTENTIONAL: legacy V7 control arm — DO NOT MIGRATE to env.risk:.\n"
        "env:\n  prop_firm:\n    enabled: true\n",
        encoding="utf-8",
    )
    cfg = {"env": {"prop_firm": {"enabled": True}}}
    for stage in ("hpo", "l1-multiseed", "wf", "paper-deploy"):
        r = ValidationResult()
        check_legacy_prop_firm_block(cfg, stage, r, config_path=cfg_path)
        assert r.warnings == [], f"unexpected WARN at stage={stage}: {r.warnings}"
        assert r.failures == [], f"unexpected FAIL at stage={stage}: {r.failures}"


def test_prop_firm_block_without_marker_still_fails_with_path(tmp_path):
    """A config_path is provided but the file lacks the marker — must still FAIL."""
    cfg_path = tmp_path / "unmigrated.yaml"
    cfg_path.write_text(
        "env:\n  prop_firm:\n    enabled: true\n", encoding="utf-8"
    )
    r = ValidationResult()
    check_legacy_prop_firm_block(
        {"env": {"prop_firm": {"enabled": True}}}, "hpo", r, config_path=cfg_path
    )
    assert any("env.prop_firm" in f for f in r.failures)


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


# ---------------------------------------------------------------------------
# check_drift_baseline_manifest_schema (S495 Open Question #5)
# ---------------------------------------------------------------------------


def _prop_firm_cfg(baseline: str | None = None) -> dict:
    cfg = {
        "wandb": {"tags": ["prop-firm"]},
        "env": {
            "risk": {"enabled": True},
        },
        "challenge": {"enabled": True, "phase": "step1"},
        "drift": {"enabled": True},
    }
    if baseline is not None:
        cfg["drift"]["baseline_path"] = baseline
    return cfg


def test_baseline_manifest_check_skips_non_prop_firm():
    r = ValidationResult()
    cfg = {"env": {}, "drift": {"enabled": True, "baseline_path": "missing.json"}}
    check_drift_baseline_manifest_schema(cfg, r)
    assert r.failures == [] and r.warnings == []


def test_baseline_manifest_check_skips_when_drift_disabled():
    r = ValidationResult()
    cfg = _prop_firm_cfg("missing.json")
    cfg["drift"]["enabled"] = False
    check_drift_baseline_manifest_schema(cfg, r)
    assert r.warnings == []


def test_baseline_manifest_check_silent_when_path_missing(tmp_path):
    r = ValidationResult()
    cfg = _prop_firm_cfg(str(tmp_path / "no_such_file.json"))
    check_drift_baseline_manifest_schema(cfg, r)
    assert r.warnings == []  # remote-host case


def test_baseline_manifest_check_silent_for_seed_report_with_field(tmp_path):
    p = tmp_path / "seed_report.json"
    p.write_text(json.dumps({
        "protocol": "v2.2_stage_2_seed_report",
        "challenge_target_hit_rate_by_seed": {"42": {}},
    }))
    r = ValidationResult()
    check_drift_baseline_manifest_schema(_prop_firm_cfg(str(p)), r)
    assert r.warnings == []


def test_baseline_manifest_check_warns_for_seed_report_missing_field(tmp_path):
    p = tmp_path / "seed_report.json"
    p.write_text(json.dumps({"protocol": "v2.2_stage_2_seed_report"}))
    r = ValidationResult()
    check_drift_baseline_manifest_schema(_prop_firm_cfg(str(p)), r)
    assert any("challenge_target_hit_rate_by_seed" in w for w in r.warnings)


def test_baseline_manifest_check_warns_for_ensemble_report_missing_either_field(tmp_path):
    p = tmp_path / "ensemble_report.json"
    p.write_text(json.dumps({
        "protocol": "v2.2_stage_2_5_ensemble_report",
        "challenge_target_hit_rate_by_seed": {"42": {}},
        # ensemble_challenge_target_hit_rate intentionally missing
    }))
    r = ValidationResult()
    check_drift_baseline_manifest_schema(_prop_firm_cfg(str(p)), r)
    assert any("ensemble_challenge_target_hit_rate" in w for w in r.warnings)


def test_baseline_manifest_check_silent_for_ensemble_with_both(tmp_path):
    p = tmp_path / "ensemble_report.json"
    p.write_text(json.dumps({
        "protocol": "v2.2_stage_2_5_ensemble_report",
        "challenge_target_hit_rate_by_seed": {"42": {}},
        "ensemble_challenge_target_hit_rate": {"step1": {}},
    }))
    r = ValidationResult()
    check_drift_baseline_manifest_schema(_prop_firm_cfg(str(p)), r)
    assert r.warnings == []


def test_baseline_manifest_check_silent_on_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    r = ValidationResult()
    check_drift_baseline_manifest_schema(_prop_firm_cfg(str(p)), r)
    assert r.warnings == []  # malformed file isn't this check's concern


def test_baseline_manifest_check_strips_app_prefix(monkeypatch, tmp_path):
    """Live configs use container path /app/baselines/...; the check should
    transparently resolve that against repo-root for best-effort host validation."""
    fake_repo = tmp_path / "repo"
    fake_baseline = fake_repo / "baselines/x/seed_report.json"
    fake_baseline.parent.mkdir(parents=True)
    fake_baseline.write_text(json.dumps({"protocol": "v2.2_stage_2_seed_report"}))
    # Point the check's repo_root resolution at our fake tree by monkeypatching
    # the module's __file__ via a wrapper.
    import scripts.validate_config as vc
    monkeypatch.setattr(vc, "__file__", str(fake_repo / "scripts" / "validate_config.py"))
    r = ValidationResult()
    check_drift_baseline_manifest_schema(
        _prop_firm_cfg("/app/baselines/x/seed_report.json"), r,
    )
    assert any("challenge_target_hit_rate_by_seed" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# check_v23_agreement_decay_gates  (v2.3 §8.2-extension)
# ---------------------------------------------------------------------------


_FULL_AGREEMENT_GATES = {
    "agreement_flat_delta_warn": 0.20,
    "agreement_flat_delta_crit": 0.40,
    "agreement_flat_window_bars": 2000,
}


def _ensemble_cfg(rule: str, *, prop_firm: bool = True, gates: dict | None = None) -> dict:
    tags = ["live"]
    if prop_firm:
        tags.append("FTMO")
    return {
        "wandb": {"tags": tags},
        "agent": {"ensemble": {"bundle_path": "x.tar.gz", "aggregation_rule": rule}},
        "gates": {"drift": dict(gates or {})},
    }


def test_agreement_decay_skipped_for_non_prop_firm():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg("ens_agreement", prop_firm=False), r,
    )
    assert r.failures == []
    assert r.passed == []


def test_agreement_decay_skipped_for_non_consensus_rule():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg("ens_mean", prop_firm=True), r,
    )
    assert r.failures == []
    assert r.passed == []


def test_agreement_decay_fails_when_keys_missing_for_consensus_rule():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg("ens_agreement", prop_firm=True), r,
    )
    assert any("agreement_flat" in f for f in r.failures)
    assert any("ens_agreement" in f for f in r.failures)


def test_agreement_decay_fails_when_partial_keys_present():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg(
            "ens_majority", prop_firm=True,
            gates={"agreement_flat_delta_warn": 0.20},
        ),
        r,
    )
    assert any("agreement_flat_delta_crit" in f for f in r.failures)


def test_agreement_decay_passes_with_all_keys():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg("ens_agreement", prop_firm=True, gates=_FULL_AGREEMENT_GATES),
        r,
    )
    assert r.failures == []
    assert any("agreement-decay keys" in p for p in r.passed)


def test_agreement_decay_warn_ge_crit_rejected():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg(
            "ens_agreement", prop_firm=True,
            gates={
                "agreement_flat_delta_warn": 0.50,
                "agreement_flat_delta_crit": 0.30,
                "agreement_flat_window_bars": 2000,
            },
        ),
        r,
    )
    assert any("warn < crit" in f for f in r.failures)


def test_agreement_decay_window_too_small_rejected():
    r = ValidationResult()
    check_v23_agreement_decay_gates(
        _ensemble_cfg(
            "ens_agreement", prop_firm=True,
            gates={
                "agreement_flat_delta_warn": 0.20,
                "agreement_flat_delta_crit": 0.40,
                "agreement_flat_window_bars": 50,
            },
        ),
        r,
    )
    assert any("too small" in f for f in r.failures)


# ---------------------------------------------------------------------------
# check_v23_swap_handshake (v2.3 §4.5 step 6)
# ---------------------------------------------------------------------------


def test_swap_handshake_skipped_for_non_prop_firm():
    cfg = {
        "wandb": {"tags": ["live"]},
        "agent": {"ensemble": {"bundle_path": "x.tar.gz"}},
        "safety": {"kill_file": "/app/state/kf"},
    }
    r = ValidationResult()
    check_v23_swap_handshake(cfg, r)
    assert r.passed == []
    assert r.warnings == []
    assert r.failures == []


def test_swap_handshake_skipped_when_no_bundle():
    cfg = {
        "wandb": {"tags": ["FTMO"]},
        "agent": {"ensemble": {"seeds": [42, 1]}},  # legacy retro-apply path
        "safety": {"kill_file": "/app/state/kf"},
    }
    r = ValidationResult()
    check_v23_swap_handshake(cfg, r)
    assert r.passed == []
    assert r.failures == []


def test_swap_handshake_skipped_when_no_kill_file():
    # Already FAILed by check_paper_deploy upstream — bail silently to
    # avoid duplicate noise.
    cfg = {
        "wandb": {"tags": ["FTMO"]},
        "agent": {"ensemble": {"bundle_path": "x.tar.gz"}},
        "safety": {},
    }
    r = ValidationResult()
    check_v23_swap_handshake(cfg, r)
    assert r.failures == []


def test_swap_handshake_warns_without_explicit_state_file():
    cfg = {
        "wandb": {"tags": ["FTMO"]},
        "agent": {"ensemble": {"bundle_path": "x.tar.gz"}},
        "safety": {"kill_file": "/app/state/kf"},
    }
    r = ValidationResult()
    check_v23_swap_handshake(cfg, r)
    assert any("last_bundle_file not declared" in w for w in r.warnings)


def test_swap_handshake_passes_with_explicit_state_file():
    cfg = {
        "wandb": {"tags": ["Velotrade"]},
        "agent": {"ensemble": {"bundle_path": "x.tar.gz"}},
        "safety": {
            "kill_file": "/app/state/kf",
            "last_bundle_file": "/app/state/last_bundle.json",
            "swap_approved_file": "/app/state/kf.swap_approved",
        },
    }
    r = ValidationResult()
    check_v23_swap_handshake(cfg, r)
    assert r.failures == []
    assert any("last_bundle.json" in p for p in r.passed)
    assert any("swap_approved" in p for p in r.passed)


# ---------------------------------------------------------------------------
# check_retrain_gate (X6 — Protocol v2 §4.5 retrain automation)
# ---------------------------------------------------------------------------

def _retrain_cfg(*, prop_firm=True, with_policy=True, with_gates=True, **policy_over):
    """Build a minimal live config for check_retrain_gate."""
    cfg: dict = {
        "wandb": {"tags": ["prop-firm"] if prop_firm else ["research"]},
        "agent": {"checkpoint_path": ""},  # bundle deploy: empty agent ckpt
    }
    if with_gates:
        cfg["gates"] = {"retrain": {"cost_drift_ratio": 1.20,
                                    "cost_drift_window_trades": 100}}
    if with_policy:
        policy = {
            "enabled": True,
            "last_trained_date": "2026-02-01",
            "staleness_cap_days": 180,
            "validation_pf_baseline": 2.46,
            "oos_pf_floor_ratio": 0.70,
            "dd_trigger_ratio": 0.50,
            "gate_window_days": 90,
            "feature_drift_ks_stat_threshold": 0.20,
            "backtest_config_ref": "configs/x.yaml",
            "data_file_path": "data/x.parquet",
            "checkpoint_path": "checkpoints/x/checkpoint_final.pth",
        }
        policy.update(policy_over)
        cfg["retrain_policy"] = policy
    return cfg


def test_retrain_gate_wellformed_passes():
    r = ValidationResult()
    check_retrain_gate(_retrain_cfg(), r)
    assert r.failures == []
    assert any("cost-drift keys present" in p for p in r.passed)
    assert any("retrain_policy block present" in p for p in r.passed)


def test_retrain_gate_cost_ratio_le_one_fails():
    r = ValidationResult()
    cfg = _retrain_cfg()
    cfg["gates"]["retrain"]["cost_drift_ratio"] = 1.0
    check_retrain_gate(cfg, r)
    assert any("cost_drift_ratio must be a number > 1.0" in f for f in r.failures)


def test_retrain_gate_window_must_be_int_ge_one():
    r = ValidationResult()
    cfg = _retrain_cfg()
    cfg["gates"]["retrain"]["cost_drift_window_trades"] = 0
    check_retrain_gate(cfg, r)
    assert any("cost_drift_window_trades must be an int >= 1" in f for f in r.failures)


def test_retrain_gate_missing_gates_retrain_warns_propfirm():
    r = ValidationResult()
    check_retrain_gate(_retrain_cfg(with_gates=False), r)
    assert r.failures == []  # WARN, not FAIL
    assert any("gates.retrain absent" in w for w in r.warnings)


def test_retrain_gate_missing_policy_warns_propfirm():
    r = ValidationResult()
    check_retrain_gate(_retrain_cfg(with_policy=False), r)
    assert r.failures == []
    assert any("retrain_policy absent" in w for w in r.warnings)


def test_retrain_gate_non_propfirm_missing_is_silent():
    r = ValidationResult()
    check_retrain_gate(_retrain_cfg(prop_firm=False, with_policy=False,
                                    with_gates=False), r)
    assert r.failures == []
    assert r.warnings == []


def test_retrain_gate_enabled_missing_required_key_fails():
    r = ValidationResult()
    cfg = _retrain_cfg()
    del cfg["retrain_policy"]["validation_pf_baseline"]
    check_retrain_gate(cfg, r)
    assert any("missing required key" in f for f in r.failures)


def test_retrain_gate_bundle_deploy_no_checkpoint_fails():
    # Bundle deploy: agent.checkpoint_path empty AND no retrain_policy.checkpoint_path.
    r = ValidationResult()
    cfg = _retrain_cfg()
    del cfg["retrain_policy"]["checkpoint_path"]
    check_retrain_gate(cfg, r)
    assert any("no checkpoint to backtest" in f for f in r.failures)


def test_retrain_gate_agent_checkpoint_fallback_ok():
    # Solo deploy: no retrain_policy.checkpoint_path but agent.checkpoint_path set.
    r = ValidationResult()
    cfg = _retrain_cfg()
    del cfg["retrain_policy"]["checkpoint_path"]
    cfg["agent"]["checkpoint_path"] = "checkpoints/solo/checkpoint_final.pth"
    check_retrain_gate(cfg, r)
    assert not any("no checkpoint to backtest" in f for f in r.failures)


def test_retrain_gate_oos_floor_out_of_range_fails():
    r = ValidationResult()
    cfg = _retrain_cfg(oos_pf_floor_ratio=1.5)
    check_retrain_gate(cfg, r)
    assert any("oos_pf_floor_ratio must be in (0, 1]" in f for f in r.failures)


def test_retrain_gate_disabled_policy_warns_no_keychecks():
    r = ValidationResult()
    cfg = _retrain_cfg(enabled=False)
    del cfg["retrain_policy"]["validation_pf_baseline"]  # would FAIL if validated
    check_retrain_gate(cfg, r)
    assert r.failures == []
    assert any("retrain_policy.enabled is false" in w for w in r.warnings)


def test_retrain_gate_bad_last_trained_date_fails():
    r = ValidationResult()
    cfg = _retrain_cfg(last_trained_date="not-a-date")
    check_retrain_gate(cfg, r)
    assert any("last_trained_date not ISO-parseable" in f for f in r.failures)


# ---------------------------------------------------------------------------
# check_gates_block — gates may be inline OR delegated to ensemble.gates_file
#
# Pre-existing bug: check_gates_block read only the inline `gates:` key and
# FAILed "Missing `gates:` block" for configs using the documented
# `ensemble.gates_file` indirection (e.g. configs/tailwind_v1_challenge.yaml),
# even though five other call sites already resolve it via
# _load_ensemble_gates_overlay. Both shapes are pinned below.
# ---------------------------------------------------------------------------

def _write_gates_file(tmp_path: Path, gates: dict) -> Path:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(exist_ok=True)
    path = cfg_dir / "ws_challenge.gates.yaml"
    path.write_text(yaml.safe_dump({"gates": gates}, sort_keys=False))
    return path


def test_gates_block_inline_passes():
    r = ValidationResult()
    check_gates_block({"gates": {"rl_beats_linear": 1.10}}, r)
    assert r.failures == []
    assert any("gates block present (1 keys, from inline)" in p for p in r.passed)


def test_gates_block_from_ensemble_gates_file_passes(tmp_path, monkeypatch):
    """The reported repro shape: no inline `gates:`, only ensemble.gates_file."""
    monkeypatch.chdir(tmp_path)
    _write_gates_file(tmp_path, {"rl_beats_linear": 1.10, "g_cost_gap": 0.5})
    cfg = {"ensemble": {"gates_file": "configs/ws_challenge.gates.yaml"}}
    r = ValidationResult()
    check_gates_block(cfg, r)
    assert r.failures == [], f"unexpected failures: {r.failures}"
    assert any("gates block present (2 keys" in p for p in r.passed)


def test_gates_block_inline_and_file_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_gates_file(tmp_path, {"g_cost_gap": 0.5})
    cfg = {
        "gates": {"rl_beats_linear": 1.10},
        "ensemble": {"gates_file": "configs/ws_challenge.gates.yaml"},
    }
    r = ValidationResult()
    check_gates_block(cfg, r)
    assert r.failures == []
    assert any("gates block present (2 keys, from inline + " in p for p in r.passed)


def test_gates_block_absent_entirely_still_fails():
    """Negative tripwire — the check must not become a no-op."""
    r = ValidationResult()
    check_gates_block({"env": {}}, r)
    assert any("Missing `gates:` block" in f for f in r.failures)


def test_gates_block_empty_inline_still_fails():
    r = ValidationResult()
    check_gates_block({"gates": {}}, r)
    assert any("Missing `gates:` block" in f for f in r.failures)


def test_gates_block_unresolvable_gates_file_fails_with_path(tmp_path, monkeypatch):
    """A dangling gates_file must FAIL and name the path, not silently pass."""
    monkeypatch.chdir(tmp_path)
    cfg = {"ensemble": {"gates_file": "configs/does_not_exist.yaml"}}
    r = ValidationResult()
    check_gates_block(cfg, r)
    assert any(
        "Missing `gates:` block" in f and "configs/does_not_exist.yaml" in f
        for f in r.failures
    )


def test_gates_file_present_but_no_gates_key_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "empty.gates.yaml").write_text(yaml.safe_dump({"provenance": {"x": 1}}))
    cfg = {"ensemble": {"gates_file": "configs/empty.gates.yaml"}}
    r = ValidationResult()
    check_gates_block(cfg, r)
    assert any("Missing `gates:` block" in f for f in r.failures)


def test_gates_block_non_mapping_fails_without_crashing():
    """`gates:` as a list must FAIL cleanly, not raise out of the overlay."""
    r = ValidationResult()
    check_gates_block({"gates": ["rl_beats_linear"]}, r)
    assert any("must be a mapping" in f for f in r.failures)
