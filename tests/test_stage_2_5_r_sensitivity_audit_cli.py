"""Smoke + integration tests for `scripts/stage_2_5_r_sensitivity_audit.py`
(v2.7-A C4, S553).

The full env-rollout path (run_cell with real checkpoints) is exercised by
operator backfill on the 5 deployed strategies. These tests focus on:
  - --dry-run path: orchestration produces a valid v2.6 verdict block
  - exit codes mapped correctly to PASS / FAIL / UNKNOWN
  - pre-flight failures (missing verdict, missing gates, missing
    deployed_config) yield EXIT_PREFLIGHT
  - WandB logging is best-effort (no-op when WANDB_MODE=disabled)

We do NOT exercise the real rollout path here. C4 ships the CLI; C5 docs
+ operator backfill exercise it against real workstreams.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage_2_5_r_sensitivity_audit as audit_cli  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: synthetic workstream tree with verdict + gates
# ---------------------------------------------------------------------------


def _write_workstream_fixture(
    tmp_path: Path,
    *,
    workstream: str = "test_ws",
    deployed_deadband: float = 0.25,
    deployed_max_leverage: float = 1.0,
    chosen_rule: str = "ens_pf_weighted",
    gates_overrides: dict = None,
    include_deployed_config: bool = True,
    decision: str = "PROMOTE",
):
    """Construct a minimal but realistic workstream fixture.

    Returns (results_root, configs_root, verdict_path, gates_path).
    """
    results_root = tmp_path / "results"
    configs_root = tmp_path / "configs"
    ws_dir = results_root / f"{workstream}_ensemble"
    ws_dir.mkdir(parents=True)
    configs_root.mkdir(parents=True)

    verdict = {
        "schema_version": "2.5",
        "chosen_rule": chosen_rule,
        "decision": decision,
        "seeds": [456, 1024, 2025],
        "test_solo_pfs": {"456": 2.55, "1024": 2.47, "2025": 2.50},
    }
    if include_deployed_config:
        verdict["deployed_config"] = {
            "deadband_threshold": deployed_deadband,
            "max_leverage": deployed_max_leverage,
        }
    verdict_path = ws_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    gates = {
        "ensemble_bootstrap_p_pf_promote": 0.90,
        "ensemble_bootstrap_p_mdd_promote": 0.90,
        "edge_stability_pf_ratio_floor": 0.70,
        "sensitivity_deadband_grid": [0.20, 0.25, 0.30],
        "sensitivity_max_leverage_mults": [0.5, 1.0, 1.5],
        "sensitivity_deployable_max_leverage_cap": 1.0,
        "sensitivity_audit_required_min_deployable_neighbors": 4,
        "sensitivity_audit_required": False,
    }
    if gates_overrides:
        gates.update(gates_overrides)
    gates_path = configs_root / f"{workstream}_ensemble.gates.yaml"
    gates_path.write_text(yaml.safe_dump({"gates": gates}), encoding="utf-8")

    return results_root, configs_root, verdict_path, gates_path


# ---------------------------------------------------------------------------
# Smoke test: --dry-run end-to-end
# ---------------------------------------------------------------------------


def test_dry_run_smoke_produces_valid_v26_verdict(tmp_path, monkeypatch):
    """End-to-end --dry-run: synthetic cells → resolver → verdict_v2_6.json."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, verdict_path, gates_path = _write_workstream_fixture(tmp_path)

    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    # _dry_run_cell PF degrades 10% per distance unit → corners ~1.4 against
    # center 2.0 = ratio 0.70 — exactly at floor → PASS (boundary).
    assert code == audit_cli.EXIT_PASS, (
        f"expected EXIT_PASS for synthetic dry-run; got {code}"
    )

    out_path = results_root / "test_ws_ensemble" / "verdict_v2_6.json"
    assert out_path.exists()
    new_verdict = json.loads(out_path.read_text(encoding="utf-8"))

    assert new_verdict["schema_version"] == "2.6"
    assert new_verdict["chosen_rule"] == "ens_pf_weighted"  # preserved
    assert new_verdict["decision"] == "PROMOTE"  # preserved
    assert new_verdict["deployed_config"]["deadband_threshold"] == 0.25

    sa = new_verdict["sensitivity_audit"]
    assert sa["schema"] == "1.0"
    assert sa["rule"] == "ens_pf_weighted"
    assert sa["n_seeds_per_cell"] == 3
    assert len(sa["cells"]) == 9
    assert sa["edge_stability"]["decision"] == "PASS"
    assert sa["thresholds_used"]["edge_stability_pf_ratio_floor"] == 0.70

    # Original verdict.json untouched
    prior = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert prior["schema_version"] == "2.5"
    assert "sensitivity_audit" not in prior


def test_dry_run_with_strict_floor_produces_fail_exit(tmp_path, monkeypatch):
    """Floor=0.95 stricter than synthetic corners → FAIL → exit 1."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, verdict_path, gates_path = _write_workstream_fixture(
        tmp_path,
        gates_overrides={"edge_stability_pf_ratio_floor": 0.95},
    )
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_FAIL

    out_path = results_root / "test_ws_ensemble" / "verdict_v2_6.json"
    new_verdict = json.loads(out_path.read_text(encoding="utf-8"))
    assert new_verdict["sensitivity_audit"]["edge_stability"]["decision"] == "FAIL"


# ---------------------------------------------------------------------------
# SENS-2: run-after-PROMOTE only
# ---------------------------------------------------------------------------


def test_solo_best_fallback_is_informational_exit_pass(tmp_path, monkeypatch):
    """SENS-2: SOLO_BEST_FALLBACK runs informational-only — even a FAIL ratio
    exits EXIT_PASS (non-blocking) but records the true decision in the block."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(
        tmp_path,
        decision="SOLO_BEST_FALLBACK",
        gates_overrides={"edge_stability_pf_ratio_floor": 0.95},  # would FAIL
    )
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_PASS  # informational → never blocks

    out_path = results_root / "test_ws_ensemble" / "verdict_v2_6.json"
    sa = json.loads(out_path.read_text(encoding="utf-8"))["sensitivity_audit"]
    assert sa["informational_only"] is True
    assert sa["stage_2_5_decision"] == "SOLO_BEST_FALLBACK"
    assert sa["edge_stability"]["decision"] == "FAIL"  # true decision preserved


def test_non_promote_decision_refused_preflight(tmp_path, monkeypatch):
    """SENS-2: auditing a non-PROMOTE'd (e.g. REJECT) verdict is a category
    error → EXIT_PREFLIGHT, no verdict written."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(
        tmp_path, decision="REJECT",
    )
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_PREFLIGHT
    assert not (results_root / "test_ws_ensemble" / "verdict_v2_6.json").exists()


# ---------------------------------------------------------------------------
# Pre-flight failure paths (exit 3)
# ---------------------------------------------------------------------------


def test_missing_verdict_returns_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    (tmp_path / "results").mkdir()
    (tmp_path / "configs").mkdir()
    # No workstream dir; locate_verdict will return a missing path.
    code = audit_cli.main([
        "--workstream", "ghost",
        "--results-root", str(tmp_path / "results"),
        "--configs-root", str(tmp_path / "configs"),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_PREFLIGHT


def test_missing_gates_returns_preflight(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(tmp_path)
    gates_path.unlink()  # delete gates file
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_PREFLIGHT


def test_missing_deployed_config_returns_preflight_when_no_cli_config(tmp_path, monkeypatch):
    """v2.6 requires deployed_config in verdict; backfill script populates it.

    Dry-run with no --config and no deployed_config in verdict → pre-flight.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(
        tmp_path, include_deployed_config=False,
    )
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_PREFLIGHT


def test_sens1_violation_in_gates_returns_preflight(tmp_path, monkeypatch):
    """SENS-1: grid center [0.30] != deployed deadband [0.25] → InvariantViolation
    surfaced as pre-flight failure."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(
        tmp_path,
        gates_overrides={"sensitivity_deadband_grid": [0.25, 0.30, 0.35]},
    )
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--dry-run",
    ])
    assert code == audit_cli.EXIT_PREFLIGHT


# ---------------------------------------------------------------------------
# WandB best-effort behavior
# ---------------------------------------------------------------------------


def test_wandb_logging_skipped_when_disabled(tmp_path, monkeypatch, capsys):
    """WANDB_MODE=disabled → audit completes without attempting wandb.init."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(tmp_path)
    code = audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--wandb-run-id", "fake_parent_run",
        "--dry-run",
    ])
    # dry-run path skips wandb call regardless; but verdict still writes.
    assert code in (audit_cli.EXIT_PASS, audit_cli.EXIT_FAIL)
    out_path = results_root / "test_ws_ensemble" / "verdict_v2_6.json"
    assert out_path.exists()
    new_verdict = json.loads(out_path.read_text(encoding="utf-8"))
    # wandb_run_id is recorded in the block (operator override) for audit trail.
    assert new_verdict["sensitivity_audit"]["wandb_run_id"] == "fake_parent_run"


# ---------------------------------------------------------------------------
# Out-suffix override
# ---------------------------------------------------------------------------


def test_out_suffix_override_writes_named_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    results_root, configs_root, _, gates_path = _write_workstream_fixture(tmp_path)
    audit_cli.main([
        "--workstream", "test_ws",
        "--gates-file", str(gates_path),
        "--results-root", str(results_root),
        "--configs-root", str(configs_root),
        "--out-suffix", "audit_backfill_20260530",
        "--dry-run",
    ])
    expected = results_root / "test_ws_ensemble" / "verdict_audit_backfill_20260530.json"
    assert expected.exists()
