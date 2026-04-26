"""Regression tests for the Stage 2.5 gates-file overlay (S498-cont fix).

The fix lives at scripts/sg1_xauusd_ensemble_eval.py:_load_gates_with_overlay.
Bug: pre-fix, run_stage_2_5_val_selection read gates from the L1 multiseed
config only. v2.3 bootstrap thresholds defined in standalone
<workstream>_ensemble.gates.yaml were never loaded → bootstrap_decision was
LEGACY_GATE_DEFER even when the bootstrap evidence was conclusive
(SG-1-BTC P(PF)=1.0 → AMBIGUOUS_RERUN instead of v2.3 verdict).

These tests pin the merge contract so future refactors don't silently regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sg1_xauusd_ensemble_eval import _load_gates_with_overlay  # noqa: E402


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def test_no_gates_file_returns_l1_gates_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "gates": {
            "ensemble_uplift_min": 1.10,
            "ensemble_ambiguous_min": 1.05,
        }
    }
    out = _load_gates_with_overlay(config)
    assert out == config["gates"]
    # must be a copy, not the same object — caller shouldn't be able to mutate config
    assert out is not config["gates"]


def test_missing_gates_file_warns_and_returns_l1_only(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    config = {
        "ensemble": {"gates_file": "configs/does_not_exist.yaml"},
        "gates": {"ensemble_uplift_min": 1.10},
    }
    with caplog.at_level("WARNING"):
        out = _load_gates_with_overlay(config)
    assert out == {"ensemble_uplift_min": 1.10}
    assert any("not found" in rec.message for rec in caplog.records)


def test_overlay_adds_bootstrap_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    gates_path = cfg_dir / "ws_ensemble.gates.yaml"
    _write_yaml(
        gates_path,
        {
            "gates": {
                "ensemble_bootstrap_p_pf_promote": 0.90,
                "ensemble_bootstrap_p_mdd_promote": 0.90,
                "ensemble_bootstrap_resamples": 10000,
                "ensemble_bootstrap_block_len": None,
            }
        },
    )
    config = {
        "ensemble": {"gates_file": "configs/ws_ensemble.gates.yaml"},
        "gates": {
            "ensemble_uplift_min": 1.10,
            "ensemble_ambiguous_min": 1.05,
        },
    }
    out = _load_gates_with_overlay(config)
    # L1 keys preserved
    assert out["ensemble_uplift_min"] == 1.10
    assert out["ensemble_ambiguous_min"] == 1.05
    # Bootstrap keys merged in — this is the bug-fix surface
    assert out["ensemble_bootstrap_p_pf_promote"] == 0.90
    assert out["ensemble_bootstrap_p_mdd_promote"] == 0.90
    assert out["ensemble_bootstrap_resamples"] == 10000
    assert out["ensemble_bootstrap_block_len"] is None


def test_standalone_wins_on_overlap(tmp_path, monkeypatch):
    """If a key appears in both, the standalone gates_file wins."""
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    gates_path = cfg_dir / "ws_ensemble.gates.yaml"
    _write_yaml(
        gates_path,
        {
            "gates": {
                "ensemble_uplift_min": 1.05,  # standalone says different value
                "ensemble_bootstrap_p_pf_promote": 0.90,
            }
        },
    )
    config = {
        "ensemble": {"gates_file": "configs/ws_ensemble.gates.yaml"},
        "gates": {
            "ensemble_uplift_min": 1.10,  # L1 says 1.10; standalone should override
        },
    }
    out = _load_gates_with_overlay(config)
    assert out["ensemble_uplift_min"] == 1.05  # standalone wins
    assert out["ensemble_bootstrap_p_pf_promote"] == 0.90


def test_absolute_gates_file_path_resolves(tmp_path, monkeypatch):
    """Non-cwd-relative absolute paths must work — operators sometimes pass them."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    gates_path = other / "ws_ensemble.gates.yaml"
    _write_yaml(
        gates_path,
        {"gates": {"ensemble_bootstrap_p_pf_promote": 0.85}},
    )
    monkeypatch.chdir(tmp_path)  # cwd elsewhere from gates file
    config = {
        "ensemble": {"gates_file": str(gates_path)},  # absolute
        "gates": {"ensemble_uplift_min": 1.10},
    }
    out = _load_gates_with_overlay(config)
    assert out["ensemble_bootstrap_p_pf_promote"] == 0.85


def test_real_sg1_btc_gates_yaml_overlays_correctly():
    """End-to-end: load the real SG-1-BTC standalone gates file from the repo
    against a synthesized L1 config and assert v2.3 keys land in the merged
    dict. Pins that the in-tree gate file hasn't drifted to a structure the
    helper can't read."""
    repo_gates = REPO_ROOT / "configs" / "sg1_btc_velotrade_ensemble.gates.yaml"
    assert repo_gates.exists(), f"in-tree gates file moved: {repo_gates}"
    config = {
        "ensemble": {"gates_file": str(repo_gates)},  # absolute → cwd-independent
        "gates": {
            "ensemble_uplift_min": 1.10,
            "ensemble_ambiguous_min": 1.05,
            "ensemble_rule_selection": "val_argmax_pf",
        },
    }
    out = _load_gates_with_overlay(config)
    # The bug-symptom keys must be present after overlay
    assert "ensemble_bootstrap_p_pf_promote" in out
    assert "ensemble_bootstrap_p_mdd_promote" in out
    assert out["ensemble_bootstrap_p_pf_promote"] == 0.90
    assert out["ensemble_bootstrap_p_mdd_promote"] == 0.90
    # L1-only key (rule_selection) is preserved (standalone doesn't define it)
    assert out["ensemble_rule_selection"] == "val_argmax_pf"


def test_real_gmgp1_btc_gates_yaml_overlays_correctly():
    """Twin of SG-1-BTC test for the new GMGP1-BTC gates file (S498-cont)."""
    repo_gates = REPO_ROOT / "configs" / "gmgp1_btc_velotrade_ensemble.gates.yaml"
    assert repo_gates.exists(), f"in-tree gates file moved: {repo_gates}"
    config = {
        "ensemble": {"gates_file": str(repo_gates)},
        "gates": {
            "ensemble_uplift_min": 1.10,
            "ensemble_ambiguous_min": 1.05,
            "ensemble_rule_selection": "val_argmax_pf",
        },
    }
    out = _load_gates_with_overlay(config)
    assert out["ensemble_bootstrap_p_pf_promote"] == 0.90
    assert out["ensemble_bootstrap_p_mdd_promote"] == 0.90
    assert out["ensemble_rule_selection"] == "val_argmax_pf"
