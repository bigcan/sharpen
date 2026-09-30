"""TAILWIND Tier-2 N14: gate thresholds live in the gates yaml, never in the enforcing scripts,
and a missing key fails closed. Also the T7-P09 overlay routing in validate_config.

Data-free: only yaml, module imports and the validator's pure checks are exercised.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ the old literals are gone
@pytest.mark.parametrize("rel,literal", [
    ("scripts/research/audit_tailwind_book.py", "0.389 /"),
    ("scripts/research/audit_tailwind_book.py", "<= 0.50"),
    ("scripts/research/audit_tailwind_book.py", "< 0.40"),
    ("scripts/research/audit_tailwind_book.py", "n_splits=16"),
    ("scripts/research/breadth_expansion.py", "<= 0.50"),
    ("scripts/research/breadth_expansion.py", ">= 0.601"),
    ("scripts/research/tailwind_forward_path_render.py", "< 0.005"),
    ("scripts/research/tailwind_forward_path_render.py", "- 15.0"),
    ("scripts/research/tailwind_executor_recompute.py", "0.0334 *"),
    ("scripts/research/tailwind_dsr_consistent.py", "REPRO_TOL = "),
    ("scripts/ftmo_compliance_report.py", "MIN_ACTIVE_DAYS = 4"),
    ("scripts/cross_asset_pipeline.py", "defaults = {"),
    ("scripts/options_vol_pipeline.py", "defaults = {"),
])
def test_threshold_literal_is_not_in_the_enforcing_script(rel, literal):
    assert literal not in (ROOT / rel).read_text(encoding="utf-8")


# ------------------------------------------------------------------ audit_tailwind_book
def test_audit_gates_load_and_values_are_unchanged():
    atb = _load("scripts/research/audit_tailwind_book.py", "atb_n14")
    g = atb.load_audit_gates()
    ab = g["audit_book"]
    assert (ab["bab_leak_gap_max"], ab["max_corr_mom_bab_subperiod"], ab["min_oos_sharpe"],
            ab["min_sharpe_harsh_cost"], ab["legacy_honest_haircut_sharpe"]) == (0.10, 0.40, 0.30, 0.40, 0.389)
    assert ab["pbo"] == {"max_pbo": 0.50, "n_splits": 16, "metric": "sharpe", "strip_warmup": "deployed_live"}
    assert set(g["subperiods"]) == {"2006-09", "2010-15", "2016-20", "2021-26"}     # one source (stage4)


def test_audit_gates_fail_closed_on_a_missing_key(tmp_path, monkeypatch):
    atb = _load("scripts/research/audit_tailwind_book.py", "atb_n14b")
    g = yaml.safe_load((ROOT / "configs" / "tailwind_v1.gates.yaml").read_text(encoding="utf-8"))
    del g["audit_book"]["subperiods_source"]
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(g), encoding="utf-8")
    monkeypatch.setattr(atb, "GATES_PATH", p)
    with pytest.raises(KeyError, match="subperiods_source"):
        atb.load_audit_gates()


def test_recorded_references_carry_their_own_n():
    tdc = _load("scripts/research/tailwind_dsr_consistent.py", "tdc_n14")
    ref = tdc.REF()
    assert ref["n_trials"] == 24                                  # not dsr_n_trials (77)
    assert ref["recorded"] == {"research_honest": 0.896, "research_curated": 0.9744, "executor": 0.871}
    assert ref["external_haircut"] == 0.389 and ref["repro_tol"] == 0.002


# ------------------------------------------------------------------ ftmo compliance
def test_ftmo_thresholds_come_from_the_stage4_gates(tmp_path):
    fc = _load("scripts/ftmo_compliance_report.py", "ftmo_n14")
    assert fc.load_thresholds() == (4, 0.5)
    bad = tmp_path / "g.yaml"
    bad.write_text(yaml.safe_dump({"stage4_recent_oos": {"compliance": {"ftmo_min_active_days": 4}}}),
                   encoding="utf-8")
    with pytest.raises(KeyError):
        fc.load_thresholds(bad)


# ------------------------------------------------------------------ validator overlay (T7-P09)
@pytest.fixture()
def vc():
    import scripts.validate_config as m
    return m


def _cfg_with_overlay(tmp_path, gates: dict) -> dict:
    g = tmp_path / "w.gates.yaml"
    g.write_text(yaml.safe_dump({"gates": gates}), encoding="utf-8")
    return {"ensemble": {"gates_file": str(g)}}


def test_overlay_resolves_from_the_repo_root_not_the_cwd(vc, tmp_path, monkeypatch):
    cfg = {"ensemble": {"gates_file": "configs/tailwind_v1_challenge_v2.gates.yaml"}}
    monkeypatch.chdir(tmp_path)                                    # validator run from elsewhere
    assert "rl_beats_linear" in vc._load_ensemble_gates_overlay(cfg)


def test_wf_check_sees_overlay_keys(vc, tmp_path):
    cfg = _cfg_with_overlay(tmp_path, {"wf_windows": 99})
    seen = {}
    real = vc._load_ensemble_gates_overlay
    assert real(cfg)["wf_windows"] == 99
    src = (ROOT / "scripts" / "validate_config.py").read_text(encoding="utf-8")
    for fn in ("check_hpo", "check_l1_multiseed", "check_wf"):
        body = src.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        seen[fn] = "_load_ensemble_gates_overlay(cfg)" in body
    assert all(seen.values()), seen


def test_live_consumed_checks_stay_inline(vc, tmp_path):
    """The live engine and forward runner read config['gates'] inline; a drift/safe-mode key that
    exists only in the gates file is invisible to them, so the validator must NOT count it."""
    drift = {k: 0.1 for k in vc._V22_DRIFT_GATE_KEYS}
    cfg = _cfg_with_overlay(tmp_path, {"drift": drift, "safe_mode": {k: 1 for k in vc._V22_SAFE_MODE_KEYS}})
    r = vc.ValidationResult()
    vc.check_drift_safemode_gates(cfg, r)
    assert any("missing v2.2" in m for m in r.failures + r.warnings)
    src = (ROOT / "scripts" / "validate_config.py").read_text(encoding="utf-8")
    for fn in ("check_drift_safemode_gates", "check_v23_agreement_decay_gates", "check_retrain_gate"):
        body = src.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
        assert "_load_ensemble_gates_overlay" not in body, fn

