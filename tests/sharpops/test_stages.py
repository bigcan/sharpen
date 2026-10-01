"""SharpOps P1: the stage decides the pipeline phases, and a run with no stage is refused.

Before 2026-09-30 run_full_pipeline ran HPO -> train -> backtest whatever --stage said, and a
missing --stage only warned (Protocol v2 audit 2026-09-29, §4 P1). These pin the contract both
ways, plus the run_walk_forward metric reader that returned None for every fold.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sharpen.sharpops.stages import (
    BACKTEST,
    HPO,
    TRAIN,
    StageError,
    resolve_stage_phases,
    validator_stage,
)

ROOT = Path(__file__).resolve().parents[2]


def test_missing_stage_is_refused():
    for stage in (None, ""):
        with pytest.raises(StageError, match="no --stage"):
            resolve_stage_phases(stage)


def test_each_stage_runs_only_its_phases():
    assert resolve_stage_phases("hpo") == {HPO}
    assert resolve_stage_phases("l1-multiseed") == {TRAIN, BACKTEST}
    assert resolve_stage_phases("wf") == {TRAIN, BACKTEST}
    assert resolve_stage_phases("oos", has_checkpoint=True) == {BACKTEST}
    assert resolve_stage_phases("ensemble-confirm", has_checkpoint=True) == {BACKTEST}
    assert resolve_stage_phases("all") == {HPO, TRAIN, BACKTEST}      # explicit fused only


def test_only_all_runs_hpo_with_training():
    fused = [s for s in ("hpo", "l1-multiseed", "wf", "all")
             if {HPO, TRAIN} <= resolve_stage_phases(s)]
    assert fused == ["all"]


def test_eval_stages_need_a_checkpoint():
    for stage in ("oos", "ensemble-confirm"):
        with pytest.raises(StageError, match="--checkpoint"):
            resolve_stage_phases(stage)


def test_foreign_and_unknown_stages_are_refused():
    for stage in ("data-prep", "paper-deploy", "stage4", "WF"):
        with pytest.raises(StageError):
            resolve_stage_phases(stage)


def test_backtest_only_narrows_and_conflicts():
    assert resolve_stage_phases("wf", backtest_only=True, has_checkpoint=True) == {BACKTEST}
    with pytest.raises(StageError, match="conflicts"):
        resolve_stage_phases("hpo", backtest_only=True, has_checkpoint=True)


def test_validator_stage():
    assert validator_stage("all") == "hpo"
    assert validator_stage("wf") == "wf"


def test_every_pipeline_stage_is_a_validator_stage():
    import re

    src = (ROOT / "scripts" / "validate_config.py").read_text(encoding="utf-8")
    valid = set(re.search(r"VALID_STAGES = \(([^)]*)\)", src).group(1).replace('"', "").replace(" ", "").split(","))
    from sharpen.sharpops.stages import STAGE_PHASES
    assert {validator_stage(s) for s in STAGE_PHASES} <= valid


# ------------------------------------------------------------------ run_walk_forward metrics
@pytest.fixture(scope="module")
def rwf():
    spec = importlib.util.spec_from_file_location("run_walk_forward", ROOT / "scripts" / "run_walk_forward.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fold_metrics_reads_run_backtest_keys(rwf):
    metrics = {"wf_fold_03/sharpe": 1.2, "wf_fold_03/sharpe_daily": 0.4,
               "wf_fold_03/max_drawdown": -0.05, "wf_fold_03/total_return": 0.031,
               "wf_fold_03/profit_factor": 1.1}
    r = rwf.fold_metrics(metrics, "wf_fold_03", 3, ("a", "b"), ("b", "c"), ("c", "d"))
    assert (r["sharpe"], r["max_drawdown"], r["profit_factor"]) == (1.2, -0.05, 1.1)
    assert r["total_return_pct"] == pytest.approx(3.1)
    assert r["status"] == "completed"


def test_fold_metrics_empty_is_not_completed(rwf):
    r = rwf.fold_metrics({}, "wf_fold_00", 0, None, None, None)
    assert r["status"] == "no_metrics" and r["sharpe"] is None


def test_every_pipeline_launcher_names_a_stage():
    """Every launcher that builds a run_full_pipeline.py command must pass --stage."""
    import re

    # A command build: `python ... -u scripts/run_full_pipeline.py ...` in a string, or the
    # path joined as `"scripts" / "run_full_pipeline.py"` into an argv list. The stage must
    # appear in the next ~300 characters (the same command).
    cmd = re.compile(r'python[^\n"]*\s-u\s+scripts/run_full_pipeline\.py|"scripts"\s*/\s*"run_full_pipeline\.py"')
    offenders = []
    for p in (ROOT / "scripts").glob("*.py"):
        src = p.read_text(encoding="utf-8", errors="replace")
        for m in cmd.finditer(src):
            if "--stage" not in src[m.end():m.end() + 300]:
                offenders.append(f"{p.name}:{src.count(chr(10), 0, m.start()) + 1}")
    assert offenders == []
