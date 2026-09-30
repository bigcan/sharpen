"""Which pipeline phases a SharpOps stage may run (P1: one stage = one run).

``run_full_pipeline.py`` used to run HPO -> train -> backtest whatever ``--stage`` said;
``--stage`` only triggered config validation (Protocol v2 audit 2026-09-29, §4 P1). Now the
stage decides the phases, a missing stage is refused, and the legacy fused run needs the
explicit ``--stage all``.
"""
from __future__ import annotations

HPO, TRAIN, BACKTEST = "hpo", "train", "backtest"

STAGE_PHASES: dict[str, frozenset[str]] = {
    "hpo": frozenset({HPO}),
    "l1-multiseed": frozenset({TRAIN, BACKTEST}),
    "wf": frozenset({TRAIN, BACKTEST}),
    "ensemble-confirm": frozenset({BACKTEST}),
    "oos": frozenset({BACKTEST}),
    "all": frozenset({HPO, TRAIN, BACKTEST}),   # legacy fused run; explicit opt-in only
}
NOT_PIPELINE_STAGES = {
    "data-prep": "run scripts/clean_ohlcv.py and scripts/build_data_manifest.py",
    "paper-deploy": "use the deploy / live-trading path, not the training pipeline",
}


class StageError(ValueError):
    """The requested stage cannot run in this launcher."""


def resolve_stage_phases(stage: str | None, *, backtest_only: bool = False,
                         has_checkpoint: bool = False) -> frozenset[str]:
    """Phases for ``stage``. Fails closed: no stage, an unknown stage, a stage owned by
    another tool, or a backtest-only stage without a checkpoint all raise StageError."""
    if not stage:
        raise StageError("no --stage given: name the SharpOps stage "
                         f"({', '.join(sorted(STAGE_PHASES))}); '--stage all' is the explicit "
                         "legacy fused HPO+train+backtest run")
    if stage in NOT_PIPELINE_STAGES:
        raise StageError(f"stage {stage!r} is not run by this pipeline: {NOT_PIPELINE_STAGES[stage]}")
    if stage not in STAGE_PHASES:
        raise StageError(f"unknown stage {stage!r}")
    phases = STAGE_PHASES[stage]
    if backtest_only:
        phases = phases & {BACKTEST}
        if not phases:
            raise StageError(f"--backtest_only conflicts with stage {stage!r}, which runs no backtest")
    if TRAIN not in phases and BACKTEST in phases and not has_checkpoint:
        raise StageError(f"stage {stage!r} evaluates an existing model: pass --checkpoint")
    return phases


def validator_stage(stage: str) -> str:
    """The validate_config stage for a pipeline stage (the fused run starts with HPO)."""
    return "hpo" if stage == "all" else stage
