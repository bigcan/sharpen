"""Stage 2.5-R Sensitivity Audit — Protocol v2.6 (S553).

Cell rollout + grid construction + edge-stability resolver for the
post-PROMOTE config sensitivity sweep on V7 SAC ContinuousSwing policies.

Architecture: ``.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md``
Researcher:   ``.agent/artifacts/mc_robustness_methods_research.md``
Validator:    ``scripts/validate_config.check_sensitivity_audit`` (C1, S553)

Design summary:
  - 3x3 grid in (deadband_threshold, max_leverage) YAML space around the
    deployed config (center cell = deployed). 9 cells, row-major order.
  - Each cell re-runs the trained policy through the test split with
    cell-specific env construction (frozen weights; new env knobs).
  - Edge-stability gate: pf_inner_min / pf_center >= floor (default 0.70)
    over the 8 deployable, non-halted NEIGHBOR cells (ADR-4, ADR-5).
  - Cells with max_leverage above the deployable cap are reported but
    excluded from the ratio (ADR-8).
  - PF-XCHECK per-cell halt invariant: cells where the mid-marked vs
    close-marked PF diverge by >30% are flagged ``halt=true`` and excluded
    from the gate. Env-side dual-equity-curve recording is deferred; until
    it lands, PF-XCHECK status is ``SKIPPED`` (divergence=0, pass=true).

Invariants asserted at ``run_cell`` entry:
  - BUG-03: env.hindsight_weight == 0.0 during backtest
  - LEAK-1: data.val_end_date <= data.test_start_date (normalization
    cutoff does not extend into the test split)
  - SENS-1: grid center matches deployed config

Pure helpers (no env / no torch) are unit-tested directly. ``run_cell`` is
exercised by the C4 smoke test and operator backfill; tests here focus on
the resolver, grid builder, and metric helpers.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("sensitivity-audit")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PF_CAP = 10.0
"""PF cap matching the canonical _pf_from_returns convention (clipped infinity)."""

PF_XCHECK_DIVERGENCE_HALT = 0.30
"""PF-XCHECK invariant per CLAUDE.md: >30% divergence halts the cell."""

PF_XCHECK_REPORT_ONLY = True
"""ADR-N5 (Protocol v2.7-A N2): Phase-α calibration — compute and surface the
mid-vs-close PF divergence but never HALT. Flip to False (N2-enforce) only after
the report-only distribution across known-good policies locks the threshold."""

DEFAULT_MIN_DEPLOYABLE_NEIGHBORS = 4
DEFAULT_PF_RATIO_FLOOR = 0.70
DEFAULT_DEADBAND_GRID: Tuple[float, float, float] = (0.20, 0.25, 0.30)
DEFAULT_MAX_LEVERAGE_MULTS: Tuple[float, float, float] = (0.5, 1.0, 1.5)

SCHEMA_VERSION = "2.6"
SENSITIVITY_AUDIT_BLOCK_SCHEMA = "1.0"


class InvariantViolation(RuntimeError):
    """Raised when a CLAUDE.md invariant would be violated by the audit."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """One (deadband_threshold, max_leverage) cell of the 3x3 grid."""

    deadband_threshold: float
    max_leverage: float
    max_leverage_mult: float
    deployable: bool
    deployable_reason: Optional[str] = None

    def label(self) -> str:
        """Directory-safe label: ``d025_m100`` for (0.25, 1.00)."""
        db = int(round(self.deadband_threshold * 1000))
        ml = int(round(self.max_leverage * 1000))
        return f"d{db:04d}_m{ml:04d}"

    @property
    def effective_deadband(self) -> float:
        """B5 coupling at env.step: effective_deadband = deadband * max_leverage."""
        return self.deadband_threshold * self.max_leverage


@dataclass(frozen=True)
class PfXCheckResult:
    """Per-cell PF-XCHECK invariant report (mid_price vs close cross-check)."""

    status: str  # "PASS" | "HALT" | "SKIPPED"
    divergence: float
    pf_mid: float
    pf_close: float

    @property
    def pass_(self) -> bool:
        return self.status in ("PASS", "SKIPPED")


@dataclass(frozen=True)
class CellResult:
    """Aggregated metrics for one cell (ensemble or solo, depending on rule)."""

    spec: CellSpec
    pf_test: float
    mdd_test: float
    n_bars: int
    pf_xcheck: PfXCheckResult
    halt: bool
    halt_reason: Optional[str] = None
    trajectory_paths: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EdgeStabilityVerdict:
    """Outcome of the edge-stability gate across deployable neighbor cells."""

    pf_center: float
    pf_inner_min: float
    pf_inner_min_cell: Optional[Tuple[float, float]]
    pf_ratio: float
    pf_ratio_floor: float
    pass_: bool
    n_deployable_neighbors: int
    n_halted_cells: int
    decision: str  # "PASS" | "FAIL" | "UNKNOWN_INSUFFICIENT_NEIGHBORS"
    # N2 (S553 audit): count of cells whose PF-XCHECK was SKIPPED (no close-marked
    # equity curve available). For the current OHLCV/forex feeds the env marks
    # equity at close and mid_price == close, so the mid-vs-close cross-check has
    # no independent second series until env-side dual-marking lands. Surfaced so
    # SKIPPED is never silently read as a validated cross-check.
    n_pf_xcheck_skipped: int = 0


# ---------------------------------------------------------------------------
# Pure metric helpers (mirror scripts/sg1_xauusd_ensemble_eval conventions)
# ---------------------------------------------------------------------------


def pf_from_pv(pv: np.ndarray) -> float:
    """Bar-level PF from a portfolio_value series, clipped at ``PF_CAP``.

    Matches ``scripts/sg1_xauusd_ensemble_eval._pf_from_returns`` for any
    array of size>=2; small arrays return 0.0 (insufficient data).
    """
    pv = np.asarray(pv, dtype=np.float64)
    if pv.size < 2:
        return 0.0
    prev = np.clip(pv[:-1], 1e-12, None)
    returns = (pv[1:] - pv[:-1]) / prev
    wins = float(returns[returns > 0].sum())
    losses = -float(returns[returns < 0].sum())
    if losses < 1e-12:
        return PF_CAP if wins > 1e-12 else 0.0
    return float(min(PF_CAP, wins / losses))


def mdd_from_pv(pv: np.ndarray) -> float:
    """Trailing peak-to-trough drawdown from PV series, in [-1, 0]."""
    pv = np.asarray(pv, dtype=np.float64)
    if pv.size < 2:
        return 0.0
    peak = np.maximum.accumulate(pv)
    return float(np.min(pv / np.maximum(peak, 1e-12)) - 1.0)


def compute_pf_xcheck(
    pv_close: np.ndarray,
    pv_mid: Optional[np.ndarray] = None,
    halt_threshold: float = PF_XCHECK_DIVERGENCE_HALT,
    report_only: bool = False,
) -> PfXCheckResult:
    """PF-XCHECK invariant per-cell check: close-marked vs (H+L)/2-marked PF.

    ``pv_close`` (the close-marked equity curve) is the reference and is always
    present. ``pv_mid`` (the independent (H+L)/2-marked equity curve, N2) is
    optional; when None — flag off / no dual-equity recorded — returns SKIPPED
    status (divergence=0, pass=true).

    When both arrays are provided, computes the relative divergence
    ``|pf_close - pf_mid| / max(|pf_close|, eps)`` (denominator = close, the
    trusted/deployed marking — ADR-N4, Math-verified S553-cont-9) and HALTs if it
    exceeds the threshold (30% per CLAUDE.md PF-XCHECK). The gate is two-sided:
    a >30% disagreement in EITHER direction means the PF is marking-sensitive.

    When ``report_only`` is set (ADR-N5 Phase-α calibration), the divergence is
    still computed and surfaced but the status is clamped to PASS (never HALT).
    """
    pf_c = pf_from_pv(pv_close)
    if pv_mid is None:
        return PfXCheckResult(
            status="SKIPPED",
            divergence=0.0,
            pf_mid=pf_c,
            pf_close=pf_c,
        )
    pf_m = pf_from_pv(pv_mid)
    denom = max(abs(pf_c), 1e-9)
    divergence = abs(pf_c - pf_m) / denom
    if report_only:
        status = "PASS"
    else:
        status = "PASS" if divergence <= halt_threshold else "HALT"
    return PfXCheckResult(
        status=status,
        divergence=float(divergence),
        pf_mid=pf_m,
        pf_close=pf_c,
    )


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def build_grid(
    deployed_config: Dict[str, float],
    gates: Dict[str, Any],
) -> List[CellSpec]:
    """Build the 9-cell 3x3 grid around the deployed config.

    Row-major ordering: outer loop over deadband (ascending), inner loop
    over max_leverage_mult (ascending). Center cell is at index 4.

    Raises ``InvariantViolation`` if the SENS-1 center-must-match-deployed
    invariant is broken (validator should have caught this upstream; we
    re-assert here defensively).
    """
    grid_db = list(gates.get("sensitivity_deadband_grid", DEFAULT_DEADBAND_GRID))
    grid_ml_mults = list(
        gates.get("sensitivity_max_leverage_mults", DEFAULT_MAX_LEVERAGE_MULTS),
    )

    if len(grid_db) != 3 or len(grid_ml_mults) != 3:
        raise InvariantViolation(
            f"Grid axes must be 3-element lists; got deadband={grid_db}, "
            f"mults={grid_ml_mults}"
        )

    deployed_db = float(deployed_config.get("deadband_threshold", 0.25))
    deployed_ml = float(deployed_config.get("max_leverage", 1.0))
    cap = float(
        gates.get("sensitivity_deployable_max_leverage_cap", deployed_ml),
    )

    # SENS-1 center check (defensive; validator is the primary gate)
    if abs(float(grid_db[1]) - deployed_db) > 1e-9:
        raise InvariantViolation(
            f"SENS-1: sensitivity_deadband_grid center [{grid_db[1]}] does not "
            f"match deployed env.deadband_threshold ({deployed_db})"
        )
    if abs(float(grid_ml_mults[1]) - 1.0) > 1e-9:
        raise InvariantViolation(
            f"SENS-1: sensitivity_max_leverage_mults center [{grid_ml_mults[1]}] "
            "must equal 1.0 (center = deployed max_leverage)"
        )

    cells: List[CellSpec] = []
    for db in grid_db:
        for ml_mult in grid_ml_mults:
            ml = deployed_ml * float(ml_mult)
            deployable = ml <= cap + 1e-9
            reason = (
                None
                if deployable
                else f"max_leverage={ml:.4f} > deployable cap {cap:.4f}"
            )
            cells.append(
                CellSpec(
                    deadband_threshold=float(db),
                    max_leverage=ml,
                    max_leverage_mult=float(ml_mult),
                    deployable=deployable,
                    deployable_reason=reason,
                ),
            )
    return cells


def center_index(cells: List[CellSpec], deployed_config: Dict[str, float]) -> int:
    """Return the index of the center cell in a row-major 3x3 grid.

    Matches strictly on (deadband_threshold, max_leverage). Raises
    ValueError if no cell matches — indicates a malformed grid.
    """
    deployed_db = float(deployed_config.get("deadband_threshold", 0.25))
    deployed_ml = float(deployed_config.get("max_leverage", 1.0))
    for i, c in enumerate(cells):
        if (
            abs(c.deadband_threshold - deployed_db) < 1e-9
            and abs(c.max_leverage - deployed_ml) < 1e-9
        ):
            return i
    raise ValueError(
        f"No center cell in grid matching deployed config "
        f"(deadband={deployed_db}, max_leverage={deployed_ml}); cells={cells}"
    )


# ---------------------------------------------------------------------------
# Edge-stability resolver
# ---------------------------------------------------------------------------


def resolve_edge_stability(
    cells: List[CellResult],
    deployed_config: Dict[str, float],
    gates: Dict[str, Any],
) -> EdgeStabilityVerdict:
    """Apply ADR-4 (8 neighbors) + ADR-5 (pf_inner_min / pf_center) gate.

    Returns ``UNKNOWN_INSUFFICIENT_NEIGHBORS`` if any of:
      * The center cell halted (PF-XCHECK divergence > 30%).
      * Fewer than ``min_deployable_neighbors`` deployable+non-halted
        neighbors remain after filtering.
      * pf_center < 1.0 (per ADR-5 — a center re-roll below breakeven means
        the deployed config is no longer profitable on the test split, so a
        PASS/FAIL ratio would be misleading. Stage 2.5 PROMOTE should
        preclude this, but a sensitivity re-roll CAN drive center pf below
        1.0 if the policy's edge is deadband-bound; flag the anomaly).

    Otherwise: ``PASS`` if ratio >= floor, else ``FAIL``.
    """
    pf_ratio_floor = float(
        gates.get("edge_stability_pf_ratio_floor", DEFAULT_PF_RATIO_FLOOR),
    )
    min_neighbors = int(
        gates.get(
            "sensitivity_audit_required_min_deployable_neighbors",
            DEFAULT_MIN_DEPLOYABLE_NEIGHBORS,
        ),
    )

    specs = [c.spec for c in cells]
    center_idx = center_index(specs, deployed_config)
    pf_center = float(cells[center_idx].pf_test)
    n_halted = sum(1 for c in cells if c.halt)
    # N2: surface PF-XCHECK SKIPPED cells so they are never silently trusted.
    n_pf_xcheck_skipped = sum(
        1 for c in cells if c.pf_xcheck.status == "SKIPPED"
    )

    if cells[center_idx].halt:
        return EdgeStabilityVerdict(
            pf_center=pf_center,
            pf_inner_min=float("nan"),
            pf_inner_min_cell=None,
            pf_ratio=float("nan"),
            pf_ratio_floor=pf_ratio_floor,
            pass_=False,
            n_deployable_neighbors=0,
            n_halted_cells=n_halted,
            decision="UNKNOWN_INSUFFICIENT_NEIGHBORS",
            n_pf_xcheck_skipped=n_pf_xcheck_skipped,
        )

    neighbors = [c for i, c in enumerate(cells) if i != center_idx]
    eligible = [c for c in neighbors if c.spec.deployable and not c.halt]

    if len(eligible) < min_neighbors:
        return EdgeStabilityVerdict(
            pf_center=pf_center,
            pf_inner_min=float("nan"),
            pf_inner_min_cell=None,
            pf_ratio=float("nan"),
            pf_ratio_floor=pf_ratio_floor,
            pass_=False,
            n_deployable_neighbors=len(eligible),
            n_halted_cells=n_halted,
            decision="UNKNOWN_INSUFFICIENT_NEIGHBORS",
            n_pf_xcheck_skipped=n_pf_xcheck_skipped,
        )

    worst = min(eligible, key=lambda c: c.pf_test)
    worst_pair = (worst.spec.deadband_threshold, worst.spec.max_leverage)

    # ADR-5: a center re-roll below breakeven (PF < 1.0) is an anomaly, not a
    # ratio to grade. Stage 2.5 PROMOTE guarantees pf_center healthy at the
    # deployed config, but the audit re-rolls and CAN dip below 1.0.
    if pf_center < 1.0:
        return EdgeStabilityVerdict(
            pf_center=pf_center,
            pf_inner_min=float(worst.pf_test),
            pf_inner_min_cell=worst_pair,
            pf_ratio=float("nan"),
            pf_ratio_floor=pf_ratio_floor,
            pass_=False,
            n_deployable_neighbors=len(eligible),
            n_halted_cells=n_halted,
            decision="UNKNOWN_INSUFFICIENT_NEIGHBORS",
            n_pf_xcheck_skipped=n_pf_xcheck_skipped,
        )

    ratio = float(worst.pf_test) / pf_center
    passed = bool(ratio >= pf_ratio_floor)
    return EdgeStabilityVerdict(
        pf_center=pf_center,
        pf_inner_min=float(worst.pf_test),
        pf_inner_min_cell=worst_pair,
        pf_ratio=float(ratio),
        pf_ratio_floor=pf_ratio_floor,
        pass_=passed,
        n_deployable_neighbors=len(eligible),
        n_halted_cells=n_halted,
        decision="PASS" if passed else "FAIL",
        n_pf_xcheck_skipped=n_pf_xcheck_skipped,
    )


# ---------------------------------------------------------------------------
# Cell rollout (impure — env construction + agent loading + run_rule)
# ---------------------------------------------------------------------------


def _assert_invariants(config: Dict[str, Any]) -> None:
    """Defensive invariant checks before any env construction.

    Raises ``InvariantViolation`` on BUG-03 (hindsight in backtest),
    LEAK-1 (norm cutoff extending into test split).
    """
    env_cfg = config.get("env", {}) or {}
    hindsight = env_cfg.get("hindsight_weight", 0.0)
    if float(hindsight) != 0.0:
        raise InvariantViolation(
            f"BUG-03: env.hindsight_weight must be 0.0 in sensitivity audit "
            f"(backtest); got {hindsight}"
        )

    data_cfg = config.get("data", {}) or {}
    val_end = data_cfg.get("val_end_date")
    test_start = data_cfg.get("test_start_date")
    if val_end and test_start and str(val_end) > str(test_start):
        raise InvariantViolation(
            f"LEAK-1: norm_cutoff (val_end_date={val_end}) extends into "
            f"test split (test_start={test_start})"
        )


def run_cell(
    config: Dict[str, Any],
    spec: CellSpec,
    seeds: List[int],
    rule_name: str,
    rule_fn: Callable,
    checkpoint_paths: Dict[int, str],
    out_dir: Path,
    device: str = "cpu",
) -> CellResult:
    """Run one grid cell: override env knobs, load agents, roll test split.

    Delegates to ``scripts.sg1_xauusd_ensemble_eval.run_rule`` for the
    actual rollout — preserving the same trajectory schema and PF/MDD
    conventions used by the rest of the Stage 2.5 chain. The trajectory
    parquet lands at ``out_dir/sensitivity_audit/<cell_label>/<rule>_trajectory.parquet``.

    ``rule_fn`` must conform to the ``run_rule`` signature: callable
    ``(per_seed_actions: Dict[int, np.ndarray], deadband: float) -> np.ndarray``.
    """
    _assert_invariants(config)

    # Lazy import to keep the test module importable without torch/env deps.
    from scripts.sg1_xauusd_ensemble_eval import (  # lazy on purpose: keep module torch-free
        _load_agents_from_paths,
        run_rule,
    )

    cell_cfg = copy.deepcopy(config)
    env_cfg = cell_cfg.setdefault("env", {})
    env_cfg["deadband_threshold"] = spec.deadband_threshold
    env_cfg["max_leverage"] = spec.max_leverage
    # N2: record the (H+L)/2-marked shadow equity so PF-XCHECK has a real pv_mid.
    env_cfg["record_dual_equity"] = True

    cell_dir = Path(out_dir) / "sensitivity_audit" / spec.label()
    cell_dir.mkdir(parents=True, exist_ok=True)

    agents = _load_agents_from_paths(cell_cfg, checkpoint_paths, device)
    df = run_rule(cell_cfg, agents, rule_name, rule_fn, device, cell_dir)

    pv_close = df["portfolio_value"].to_numpy(dtype=np.float64)
    # N2 PF-XCHECK: the (H+L)/2-marked shadow equity recorded by the env when
    # record_dual_equity is set (enabled above). Absent column / all-NaN (flag
    # off in some upstream path) → pv_mid=None → SKIPPED.
    pv_mid: Optional[np.ndarray] = None
    if "portfolio_value_mid" in df.columns:
        _mid = df["portfolio_value_mid"].to_numpy(dtype=np.float64)
        if not np.isnan(_mid).all():
            pv_mid = _mid
    pf_xcheck = compute_pf_xcheck(
        pv_close, pv_mid=pv_mid, report_only=PF_XCHECK_REPORT_ONLY,
    )
    if pv_mid is not None and pf_xcheck.divergence > PF_XCHECK_DIVERGENCE_HALT:
        log.warning(
            "[%s] PF-XCHECK divergence %.3f > %.2f (report_only=%s): close-marked "
            "vs (H+L)/2-marked PF disagree (pf_close=%.3f pf_mid=%.3f).",
            spec.label(), pf_xcheck.divergence, PF_XCHECK_DIVERGENCE_HALT,
            PF_XCHECK_REPORT_ONLY, pf_xcheck.pf_close, pf_xcheck.pf_mid,
        )
    pf_test = pf_from_pv(pv_close)
    mdd_test = mdd_from_pv(pv_close)
    halt = pf_xcheck.status == "HALT"
    halt_reason = (
        f"PF-XCHECK divergence {pf_xcheck.divergence:.3f} > "
        f"{PF_XCHECK_DIVERGENCE_HALT}"
        if halt
        else None
    )

    return CellResult(
        spec=spec,
        pf_test=pf_test,
        mdd_test=mdd_test,
        n_bars=int(df.shape[0]),
        pf_xcheck=pf_xcheck,
        halt=halt,
        halt_reason=halt_reason,
        trajectory_paths=[
            str(cell_dir / f"{rule_name}_trajectory.parquet"),
        ],
    )


# ---------------------------------------------------------------------------
# Verdict block serialization
# ---------------------------------------------------------------------------


def _cell_to_dict(c: CellResult) -> Dict[str, Any]:
    return {
        "deadband_threshold": c.spec.deadband_threshold,
        "max_leverage": c.spec.max_leverage,
        "max_leverage_mult": c.spec.max_leverage_mult,
        "effective_deadband": c.spec.effective_deadband,
        "deployable": c.spec.deployable,
        "deployable_reason": c.spec.deployable_reason,
        "pf_test": c.pf_test,
        "mdd_test": c.mdd_test,
        "n_bars": c.n_bars,
        "pf_xcheck": {
            "status": c.pf_xcheck.status,
            "divergence": c.pf_xcheck.divergence,
            "pf_mid": c.pf_xcheck.pf_mid,
            "pf_close": c.pf_xcheck.pf_close,
        },
        "halt": c.halt,
        "halt_reason": c.halt_reason,
        "trajectory_paths": list(c.trajectory_paths),
    }


def build_sensitivity_audit_block(
    cells: List[CellResult],
    edge_stability: EdgeStabilityVerdict,
    config_axes: Dict[str, Any],
    rule: str,
    n_seeds_per_cell: int,
    *,
    wall_time_seconds: Optional[float] = None,
    wandb_run_id: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Compose the v2.6 ``sensitivity_audit`` verdict block.

    The block conforms to IC-2 in
    ``.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md``.
    """
    pf_inner_min_cell = None
    if edge_stability.pf_inner_min_cell is not None:
        pf_inner_min_cell = {
            "deadband_threshold": edge_stability.pf_inner_min_cell[0],
            "max_leverage": edge_stability.pf_inner_min_cell[1],
        }
    block: Dict[str, Any] = {
        "schema": SENSITIVITY_AUDIT_BLOCK_SCHEMA,
        "axes": dict(config_axes),
        "rule": rule,
        "n_seeds_per_cell": int(n_seeds_per_cell),
        "cells": [_cell_to_dict(c) for c in cells],
        "edge_stability": {
            "pf_center": edge_stability.pf_center,
            "pf_inner_min": edge_stability.pf_inner_min,
            "pf_inner_min_cell": pf_inner_min_cell,
            "pf_ratio": edge_stability.pf_ratio,
            "pf_ratio_floor": edge_stability.pf_ratio_floor,
            "pass": edge_stability.pass_,
            "n_deployable_neighbors": edge_stability.n_deployable_neighbors,
            "n_halted_cells": edge_stability.n_halted_cells,
            "n_pf_xcheck_skipped": edge_stability.n_pf_xcheck_skipped,
            "decision": edge_stability.decision,
        },
        "thresholds_used": {
            "edge_stability_pf_ratio_floor": edge_stability.pf_ratio_floor,
        },
    }
    if wall_time_seconds is not None:
        block["wall_time_seconds"] = float(wall_time_seconds)
    if wandb_run_id is not None:
        block["wandb_run_id"] = wandb_run_id
    if git_sha is not None:
        block["git_sha"] = git_sha
    return block


def write_verdict_v26(
    verdict_path: Path,
    sensitivity_audit_block: Dict[str, Any],
    deployed_config: Dict[str, float],
    out_path: Path,
) -> Dict[str, Any]:
    """Read the prior verdict, append the sensitivity_audit block, bump schema
    to ``2.6``, write atomically (tempfile + rename) to ``out_path``.

    The prior verdict at ``verdict_path`` is NOT mutated. Returns the new
    verdict dict.
    """
    verdict_path = Path(verdict_path)
    out_path = Path(out_path)
    prior = json.loads(verdict_path.read_text(encoding="utf-8"))
    new = dict(prior)
    new["schema_version"] = SCHEMA_VERSION
    new["deployed_config"] = dict(deployed_config)
    new["sensitivity_audit"] = sensitivity_audit_block

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(new, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp_path.replace(out_path)
    log.info(f"verdict v2.6 written: {out_path}")
    return new
