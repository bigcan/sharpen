#!/usr/bin/env python3
"""Stage 2.5-R Sensitivity Audit — Protocol v2.6 (S553, v2.7-A C4).

Runs the 3x3 (deadband_threshold x max_leverage) config sensitivity sweep on a
PROMOTE'd workstream's existing verdict. Each cell re-runs the trained policy
through the test split with cell-specific env knobs (frozen weights). The
9-cell grid is resolved into an edge-stability verdict per
`.agent/artifacts/protocol_v27_a_sensitivity_audit_architecture.md` ADR-4/ADR-5.

Usage:
    python scripts/stage_2_5_r_sensitivity_audit.py \\
        --workstream <name> \\
        [--config <l1_multiseed_yaml>] \\
        [--gates-file <ws>_ensemble.gates.yaml] \\
        [--out-suffix v2_6] \\
        [--parallel N] \\
        [--wandb-run-id <id>] \\
        [--dry-run]

Exit codes:
    0 — sensitivity_audit completed; edge_stability.decision = PASS
    1 — sensitivity_audit completed; edge_stability.decision = FAIL
    2 — sensitivity_audit completed; decision = UNKNOWN_INSUFFICIENT_NEIGHBORS
    3 — pre-flight failure (missing verdict / config / deployed_config / etc.)
    4 — per-cell rollout failure (env crash / OOM / ...)

This is offline-only — does NOT touch live containers. Trajectory parquets
are written under `results/<ws>_ensemble/sensitivity_audit/<cell_label>/`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.eval.sensitivity_audit import (  # noqa: E402
    CellResult,
    CellSpec,
    InvariantViolation,
    PfXCheckResult,
    build_grid,
    build_sensitivity_audit_block,
    center_index,
    mdd_from_pv,
    pf_from_pv,
    resolve_edge_stability,
    run_cell,
    write_verdict_v26,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage-2.5-r-sensitivity")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNKNOWN = 2
EXIT_PREFLIGHT = 3
EXIT_ROLLOUT = 4


# ---------------------------------------------------------------------------
# Resolution helpers (workstream → paths / config / checkpoints)
# ---------------------------------------------------------------------------


def _locate_verdict(workstream: str, results_root: Path) -> Path:
    """Resolve `results/<workstream>_ensemble/verdict.json`."""
    return results_root / f"{workstream}_ensemble" / "verdict.json"


def _locate_gates_file(workstream: str, configs_root: Path) -> Path:
    """Resolve `configs/<workstream>_ensemble.gates.yaml`."""
    return configs_root / f"{workstream}_ensemble.gates.yaml"


def _load_gates(gates_path: Path) -> Dict[str, Any]:
    with gates_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("gates") or {})


def _resolve_deployed_config(
    verdict: Dict[str, Any],
    cli_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """deployed_config is a NEW v2.6 verdict field. If absent, fall back to
    reading env.deadband_threshold / env.max_leverage from the CLI config.

    The backfill_deployed_config_field.py sidecar (C4 follow-up) writes this
    field into legacy v2.5 verdicts; for fresh v2.6 verdicts the Stage 2.5
    eval script will populate it.
    """
    dep = verdict.get("deployed_config")
    if dep:
        return {
            "deadband_threshold": float(dep["deadband_threshold"]),
            "max_leverage": float(dep["max_leverage"]),
        }
    if cli_config is None:
        return None
    env = cli_config.get("env") or {}
    if "deadband_threshold" not in env or "max_leverage" not in env:
        return None
    return {
        "deadband_threshold": float(env["deadband_threshold"]),
        "max_leverage": float(env["max_leverage"]),
    }


# ---------------------------------------------------------------------------
# Cell execution wrappers (parallel + sequential + dry-run)
# ---------------------------------------------------------------------------


def _dry_run_cell(spec: CellSpec, deployed_config: Dict[str, float]) -> CellResult:
    """Synthetic CellResult for smoke-testing the orchestration path without
    running env rollouts. PF is a deterministic function of distance from
    center so the resolver can exercise PASS / FAIL paths predictably."""
    deployed_db = float(deployed_config["deadband_threshold"])
    deployed_ml = float(deployed_config["max_leverage"])
    # Distance from center in normalized units (each axis step ~0.05 / 0.5×)
    d_db = abs(spec.deadband_threshold - deployed_db) / max(0.05, 1e-9)
    d_ml = abs(spec.max_leverage_mult - 1.0) / 0.5
    distance = d_db + d_ml
    # PF degrades 10% per unit of distance — center=2.0, far corners ~1.4
    pf = max(0.1, 2.0 * (1.0 - 0.10 * distance))
    mdd = -0.05 - 0.01 * distance
    return CellResult(
        spec=spec,
        pf_test=pf,
        mdd_test=mdd,
        n_bars=2880,
        pf_xcheck=PfXCheckResult(
            status="SKIPPED",
            divergence=0.0,
            pf_mid=pf,
            pf_close=pf,
        ),
        halt=False,
        halt_reason=None,
        trajectory_paths=[f"<dry-run:{spec.label()}>"],
    )


def _run_cells_sequential(
    cells_spec: List[CellSpec],
    *,
    config: Dict[str, Any],
    seeds: List[int],
    rule_name: str,
    rule_fn,
    checkpoint_paths: Dict[int, str],
    out_dir: Path,
    device: str,
) -> List[CellResult]:
    """Run cells one-at-a-time. Used as fallback and for the smoke-test path."""
    results: List[CellResult] = []
    for i, spec in enumerate(cells_spec):
        t0 = time.time()
        log.info(
            f"[cell {i+1}/{len(cells_spec)}] {spec.label()} "
            f"(deadband={spec.deadband_threshold}, max_leverage={spec.max_leverage})"
        )
        try:
            result = run_cell(
                config=config,
                spec=spec,
                seeds=seeds,
                rule_name=rule_name,
                rule_fn=rule_fn,
                checkpoint_paths=checkpoint_paths,
                out_dir=out_dir,
                device=device,
            )
        except InvariantViolation as e:
            log.error(f"[cell {spec.label()}] invariant violation: {e}")
            raise
        except Exception as e:  # noqa: BLE001 — surface rollout failures verbatim
            log.error(f"[cell {spec.label()}] rollout failure: {e}")
            raise RuntimeError(f"cell rollout failed: {spec.label()}") from e
        dt = time.time() - t0
        log.info(
            f"[cell {spec.label()}] done in {dt:.1f}s — "
            f"PF={result.pf_test:.3f} MDD={result.mdd_test:.4f} halt={result.halt}"
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# WandB logging (best-effort; no-op when wandb is unavailable)
# ---------------------------------------------------------------------------


def _log_to_wandb(
    parent_run_id: str,
    block: Dict[str, Any],
    workstream: str,
) -> None:
    """Attach sensitivity_audit metrics to the parent Stage 2 WandB run.

    Mirrors v2.5 precedent (Stage 2.5 reuses Stage 2 parent run). No-op when
    wandb is unavailable, when WANDB_MODE=disabled, or when parent run cannot
    be resumed.
    """
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        log.warning("wandb not installed — skipping metric logging")
        return
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        log.info("WANDB_MODE=disabled — skipping metric logging")
        return
    try:
        run = wandb.init(
            id=parent_run_id,
            resume="must",
            project=os.environ.get("WANDB_PROJECT", "FinRL-Pro-DS"),
            entity=os.environ.get("WANDB_ENTITY", "bigcan-chiwin-technology"),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not resume parent run {parent_run_id}: {e}")
        return
    try:
        es = block["edge_stability"]
        wandb.log({
            f"sensitivity_audit/{workstream}/pf_center": es["pf_center"],
            f"sensitivity_audit/{workstream}/pf_inner_min": es["pf_inner_min"],
            f"sensitivity_audit/{workstream}/pf_ratio": es["pf_ratio"],
            f"sensitivity_audit/{workstream}/pf_ratio_floor": es["pf_ratio_floor"],
            f"sensitivity_audit/{workstream}/pass": int(bool(es["pass"])),
            f"sensitivity_audit/{workstream}/n_deployable_neighbors": es["n_deployable_neighbors"],
            f"sensitivity_audit/{workstream}/n_halted_cells": es["n_halted_cells"],
            f"sensitivity_audit/{workstream}/decision_code": {
                "PASS": 0, "FAIL": 1, "UNKNOWN_INSUFFICIENT_NEIGHBORS": 2,
            }.get(es["decision"], -1),
        })
        log.info(f"sensitivity_audit metrics logged to parent run {parent_run_id}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"metric upload failed: {e}")
    finally:
        try:
            run.finish()  # type: ignore[possibly-unbound]
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _git_sha() -> Optional[str]:
    """Short HEAD sha for verdict provenance. None on git failure."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstream", required=True, type=str,
                        help="Workstream name (results/<ws>_ensemble must exist)")
    parser.add_argument("--config", type=Path, default=None,
                        help="L1 multiseed YAML; auto-locate from verdict if omitted")
    parser.add_argument("--gates-file", type=Path, default=None,
                        help="Standalone <ws>_ensemble.gates.yaml; auto-locate if omitted")
    parser.add_argument("--out-suffix", type=str, default="v2_6",
                        help="Output filename suffix (writes verdict_<suffix>.json)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Concurrent cell rollouts (1..9). Default 1 (sequential).")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Parent Stage 2 WandB run id (auto-resolved from verdict if absent)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip env rollouts; emit synthetic cell results for smoke-test")
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="Root of results/ directory (default: results)")
    parser.add_argument("--configs-root", type=Path, default=Path("configs"),
                        help="Root of configs/ directory (default: configs)")
    args = parser.parse_args(argv)

    # --- 1. Locate verdict ---
    verdict_path = _locate_verdict(args.workstream, args.results_root)
    if not verdict_path.exists():
        log.error(f"verdict.json missing: {verdict_path}")
        return EXIT_PREFLIGHT
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    log.info(f"verdict loaded: {verdict_path} (schema {verdict.get('schema_version', '?')})")

    # --- 2. Locate + load gates yaml ---
    gates_path = args.gates_file or _locate_gates_file(args.workstream, args.configs_root)
    if not gates_path.exists():
        log.error(f"gates yaml missing: {gates_path}")
        return EXIT_PREFLIGHT
    gates = _load_gates(gates_path)
    log.info(f"gates loaded: {gates_path} ({len(gates)} keys)")

    # --- 3. Load L1 config (optional in dry-run; required otherwise) ---
    cli_config: Optional[Dict[str, Any]] = None
    if args.config is not None:
        if not args.config.exists():
            log.error(f"--config path missing: {args.config}")
            return EXIT_PREFLIGHT
        with args.config.open(encoding="utf-8") as f:
            cli_config = yaml.safe_load(f) or {}
    elif not args.dry_run:
        log.error(
            "--config required when --dry-run is not set; "
            "auto-location from verdict not implemented (operator must pass explicitly)"
        )
        return EXIT_PREFLIGHT

    # --- 4. Resolve deployed_config (verdict field or fallback to CLI config) ---
    deployed_config = _resolve_deployed_config(verdict, cli_config)
    if deployed_config is None:
        log.error(
            "deployed_config not resolvable. Verdict lacks `deployed_config` "
            "field AND CLI config lacks env.deadband_threshold/max_leverage. "
            "Run scripts/backfill_deployed_config_field.py first OR pass --config."
        )
        return EXIT_PREFLIGHT
    log.info(
        f"deployed_config: deadband={deployed_config['deadband_threshold']} "
        f"max_leverage={deployed_config['max_leverage']}"
    )

    # --- 5. Build grid (SENS-1 invariant defensively re-asserted) ---
    try:
        cells_spec = build_grid(deployed_config, gates)
    except InvariantViolation as e:
        log.error(f"grid construction failed: {e}")
        return EXIT_PREFLIGHT
    log.info(
        f"grid built: {len(cells_spec)} cells "
        f"({sum(1 for c in cells_spec if c.deployable)} deployable)"
    )

    # --- 6. Resolve rule + seeds + checkpoints (real-rollout path only) ---
    rule_name = verdict.get("chosen_rule")
    seeds = verdict.get("seeds") or (verdict.get("ensemble") or {}).get("seeds") or []
    if not args.dry_run:
        if not rule_name:
            log.error("verdict.chosen_rule missing — cannot determine aggregation rule")
            return EXIT_PREFLIGHT
        if not seeds:
            log.error("verdict.seeds missing — cannot resolve checkpoints")
            return EXIT_PREFLIGHT

    # --- 7. Run cells ---
    out_dir = args.results_root / f"{args.workstream}_ensemble"
    t_start = time.time()
    try:
        if args.dry_run:
            log.info("dry-run mode: emitting synthetic cell results")
            cell_results = [_dry_run_cell(s, deployed_config) for s in cells_spec]
        else:
            # Resolve aggregation function + checkpoint paths from verdict + config.
            # For v2.6 C4 we delegate to sg1_xauusd_ensemble_eval helpers via
            # run_cell's lazy imports. The operator passes --config to surface
            # the L1 multiseed config that already declares the agent paths.
            from scripts.sg1_xauusd_ensemble_eval import (  # noqa: E402, WPS433
                _agg_agreement,
                _agg_mean,
                _agg_median,
                _agg_solo,
                _make_pf_weighted,
            )
            rule_lookup = {
                "ens_mean": _agg_mean,
                "ens_median": _agg_median,
                "ens_agreement": _agg_agreement,
            }
            if rule_name == "ens_pf_weighted":
                seed_pfs = (
                    verdict.get("test_solo_pfs")
                    or (verdict.get("ensemble") or {}).get("seed_pfs")
                    or {}
                )
                seed_pfs_typed = {int(k): float(v) for k, v in seed_pfs.items()}
                rule_fn = _make_pf_weighted(seed_pfs_typed)
            elif rule_name in rule_lookup:
                rule_fn = rule_lookup[rule_name]
            elif rule_name.startswith("solo_"):
                rule_fn = _agg_solo(int(rule_name.split("_")[1]))
            else:
                log.error(f"unsupported rule {rule_name!r}")
                return EXIT_PREFLIGHT

            # Checkpoint paths: operator-specified via L1 config under
            # agents.<seed>.ckpt; for v2.6 C4 we accept either dict-shape or
            # legacy SEED_CHECKPOINTS module-level constants.
            agents_section = (cli_config or {}).get("agents") or {}
            checkpoint_paths: Dict[int, str] = {}
            for s in seeds:
                entry = agents_section.get(str(s)) or agents_section.get(s) or {}
                ckpt = entry.get("ckpt") if isinstance(entry, dict) else None
                if not ckpt:
                    log.error(
                        f"agents.{s}.ckpt not declared in L1 config; "
                        "v2.6 sensitivity audit requires per-seed checkpoint paths "
                        "in the config."
                    )
                    return EXIT_PREFLIGHT
                checkpoint_paths[int(s)] = ckpt

            device = "cpu"  # SAC inference is small; CPU is the safe default
            cell_results = _run_cells_sequential(
                cells_spec,
                config=cli_config or {},
                seeds=[int(s) for s in seeds],
                rule_name=rule_name,
                rule_fn=rule_fn,
                checkpoint_paths=checkpoint_paths,
                out_dir=out_dir,
                device=device,
            )
    except InvariantViolation as e:
        log.error(f"invariant violation during rollout: {e}")
        return EXIT_PREFLIGHT
    except Exception as e:  # noqa: BLE001
        log.exception(f"cell rollout failure: {e}")
        return EXIT_ROLLOUT
    wall_time = time.time() - t_start

    # --- 8. Resolve edge stability ---
    verdict_es = resolve_edge_stability(cell_results, deployed_config, gates)
    log.info(
        f"edge_stability: {verdict_es.decision} "
        f"(pf_center={verdict_es.pf_center:.3f}, pf_inner_min={verdict_es.pf_inner_min:.3f}, "
        f"ratio={verdict_es.pf_ratio:.3f}, floor={verdict_es.pf_ratio_floor})"
    )

    # --- 9. Build v2.6 block + write verdict ---
    block = build_sensitivity_audit_block(
        cells=cell_results,
        edge_stability=verdict_es,
        config_axes={
            "deadband_threshold": list(gates.get("sensitivity_deadband_grid", [0.20, 0.25, 0.30])),
            "max_leverage_mults": list(gates.get("sensitivity_max_leverage_mults", [0.5, 1.0, 1.5])),
            "max_leverage_cap": float(
                gates.get(
                    "sensitivity_deployable_max_leverage_cap",
                    deployed_config["max_leverage"],
                ),
            ),
            "max_leverage_values": sorted({c.max_leverage for c in cells_spec}),
        },
        rule=str(rule_name or "<dry-run>"),
        n_seeds_per_cell=len(seeds) if seeds else 1,
        wall_time_seconds=wall_time,
        wandb_run_id=args.wandb_run_id or verdict.get("wandb_run_id"),
        git_sha=_git_sha(),
    )
    out_path = out_dir / f"verdict_{args.out_suffix}.json"
    write_verdict_v26(verdict_path, block, deployed_config, out_path)

    # --- 10. WandB metric upload (best-effort) ---
    parent_run_id = args.wandb_run_id or verdict.get("wandb_run_id")
    if parent_run_id and not args.dry_run:
        _log_to_wandb(parent_run_id, block, args.workstream)

    # --- 11. Exit per decision ---
    code = {
        "PASS": EXIT_PASS,
        "FAIL": EXIT_FAIL,
        "UNKNOWN_INSUFFICIENT_NEIGHBORS": EXIT_UNKNOWN,
    }.get(verdict_es.decision, EXIT_PREFLIGHT)
    log.info(f"verdict written: {out_path} (exit={code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
