#!/usr/bin/env python3
"""Stage 3.5 Observation-Noise Robustness — Protocol v2.7-B (S553-cont-25).

Standalone, canonical entrypoint (IC-5/ADR-7; there is no
``run_full_pipeline.py --stage`` dispatcher). Re-rolls a PROMOTE'd workstream's
frozen ensemble through price-path-randomized OHLC and gates on edge survival.

Per fold (same window list + per-fold checkpoints as Stage 3 WF):
  - a sigma=0 nominal rollout (in-band reference, ADR-3), plus
  - per sigma-level x n_noise_seeds noisy rollouts (multiplicative log-noise on
    raw 1-min OHLC; the REAL handler recomputes every feature, the X2 causal
    map, EMA-Z, ATR from the noised bars — single source of truth, ADR-1/2).
PF/MDD are aggregated into a per-(fold,sigma) edge-survival verdict and written
to ``results/<ws>_ensemble/obs_noise_report.json`` (IC-2). Metrics also land on
a dedicated WandB Stage 3.5 run under ``obs_noise/*`` (ADR-10).

Usage:
    python scripts/stage_3_5_obs_noise.py \\
        --workstream <name> \\
        --config <wf yaml>                 # splitter + data + ensemble + features
        [--gates-file <ws>_ensemble.gates.yaml]
        [--rule <ens_pf_weighted|...>]     # default: gates aggregation_rule / ensemble.rule
        [--checkpoint-pattern "..."]       # default: ensemble.checkpoint_pattern
        [--parallel N] [--keep-noised]
        [--out-dir results/<ws>_ensemble] [--scratch-dir <tmp>]
        [--device cpu] [--wandb-run-id <name>] [--dry-run]

Exit codes:
    0 — obs_noise gate PASS
    1 — obs_noise gate FAIL
    2 — UNKNOWN_INSUFFICIENT_FOLDS
    3 — pre-flight failure (missing config/gates/checkpoints/sigma-levels/invariant)
    4 — rollout crash

Offline-only — does NOT touch live containers. ``--dry-run`` substitutes a
deterministic synthetic rollout (still materializes + swaps real noised parquets)
so the full wiring can be smoke-tested without torch/checkpoints.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data.splitter import RollingWindowSplitter  # noqa: E402
from finrl_pro_ds.eval.obs_noise import (  # noqa: E402  (torch-free core)
    InvariantViolation,
    NoiseSpec,
    run_obs_noise_stage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage-3.5-obs-noise")

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNKNOWN = 2
EXIT_PREFLIGHT = 3
EXIT_ROLLOUT = 4

_DECISION_EXIT = {
    "PASS": EXIT_PASS,
    "FAIL": EXIT_FAIL,
    "UNKNOWN_INSUFFICIENT_FOLDS": EXIT_UNKNOWN,
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_gates(path: Path) -> Dict[str, Any]:
    return dict(_load_yaml(path).get("gates") or {})


def _build_specs(gates: Dict[str, Any]) -> List[NoiseSpec]:
    """NoiseSpec list from gates.obs_noise_sigma_levels + obs_noise_n_seeds.

    Sorted by sigma so the per-(fold,sigma,k) seed derivation (level_idx) is
    reproducible regardless of YAML key order (MATH-OBS-5)."""
    levels = gates.get("obs_noise_sigma_levels") or {}
    n_seeds = int(gates.get("obs_noise_n_seeds", 10))
    specs = [
        NoiseSpec(sigma=float(sigma), label=str(label), n_seeds=n_seeds)
        for label, sigma in levels.items()
    ]
    specs.sort(key=lambda s: s.sigma)
    return specs


def _build_folds(wf_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Splitter folds (same call as run_wf_ensemble)."""
    split_cfg = wf_cfg.get("splitter", {}) or {}
    data_cfg = wf_cfg.get("data", {}) or {}
    splitter = RollingWindowSplitter(
        train_months=split_cfg.get("train_months", 6),
        val_months=split_cfg.get("val_months", 1),
        test_months=split_cfg.get("test_months", 1),
        step_months=split_cfg.get("step_months", 1),
        buffer_days=split_cfg.get("buffer_days", 0),
    )
    return splitter.split(
        start_date=data_cfg.get("start_date", "2025-01-06"),
        end_date=data_cfg.get("end_date", "2026-03-31"),
    )


def _resolve_rule(args_rule: Optional[str], gates_cfg: Dict[str, Any],
                  ens: Dict[str, Any]) -> Optional[str]:
    """Stage 3.5 uses the SAME aggregation rule Stage 3 promotes: CLI override,
    else the gates top-level ``aggregation_rule``, else ``ensemble.rule``."""
    return args_rule or gates_cfg.get("aggregation_rule") or ens.get("rule")


def _synthetic_rollout():
    """Dry-run rollout: deterministic PV from the close-return path of whatever
    (possibly noised) parquet the cell config points at. Exercises the real
    noise -> parquet -> data.file_path swap end-to-end without torch."""
    import numpy as np
    import pandas as pd

    def fake(cfg, agents, rule_name, rule_fn, device, out_dir):
        df = pd.read_parquet(cfg["data"]["file_path"])
        ts = pd.to_datetime(df["timestamp"], utc=True)
        ds = pd.Timestamp(cfg["data"]["test_start_date"], tz="UTC")
        de = pd.Timestamp(cfg["data"]["test_end_date"], tz="UTC") + pd.Timedelta(days=1)
        close = df.loc[(ts >= ds) & (ts < de), "close"].to_numpy(dtype=np.float64)
        if close.size < 2:
            return pd.DataFrame({"portfolio_value": [100000.0, 100000.0]})
        rets = np.concatenate([[0.0], np.diff(close) / close[:-1]])
        return pd.DataFrame({"portfolio_value": 100000.0 * np.cumprod(1.0 + rets)})

    return fake


def _git_sha() -> Optional[str]:
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


def _log_to_wandb(run_name: str, report: Dict[str, Any], workstream: str) -> None:
    """Dedicated Stage 3.5 WandB run (ADR-10). Best-effort; no-op when wandb is
    unavailable or WANDB_MODE=disabled."""
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
            name=run_name,
            project=os.environ.get("WANDB_PROJECT", "FinRL-Pro-DS"),
            entity=os.environ.get("WANDB_ENTITY", "bigcan-chiwin-technology"),
            job_type="stage_3_5_obs_noise",
            tags=["stage-3.5", "obs-noise", workstream],
            config={"thresholds_used": report.get("thresholds_used", {})},
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"wandb.init failed: {e}")
        return
    try:
        er = report["edge_robustness"]
        metrics: Dict[str, Any] = {
            "obs_noise/decision_code": {
                "PASS": 0, "FAIL": 1, "UNKNOWN_INSUFFICIENT_FOLDS": 2,
            }.get(er["decision"], -1),
            "obs_noise/n_folds": er["n_folds"],
            "obs_noise/n_folds_graded": er["n_folds_graded"],
        }
        for label, ps in (er.get("per_sigma") or {}).items():
            metrics[f"obs_noise/{label}/worst_pf_ratio"] = ps.get("worst_pf_ratio")
            metrics[f"obs_noise/{label}/worst_mdd_degr_pp"] = ps.get("worst_mdd_degr_pp")
            metrics[f"obs_noise/{label}/pf_pass"] = int(bool(ps.get("pf_pass")))
            metrics[f"obs_noise/{label}/mdd_pass"] = int(bool(ps.get("mdd_pass")))
        wandb.log(metrics)
        log.info(f"obs_noise metrics logged to dedicated run {run_name}")
    except Exception as e:  # noqa: BLE001
        log.warning(f"metric upload failed: {e}")
    finally:
        try:
            run.finish()  # type: ignore[possibly-unbound]
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def _resolve_fold_checkpoints_all(
    pattern: str, seeds: List[int], folds: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[int, Dict[int, str]]]:
    """Resolve per-seed checkpoints for each fold via glob pattern.

    Folds whose checkpoints are missing are SKIPPED (logged) — Stage 3.5 grades
    only the folds Stage 3 actually trained. Returns the surviving folds
    (re-indexed 0..k-1, original index preserved via each fold's test range) and
    their checkpoint map keyed by the new contiguous index.
    """
    from scripts.sg1_xauusd_ensemble_eval import (  # lazy: pulls torch
        _resolve_fold_checkpoints,
    )

    resolved_folds: List[Dict[str, Any]] = []
    fold_checkpoints: Dict[int, Dict[int, str]] = {}
    for orig_idx, fold in enumerate(folds):
        try:
            ckpts = _resolve_fold_checkpoints(pattern, seeds, orig_idx)
        except (FileNotFoundError, ValueError) as e:
            log.warning(f"fold {orig_idx} (test {fold['test'].start[:10]}): skipped — {e}")
            continue
        new_idx = len(resolved_folds)
        resolved_folds.append(fold)
        fold_checkpoints[new_idx] = ckpts
    return resolved_folds, fold_checkpoints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstream", required=True, type=str,
                        help="Workstream name (results/<ws>_ensemble output root)")
    parser.add_argument("--config", required=True, type=Path,
                        help="WF YAML: splitter + data + ensemble + features")
    parser.add_argument("--gates-file", type=Path, default=None,
                        help="Standalone <ws>_ensemble.gates.yaml; auto-locate if omitted")
    parser.add_argument("--rule", type=str, default=None,
                        help="Aggregation rule; default gates.aggregation_rule / ensemble.rule")
    parser.add_argument("--checkpoint-pattern", type=str, default=None,
                        help="Per-fold checkpoint glob; default ensemble.checkpoint_pattern")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Recorded in the report; rollouts run serially in this cut")
    parser.add_argument("--keep-noised", action="store_true",
                        help="Retain noised parquets under scratch for audit (default: delete)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default results/<ws>_ensemble")
    parser.add_argument("--scratch-dir", type=Path, default=None,
                        help="Noised-parquet scratch root (default <out-dir>/obs_noise_scratch)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Dedicated Stage 3.5 run name (default obsnoise_<ws>_<ts>)")
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic rollout (still swaps real noised parquets); no torch")
    args = parser.parse_args(argv)

    # --- 1. Load WF config ---
    if not args.config.exists():
        log.error(f"--config missing: {args.config}")
        return EXIT_PREFLIGHT
    wf_cfg = _load_yaml(args.config)

    # --- 2. Load gates (block) ---
    gates_path = args.gates_file or (
        args.configs_root / f"{args.workstream}_ensemble.gates.yaml"
    )
    if not Path(gates_path).exists():
        log.error(f"gates yaml missing: {gates_path}")
        return EXIT_PREFLIGHT
    gates_cfg = _load_yaml(gates_path)
    gates = dict(gates_cfg.get("gates") or {})
    log.info(f"gates loaded: {gates_path} ({len(gates)} keys)")

    # --- 3. Build sigma-level specs ---
    specs = _build_specs(gates)
    if not specs:
        log.error("gates.obs_noise_sigma_levels missing/empty — nothing to sweep (v2.7 Stage 3.5)")
        return EXIT_PREFLIGHT
    log.info(
        "sigma levels: "
        + ", ".join(f"{s.label}(σ={s.sigma}, n={s.n_seeds})" for s in specs)
    )

    # --- 4. Folds ---
    folds_all = _build_folds(wf_cfg)
    log.info(f"splitter produced {len(folds_all)} folds")
    if not folds_all:
        log.error("splitter produced 0 folds — check data.start_date/end_date + splitter block")
        return EXIT_PREFLIGHT

    # --- 5. Ensemble: seeds, rule, seed_pfs ---
    ens = wf_cfg.get("ensemble", {}) or {}
    seeds = [int(s) for s in (ens.get("seeds") or []) if s is not None]
    rule_name = _resolve_rule(args.rule, gates_cfg, ens)
    if not rule_name:
        log.error("no aggregation rule — pass --rule or set gates.aggregation_rule / ensemble.rule")
        return EXIT_PREFLIGHT
    seed_pfs = {int(k): float(v) for k, v in (ens.get("seed_pfs") or {}).items()}
    if not seeds and not args.dry_run:
        log.error("ensemble.seeds empty — cannot resolve checkpoints")
        return EXIT_PREFLIGHT

    # --- 6. Rule fn + checkpoints (real path) or dry-run substitutes ---
    run_kwargs: Dict[str, Any] = {}
    if args.dry_run:
        import numpy as np  # local: keep module torch/numpy-import light
        rule_fn = lambda actions, db: np.array([0.0])  # noqa: E731 — ignored by synthetic rollout
        resolved_folds = folds_all
        fold_checkpoints = {
            i: {s: "<dry-run>" for s in (seeds or [0])} for i in range(len(folds_all))
        }
        run_kwargs["_run_rule"] = _synthetic_rollout()
        run_kwargs["_load_agents"] = lambda cfg, ckpts, device: {"dry": True}
        log.info("dry-run: synthetic rollout (real noise + parquet swap, no torch)")
    else:
        # _resolve_rule_fn rebuilds the (non-picklable) aggregation closure from
        # rule_name + seed_pfs; reused verbatim from the v2.7-A sensitivity CLI.
        from scripts.stage_2_5_r_sensitivity_audit import _resolve_rule_fn  # lazy (torch)
        try:
            rule_fn = _resolve_rule_fn(rule_name, seed_pfs)
        except ValueError as e:
            log.error(str(e))
            return EXIT_PREFLIGHT
        pattern = args.checkpoint_pattern or ens.get("checkpoint_pattern")
        if not pattern:
            log.error("no checkpoint pattern — pass --checkpoint-pattern or set ensemble.checkpoint_pattern")
            return EXIT_PREFLIGHT
        resolved_folds, fold_checkpoints = _resolve_fold_checkpoints_all(
            pattern, seeds, folds_all,
        )
        if not resolved_folds:
            log.error(
                f"no fold had resolvable checkpoints (pattern={pattern!r}); "
                "Stage 3 must have produced per-fold per-seed checkpoints first."
            )
            return EXIT_PREFLIGHT
        log.info(
            f"resolved checkpoints for {len(resolved_folds)}/{len(folds_all)} folds "
            f"(rule={rule_name}, seeds={seeds})"
        )

    # --- 7. Output dirs + run name ---
    out_dir = args.out_dir or (args.results_root / f"{args.workstream}_ensemble")
    scratch_dir = args.scratch_dir or (out_dir / "obs_noise_scratch")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.wandb_run_id or f"obsnoise_{args.workstream}_{ts}"

    # --- 8. Run the stage ---
    try:
        _results, verdict = run_obs_noise_stage(
            config=wf_cfg,
            folds=resolved_folds,
            fold_checkpoints=fold_checkpoints,
            rule_name=rule_name,
            rule_fn=rule_fn,
            specs=specs,
            gates=gates,
            out_dir=out_dir,
            scratch_dir=scratch_dir,
            device=args.device,
            parallel=args.parallel,
            keep_noised=args.keep_noised,
            workstream=args.workstream,
            ensemble_seeds=seeds or None,
            wandb_run_id=run_name,
            git_sha=_git_sha(),
            **run_kwargs,
        )
    except InvariantViolation as e:
        log.error(f"invariant violation (pre-flight): {e}")
        return EXIT_PREFLIGHT
    except (FileNotFoundError, ValueError) as e:
        log.error(f"pre-flight failure: {e}")
        return EXIT_PREFLIGHT
    except Exception as e:  # noqa: BLE001 — surface rollout crashes
        log.exception(f"rollout crash: {e}")
        return EXIT_ROLLOUT

    # --- 9. WandB (best-effort, dedicated run) ---
    report_path = Path(out_dir) / "obs_noise_report.json"
    if not args.dry_run and report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            _log_to_wandb(run_name, report, args.workstream)
        except Exception as e:  # noqa: BLE001
            log.warning(f"wandb logging skipped: {e}")

    # --- 10. Exit per decision ---
    code = _DECISION_EXIT.get(verdict.decision, EXIT_PREFLIGHT)
    log.info(
        f"Stage 3.5 obs-noise: {verdict.decision} "
        f"(graded {verdict.n_folds_graded}/{verdict.n_folds} folds) -> "
        f"report {report_path} (exit={code})"
    )
    for label, ps in verdict.per_sigma.items():
        log.info(
            f"  {label}: worst_pf_ratio={ps['worst_pf_ratio']} (>= {ps['pf_floor']}? "
            f"{ps['pf_pass']}); worst_mdd_degr_pp={ps['worst_mdd_degr_pp']} "
            f"(<= {ps['mdd_buffer_pp']}? {ps['mdd_pass']})"
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
