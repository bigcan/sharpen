#!/usr/bin/env python
"""GMGP1 Training-Volume Study — dispatcher.

Sweeps `training.total_timesteps` in {500K, 1M, 2M, 3M, 5M} at N=5 seeds each
(25 runs total) on GMGP1 SAC Gold 15-min. HPs are frozen from v5 HPO `6xluh826`
(T29 best, copied via `configs/gmgp1_volume_study.yaml`).

Per Protocol v2 Stage 2: one consolidated WandB run per budget, with seed<N>/*
namespaces. The 5 budget-runs are sibling experiments; analysis script joins
them by tag `volume-study`.

Approach: generate 5 derived configs (one per budget) into
`configs/_volume_study_runtime/`, validate each, then call
`scripts/launch_l1_multiseed.py` per budget. Sequential over budgets; the
launcher manages within-budget seed concurrency.

Pre-registered hypothesis gates live in `configs/gmgp1_volume_study.gates.yaml`
and are read by `scripts/analyze_volume_study.py` post-run.

Usage:
    # Smoke test (1 seed at 500K on gpuhub-2:0)
    python scripts/run_volume_study.py --smoke --instance gpuhub-2 --gpu 0

    # Full 25-run sweep, single-instance (sequential over 5 budgets)
    python scripts/run_volume_study.py --full --instance gpuhub-2 --gpu 0 \
        --concurrent 2

    # Full 25-run sweep, cross-GPU fan-out (3 slots, budgets sequential,
    # seeds within each budget distributed across slots)
    python scripts/run_volume_study.py --full \
        --slots gpuhub-1:0,gpuhub-1:1,gpuhub-2:0

    # Dry run: prints commands + writes derived configs but does not launch
    python scripts/run_volume_study.py --full --dry_run

Manifest: writes `results/volume_study_<timestamp>/manifest.json` with
(budget, seed, run_name, status, elapsed_s) entries for the analysis script.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / "configs" / "gmgp1_volume_study.yaml"
GATES_FILE = PROJECT_ROOT / "configs" / "gmgp1_volume_study.gates.yaml"
RUNTIME_DIR = PROJECT_ROOT / "configs" / "_volume_study_runtime"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_l1_multiseed.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_config.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] volume_study - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("volume_study")


def _budget_label(steps: int) -> str:
    """Human-readable budget tag, e.g. 500000 -> '500k', 5_000_000 -> '5m'."""
    if steps >= 1_000_000:
        return f"{steps // 1_000_000}m" if steps % 1_000_000 == 0 else f"{steps / 1_000_000:.1f}m"
    return f"{steps // 1000}k"


def load_gates() -> dict:
    with GATES_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_derived_config(base: dict, total_timesteps: int, label: str,
                         study_id: str) -> Path:
    """Materialize a derived config with overridden total_timesteps."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("training", {})["total_timesteps"] = total_timesteps
    # Tag each run with budget + unique study id for downstream filtering.
    tags = list(cfg.setdefault("wandb", {}).setdefault("tags", []))
    for t in (f"budget-{label}", f"study-id-{study_id}"):
        if t not in tags:
            tags.append(t)
    cfg["wandb"]["tags"] = tags

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNTIME_DIR / f"gmgp1_volume_study_{label}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def validate(cfg_path: Path) -> bool:
    """Run scripts/validate_config.py on a derived config."""
    cmd = [sys.executable, str(VALIDATOR),
           "--config", str(cfg_path), "--stage", "l1-multiseed"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if "status=FAIL" in (proc.stdout + proc.stderr):
        log.error("validator FAIL on %s\n%s", cfg_path.name, proc.stdout)
        return False
    log.info("validator %s on %s",
             "WARN" if "status=WARN" in proc.stdout else "PASS", cfg_path.name)
    return True


def launch_budget(cfg_path: Path, label: str, seeds: list[int],
                  instance: str | None, gpu: str | None, concurrent: int,
                  slots: str | None,
                  timestamp: str, dry_run: bool) -> tuple[int, float]:
    """Invoke launch_l1_multiseed.py for one budget. Blocks until all seeds done.

    When `slots` is set (cross-GPU mode), it's forwarded to the launcher and
    the launcher distributes seeds across slots; `instance/gpu/concurrent`
    are ignored. Otherwise the launcher uses the single-slot fallback.

    Note: the launcher itself appends `_<timestamp>` to the prefix and `-seed{N}`
    to per-seed names. The validate_run_name regex
    (`finrl_pro_ds/utils/naming.py`) is `^[a-z0-9-]+_\\d{8}_\\d{6}$` — descriptive
    id must be lowercase/digits/hyphens only. Do NOT embed timestamps or
    underscores in `prefix` here; study-disambiguation is handled via the
    `study-id-<ts>` wandb tag instead.
    """
    seeds_csv = ",".join(str(s) for s in seeds)
    prefix = f"gmgp1-volume-{label}"
    # Pass the config path RELATIVE to PROJECT_ROOT. deploy_bare_metal forwards
    # it verbatim to the remote run_full_pipeline.py, where the workspace
    # unpacks under `/workspace/DeepScalper/`. Absolute Windows paths
    # (`C:/FinRL/...`) crash on the Linux remote.
    cfg_rel = cfg_path.relative_to(PROJECT_ROOT).as_posix()
    cmd = [
        sys.executable, str(LAUNCHER),
        "--config", cfg_rel,
        "--seeds", seeds_csv,
        "--run_name_prefix", prefix,
    ]
    if slots:
        cmd.extend(["--slots", slots])
        target_str = f"slots={slots}"
    else:
        cmd.extend([
            "--instance", instance or "gpuhub-2",
            "--gpu", gpu or "0",
            "--concurrent", str(concurrent),
        ])
        target_str = f"{instance} (gpu={gpu}, concurrent={concurrent})"
    if dry_run:
        cmd.append("--dry_run")

    log.info("launching budget=%s seeds=%s -> %s", label, seeds_csv, target_str)
    log.info("  cmd: %s", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("budget=%s FAILED (exit=%d, %.0fs)", label, proc.returncode, elapsed)
    else:
        log.info("budget=%s OK (%.0fs)", label, elapsed)
    return proc.returncode, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="Smoke test: 1 seed (123) at 500K only")
    mode.add_argument("--full", action="store_true",
                      help="Full 25-run sweep (5 budgets x 5 seeds)")
    p.add_argument("--instance", default="gpuhub-2",
                   help="Single-slot fallback target instance (default: gpuhub-2). "
                        "Ignored if --slots is set.")
    p.add_argument("--gpu", default="0",
                   help="Single-slot CUDA_VISIBLE_DEVICES (default 0). "
                        "Ignored if --slots is set.")
    p.add_argument("--concurrent", type=int, default=2,
                   help="Single-slot within-budget seed concurrency (default 2 — "
                        "VRAM-bound). Ignored if --slots is set.")
    p.add_argument("--slots", default=None,
                   help="Cross-GPU slot pool: comma-separated host:gpu pairs "
                        "(e.g. 'gpuhub-1:0,gpuhub-1:1,gpuhub-2:0'). When set, "
                        "each budget fans seeds out across these slots; "
                        "concurrency = len(slots). Budgets remain sequential.")
    p.add_argument("--budgets", type=str, default=None,
                   help="Comma-separated subset of budgets (e.g., '500000,1000000'). "
                        "Default: read from gates file.")
    p.add_argument("--dry_run", action="store_true",
                   help="Generate + validate configs, print launch commands, do not spawn")
    args = p.parse_args()

    if not BASE_CONFIG.exists():
        log.error("base config not found: %s", BASE_CONFIG)
        return 2
    if not GATES_FILE.exists():
        log.error("gates file not found: %s", GATES_FILE)
        return 2
    if not LAUNCHER.exists():
        log.error("launcher not found: %s", LAUNCHER)
        return 2

    gates = load_gates()
    all_budgets: list[int] = list(gates["step_budgets"])
    all_seeds: list[int] = list(gates["seeds"])

    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",")]
    elif args.smoke:
        budgets = [500_000]
    else:
        budgets = all_budgets

    if args.smoke:
        seeds = [all_seeds[0]]
    else:
        seeds = all_seeds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"volume_study_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("GMGP1 Volume Study Dispatcher")
    log.info("  base config: %s", BASE_CONFIG.name)
    log.info("  budgets:     %s", budgets)
    log.info("  seeds:       %s", seeds)
    log.info("  total runs:  %d", len(budgets) * len(seeds))
    if args.slots:
        log.info("  slots:       %s (cross-GPU)", args.slots)
    else:
        log.info("  instance:    %s (gpu=%s, concurrent=%d)",
                 args.instance, args.gpu, args.concurrent)
    log.info("  results dir: %s", results_dir.relative_to(PROJECT_ROOT))
    log.info("  dry_run:     %s", args.dry_run)
    log.info("=" * 70)

    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    manifest = {
        "timestamp": timestamp,
        "instance": args.instance,
        "gpu": args.gpu,
        "concurrent": args.concurrent,
        "slots": args.slots,
        "budgets": budgets,
        "seeds": seeds,
        "dry_run": args.dry_run,
        "runs": [],
    }

    # Phase 1: generate + validate all derived configs upfront. If any fails
    # we abort BEFORE launching anything (cheap fail-fast).
    derived_paths: dict[int, Path] = {}
    for steps in budgets:
        label = _budget_label(steps)
        cfg_path = write_derived_config(base_cfg, steps, label, timestamp)
        derived_paths[steps] = cfg_path
        if not validate(cfg_path):
            log.error("aborting: validator rejected %s", cfg_path.name)
            return 3

    # Phase 2: dispatch budgets sequentially (launcher manages seed concurrency).
    for steps in budgets:
        label = _budget_label(steps)
        cfg_path = derived_paths[steps]
        rc, elapsed = launch_budget(
            cfg_path, label, seeds,
            args.instance, args.gpu, args.concurrent,
            args.slots,
            timestamp, args.dry_run,
        )
        manifest["runs"].append({
            "budget": steps,
            "label": label,
            "seeds": seeds,
            "config": str(cfg_path.relative_to(PROJECT_ROOT)),
            "exit_code": rc,
            "elapsed_s": round(elapsed, 1),
        })
        # Persist manifest progressively in case of mid-run failure.
        with (results_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if rc != 0:
            log.error("aborting subsequent budgets: budget=%s failed", label)
            return 1

    fails = [r for r in manifest["runs"] if r["exit_code"] != 0]
    log.info("=" * 70)
    log.info("Dispatcher complete: %d/%d budgets OK",
             len(manifest["runs"]) - len(fails), len(manifest["runs"]))
    log.info("Manifest: %s", (results_dir / "manifest.json").relative_to(PROJECT_ROOT))
    log.info("Next: scripts/analyze_volume_study.py --manifest %s",
             results_dir / "manifest.json")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
