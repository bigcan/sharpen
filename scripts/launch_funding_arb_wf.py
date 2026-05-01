"""Funding-Arb DSAC walk-forward launcher (Protocol v2 Stage 3).

Fans out K parallel `funding_arb_dsac_train_seed.py` invocations, one per
walk-forward window, all with the SAME seed (Stage 2.5 winner) and the SAME
HPs (baked into the L1 multiseed config). Mirrors `launch_l1_multiseed.py`
but iterates window_index instead of seeds.

Each window trains for the configured `total_timesteps` on its own train
split, then evaluates on the window's val + test. Per-window checkpoint +
manifest end up on the deploy host under
`checkpoints/<run_name>_<timestamp>/` and are aggregated post-hoc by
`scripts/funding_arb_dsac_wf_report.py`.

Usage:
    python scripts/launch_funding_arb_wf.py \
        --config configs/funding_arb_dsac_l1_multiseed.yaml \
        --gates configs/funding_arb_dsac_velotrade_wf.gates.yaml \
        --instance gpuhub-2 --gpu 0 --concurrent 8 \
        --run_name_prefix funding-arb-dsac-wf \
        --windows 0,1,2,3,4,5,6,7
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.launch_l1_multiseed import (  # noqa: E402
    DEPLOY_SCRIPT,
    _reserve_instance_setup_slot,
    parse_slots,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("wf_launch")


def run_window(config: Path, slot_pool: "queue.Queue[tuple[str, str]]",
               window_index: int, seed: int, run_name: str,
               shared_run_id: str | None,
               script: str = "scripts/funding_arb_dsac_train_seed.py",
               ) -> tuple[int, int, float, str, str]:
    """Spawn deploy_bare_metal for one window. Same control flow as
    launch_l1_multiseed.run_seed but forwards --window_index."""
    instance, gpu = slot_pool.get()
    try:
        wait_s = _reserve_instance_setup_slot(instance)
        if wait_s > 0:
            logger.info("window %d: waiting %.0fs for %s setup-slot stagger",
                        window_index, wait_s, instance)
            time.sleep(wait_s)
        start = time.time()
        cmd = [
            sys.executable, str(DEPLOY_SCRIPT),
            "--config", config.as_posix(),
            "--instance", instance,
            "--gpu", gpu,
            "--run_name", run_name,
            "--extra_args", f"--seed {seed} --window_index {window_index}",
            "--no_kill",
            "--script", script,
        ]
        child_env = os.environ.copy()
        if shared_run_id:
            child_env["FINRL_WANDB_RUN_ID"] = shared_run_id
            child_env["FINRL_WANDB_NAMESPACE"] = f"window{window_index:02d}"

        logger.info("window %d: launching on %s:%s -> %s%s",
                    window_index, instance, gpu, run_name,
                    " [consolidated]" if shared_run_id else "")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                              env=child_env)
        elapsed = time.time() - start
        if proc.returncode != 0:
            logger.error("window %d FAILED on %s:%s (exit=%d, %.1fs)",
                         window_index, instance, gpu, proc.returncode, elapsed)
            logger.error("  stdout tail: %s", proc.stdout[-500:])
            logger.error("  stderr tail: %s", proc.stderr[-500:])
        else:
            logger.info("window %d OK on %s:%s (%.1fs) — note: deploy-collect "
                        "may return early under S488 consolidation; verify "
                        "manifest mtimes before declaring done",
                        window_index, instance, gpu, elapsed)
        return window_index, proc.returncode, elapsed, instance, gpu
    finally:
        slot_pool.put((instance, gpu))


def _init_consolidated_wandb(run_name_prefix: str, timestamp: str,
                             config: dict, gates_path: Path) -> str | None:
    """Open a parent WandB run and return its id for child consolidation."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not available; running with separate child runs")
        return None
    wb_cfg = config.get("wandb") or {}
    project = wb_cfg.get("project", "FinRL-Pro-DS")
    entity = wb_cfg.get("entity")
    tags = list(wb_cfg.get("tags") or []) + ["wf", "stage3", "consolidated"]
    run = wandb.init(
        project=project, entity=entity,
        name=f"{run_name_prefix}_{timestamp}",
        tags=tags,
        config={
            "stage": 3, "workstream": "funding_arb_dsac",
            "gates_path": str(gates_path),
            "wf_protocol": "v2",
        },
    )
    rid = run.id if run else None
    if rid:
        logger.info("Parent WandB run: https://wandb.ai/%s/%s/runs/%s (id=%s)",
                    entity or "anon", project, rid, rid)
    return rid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="L1 multiseed YAML (HPs baked) — same one used for Stage 2")
    p.add_argument("--gates", required=True, type=Path,
                   help="Velotrade WF gates yaml (configs/funding_arb_dsac_velotrade_wf.gates.yaml)")
    p.add_argument("--instance", default=None,
                   help="Target instance in instances.json (e.g. gpuhub-2)")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (default 0)")
    p.add_argument("--slots", default=None,
                   help="Cross-GPU slot pool: 'host:gpu,host:gpu,...'")
    p.add_argument("--concurrent", type=int, default=4,
                   help="Single-slot mode: max concurrent windows sharing one GPU")
    p.add_argument("--windows", required=True,
                   help="Comma-separated window indices (e.g. 0,1,2,3,4,5,6,7)")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed to use for ALL windows (default: read from --gates `seed:` field)")
    p.add_argument("--run_name_prefix", default="funding-arb-dsac-wf")
    p.add_argument("--script", default="scripts/funding_arb_dsac_train_seed.py")
    p.add_argument("--separate_runs", action="store_true",
                   help="Legacy: one WandB run per window. Default is ONE consolidated parent.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.config.exists():
        logger.error("config not found: %s", args.config); return 2
    if not args.gates.exists():
        logger.error("gates not found: %s", args.gates); return 2

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(args.gates, encoding="utf-8") as f:
        gates = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else gates.get("seed")
    if seed is None:
        logger.error("no seed: pass --seed or set `seed:` in gates yaml"); return 2

    try:
        slots = parse_slots(args.slots, args.instance, args.gpu)
    except ValueError as e:
        logger.error("slot config error: %s", e); return 2

    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    if not windows:
        logger.error("no windows parsed from %r", args.windows); return 2
    if len(set(windows)) != len(windows):
        logger.error("duplicate windows in %r", windows); return 2

    expected_range = gates.get("window_index_range")
    if expected_range and (min(windows) < expected_range[0] or max(windows) > expected_range[1]):
        logger.warning("--windows %s outside gates window_index_range %s",
                       windows, expected_range)

    if args.slots:
        slot_pool: queue.Queue[tuple[str, str]] = queue.Queue()
        for slot in slots:
            slot_pool.put(slot)
        concurrency = len(slots)
    else:
        concurrency = max(1, int(args.concurrent))
        slot_pool = queue.Queue()
        for _ in range(concurrency):
            slot_pool.put(slots[0])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("Funding-Arb DSAC Walk-Forward Launcher (Stage 3)")
    logger.info("=" * 70)
    logger.info("  config:      %s", args.config)
    logger.info("  gates:       %s", args.gates)
    logger.info("  seed (all):  %d", seed)
    logger.info("  windows:     %s (n=%d)", windows, len(windows))
    logger.info("  slots:       %s (concurrency=%d)", slots, concurrency)
    logger.info("  script:      %s", args.script)
    logger.info("  prefix:      %s", args.run_name_prefix)
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("--dry_run set; exiting without dispatch")
        return 0

    shared_run_id = None
    if not args.separate_runs:
        shared_run_id = _init_consolidated_wandb(args.run_name_prefix, timestamp,
                                                 config, args.gates)

    results: list[tuple[int, int, float, str, str]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for w in windows:
            run_name = f"{args.run_name_prefix}-window{w:02d}_{timestamp}"
            futures.append(ex.submit(
                run_window, args.config, slot_pool, w, seed, run_name,
                shared_run_id, args.script,
            ))
        for fut in as_completed(futures):
            results.append(fut.result())

    n_ok = sum(1 for r in results if r[1] == 0)
    n_fail = len(results) - n_ok
    logger.info("=" * 70)
    logger.info("Summary: %d/%d OK, %d FAIL", n_ok, len(results), n_fail)
    logger.info("=" * 70)

    if shared_run_id:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"windows_ok": n_ok, "windows_failed": n_fail})
                wandb.finish()
        except ImportError:
            pass

    logger.info("Next: aggregate via scripts/funding_arb_dsac_wf_report.py "
                "--config %s --gates %s --run_prefix %s",
                args.config, args.gates, args.run_name_prefix)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
