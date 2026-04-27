#!/usr/bin/env python
"""Launch N seeds of an L1 multiseed validation with max K concurrent.

Each seed spawns `deploy_bare_metal.py --collect` as a subprocess (blocking
until the remote run finishes and artifacts are collected). Concurrency is
managed locally via ThreadPoolExecutor(max_workers=K).

**WandB consolidation (S488+):** unless `--separate_runs` is passed, the
launcher creates ONE parent WandB run and all seeds attach to it via
`FINRL_WANDB_RUN_ID` + `FINRL_WANDB_NAMESPACE=seed<N>`. Per-seed metrics land
under `seed<N>/*` with per-namespace `_step`.

Deploy target is determined by the fleet rule (S488): HPO -> gpuhub-1,
non-HPO (multiseed/WF) -> gpuhub-2.

Usage:
    python scripts/launch_l1_multiseed.py \
        --config configs/sg1_xauusd_ftmo_rehpo_l1_multiseed.yaml \
        --instance gpuhub-2 --gpu 0 \
        --seeds 42,123,456,789,1024,1337,2025,3141,5150,9999 \
        --concurrent 3 \
        --run_name_prefix sg1-xauusd-l1-multiseed-rehpo

Outputs:
    - ONE consolidated WandB run containing seed0..seedN namespaces
      (legacy behavior via --separate_runs: per-seed runs, no consolidation)
    - Per-seed checkpoint + metrics collected to local (deploy_bare_metal --collect)
    - launcher log at C:/tmp/l1_multiseed_<timestamp>.log
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import queue
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import wandb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "deploy_bare_metal.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] multiseed - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("l1_multiseed")


def parse_slots(slots_arg: str | None, fallback_instance: str | None,
                fallback_gpu: str | None) -> list[tuple[str, str]]:
    """Parse --slots CSV `host:gpu,host:gpu,...` or fall back to a 1-slot list.

    Each slot is a (instance_name, cuda_visible_devices) pair. The instance
    name must resolve via `instances.json` (deploy_bare_metal handles that);
    the gpu string is forwarded verbatim as CUDA_VISIBLE_DEVICES.

    Duplicate (host, gpu) tuples are allowed and create N concurrent replicas
    sharing one GPU's VRAM — same semantics as `--instance X --gpu Y --concurrent N`
    but extensible across multiple GPUs in one invocation. The caller is
    responsible for verifying VRAM headroom before running >1 replica per GPU.
    """
    if slots_arg:
        slots: list[tuple[str, str]] = []
        for token in slots_arg.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"slot {token!r} missing ':' (expected host:gpu)")
            host, gpu = token.split(":", 1)
            host, gpu = host.strip(), gpu.strip()
            if not host or not gpu:
                raise ValueError(f"slot {token!r} has empty host or gpu")
            slots.append((host, gpu))
        if not slots:
            raise ValueError("--slots parsed to empty list")
        return slots
    if fallback_instance is None:
        raise ValueError("must pass --slots or --instance")
    return [(fallback_instance, fallback_gpu or "0")]


def run_seed(config: Path, slot_pool: "queue.Queue[tuple[str, str]]",
             seed: int, run_name: str, no_collect: bool,
             shared_run_id: str | None,
             script: str | None = None) -> tuple[int, int, float, str, str]:
    """Spawn deploy_bare_metal for one seed and block until completion.

    Pulls a (instance, gpu) slot from `slot_pool` for the duration of the run
    and returns it on exit (success OR failure) so the next queued seed can
    pick it up. The pool is the synchronization primitive — its size caps
    concurrency, replacing the prior `max_workers` knob.

    If `shared_run_id` is given, child inherits FINRL_WANDB_RUN_ID +
    FINRL_WANDB_NAMESPACE=seed<N> so it attaches to the parent WandB run.

    Returns (seed, exit_code, elapsed_seconds, instance, gpu).
    """
    instance, gpu = slot_pool.get()
    try:
        start = time.time()
        # Force POSIX path separators — deploy_bare_metal passes --config through
        # to a remote Linux shell where backslashes are escape chars.
        cmd = [
            sys.executable, str(DEPLOY_SCRIPT),
            "--config", config.as_posix(),
            "--instance", instance,
            "--gpu", gpu,
            "--run_name", run_name,
            "--extra_args", f"--seed {seed}",
        ]
        if script:
            cmd.extend(["--script", script])
        if not no_collect:
            cmd.append("--collect")

        child_env = os.environ.copy()
        if shared_run_id:
            child_env["FINRL_WANDB_RUN_ID"] = shared_run_id
            child_env["FINRL_WANDB_NAMESPACE"] = f"seed{seed}"

        logger.info("seed %d: launching on %s:%s -> %s%s",
                    seed, instance, gpu, run_name,
                    " [consolidated]" if shared_run_id else "")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                              env=child_env)
        elapsed = time.time() - start
        if proc.returncode != 0:
            logger.error("seed %d FAILED on %s:%s (exit=%d, %.1fs)",
                         seed, instance, gpu, proc.returncode, elapsed)
            logger.error("  stdout tail: %s", proc.stdout[-500:])
            logger.error("  stderr tail: %s", proc.stderr[-500:])
        else:
            logger.info("seed %d OK on %s:%s (%.1fs)",
                        seed, instance, gpu, elapsed)
        return seed, proc.returncode, elapsed, instance, gpu
    finally:
        slot_pool.put((instance, gpu))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="Multiseed YAML config (per Protocol v2 Stage 2)")
    p.add_argument("--instance", default=None,
                   help="Single-slot fallback. Target instance in "
                        "instances.json (e.g. gpuhub-2). Ignored if --slots set.")
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES for single-slot fallback (default 0). "
                        "Ignored if --slots set.")
    p.add_argument("--slots", default=None,
                   help="Cross-GPU slot pool: comma-separated host:gpu pairs "
                        "(e.g. 'gpuhub-1:0,gpuhub-1:1,gpuhub-2:0'). When set, "
                        "seeds fan out across slots; --instance/--gpu/--concurrent "
                        "are ignored. Concurrency = len(slots).")
    p.add_argument("--seeds", required=True,
                   help="Comma-separated seed list (e.g. 42,123,456)")
    p.add_argument("--concurrent", type=int, default=3,
                   help="Single-slot mode only: max concurrent runs on the one "
                        "instance (default 3). Ignored when --slots set.")
    p.add_argument("--run_name_prefix", required=True,
                   help="WandB run name prefix; per-seed suffix added")
    p.add_argument("--no_collect", action="store_true",
                   help="Skip per-seed --collect (launcher returns faster; "
                        "manual collect-run needed later)")
    p.add_argument("--separate_runs", action="store_true",
                   help="Legacy: create one WandB run per seed. Default is "
                        "ONE consolidated parent run with seed<N>/* namespaces.")
    p.add_argument("--script", default=None,
                   help="Override the runner script (default: deploy_bare_metal "
                        "uses scripts/run_full_pipeline.py). Used for non-V7 "
                        "envs that need a workstream-specific runner — e.g. "
                        "scripts/funding_arb_dsac_train_seed.py for funding-arb.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 2
    if not DEPLOY_SCRIPT.exists():
        logger.error("deploy script not found: %s", DEPLOY_SCRIPT)
        return 2

    try:
        slots = parse_slots(args.slots, args.instance, args.gpu)
    except ValueError as e:
        logger.error("slot config error: %s", e)
        return 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 1:
        logger.error("no seeds parsed from %r", args.seeds)
        return 2
    if len(set(seeds)) != len(seeds):
        logger.error("duplicate seeds in %r", seeds)
        return 2

    # Concurrency = len(slots) when --slots is set; otherwise honor --concurrent
    # against the single fallback slot. We allow more workers than slots only
    # in the single-slot case (multiple seeds time-sharing one GPU's VRAM).
    if args.slots:
        # Slots already enforce uniqueness; pool size = len(slots).
        slot_pool: queue.Queue[tuple[str, str]] = queue.Queue()
        for slot in slots:
            slot_pool.put(slot)
        concurrency = len(slots)
    else:
        # Single-slot: replicate the slot N=concurrent times so up to N seeds
        # can share that one GPU (legacy behavior).
        concurrency = max(1, int(args.concurrent))
        slot_pool = queue.Queue()
        for _ in range(concurrency):
            slot_pool.put(slots[0])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 70)
    logger.info("L1 Multiseed Launcher")
    logger.info("  config:      %s", args.config.name)
    if args.slots:
        logger.info("  slots (%d):  %s", len(slots),
                    ",".join(f"{i}:{g}" for i, g in slots))
    else:
        logger.info("  instance:    %s (gpu=%s)", slots[0][0], slots[0][1])
        logger.info("  concurrent:  %d", concurrency)
    logger.info("  seeds (%d):  %s", len(seeds), seeds)
    logger.info("  prefix:      %s", args.run_name_prefix)
    logger.info("  collect:     %s", not args.no_collect)
    logger.info("=" * 70)

    if args.dry_run:
        for s in seeds:
            logger.info("dry_run: would launch seed %d", s)
        return 0

    # ------------------------------------------------------------------
    # Parent WandB run (consolidated mode)
    # ------------------------------------------------------------------
    shared_run_id: str | None = None
    parent_run = None
    if not args.separate_runs:
        try:
            with args.config.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("failed to load config for parent-run init: %s", e)
            return 2
        wcfg = cfg.get("wandb", {}) or {}
        parent_name = f"{args.run_name_prefix}_{timestamp}"
        shared_run_id = wandb.util.generate_id()
        parent_run = wandb.init(
            id=shared_run_id,
            project=wcfg.get("project", "FinRL-Pro-DS"),
            entity=wcfg.get("entity", "bigcan-chiwin-technology"),
            name=parent_name,
            job_type="l1_multiseed",
            tags=list(wcfg.get("tags", []) or []) + ["l1_multiseed", "consolidated"],
            config={
                "experiment": "l1_multiseed",
                "seeds": seeds,
                "n_seeds": len(seeds),
                "slots": [f"{i}:{g}" for i, g in slots],
                "concurrency": concurrency,
                "config_path": str(args.config),
                "launcher_timestamp": timestamp,
            },
        )
        logger.info("Parent WandB run: %s (id=%s)", parent_run.url, shared_run_id)
        # Define per-seed step metrics upfront so UI renders clean axes
        # as soon as children start logging.
        for s in seeds:
            wandb.define_metric(f"seed{s}/*", step_metric=f"seed{s}/_step")
            wandb.define_metric(f"seed{s}/_step", hidden=True)

    results: list[tuple[int, int, float, str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {}
        for s in seeds:
            run_name = f"{args.run_name_prefix}-seed{s}_{timestamp}"
            futs[ex.submit(
                run_seed, args.config, slot_pool, s,
                run_name, args.no_collect, shared_run_id, args.script,
            )] = s
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    fails = [(s, code) for s, code, _, _, _ in results if code != 0]
    ok = len(results) - len(fails)
    logger.info("=" * 70)
    logger.info("Summary: %d/%d OK, %d FAIL", ok, len(results), len(fails))
    if fails:
        for s, code in fails:
            logger.error("  seed %d exit=%d", s, code)

    if parent_run is not None:
        wandb.summary["seeds_ok"] = ok
        wandb.summary["seeds_failed"] = len(fails)
        wandb.summary["failed_seeds"] = [s for s, _ in fails]
        parent_run.finish()

    if fails:
        return 1
    logger.info("all seeds complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
