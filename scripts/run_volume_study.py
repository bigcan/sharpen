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
# Defaults preserve historical GMGP1 v1 dispatcher behavior. S537 generalized
# the dispatcher to drive arbitrary volume studies via --base_config /
# --gates_file / --prefix CLI flags (e.g. SG-1-XAUUSD volume_study_v2).
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "gmgp1_volume_study.yaml"
DEFAULT_GATES_FILE = PROJECT_ROOT / "configs" / "gmgp1_volume_study.gates.yaml"
DEFAULT_PREFIX = "gmgp1-volume"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "configs" / "_volume_study_runtime"
DEFAULT_RESULTS_SUBDIR = "volume_study"
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


def _compute_max_wait_h(total_timesteps: int,
                        sps: float = 62.0,
                        safety_factor: float = 1.3,
                        floor_h: float = 30.0) -> float:
    """Per-cell wandb-wait deadline scaled to budget.

    S539 lesson: hardcoded `max_wait_h=30.0` was too short for the 8M cell
    (~36h/seed at 62 SPS) and the dispatcher timed out before training finished.
    Default model: SAC v7+DSR+4090 = ~62 SPS regardless of GPU multiplexing
    (per `decision_volume_axis_v2_2m_ceiling.md` and S529/S539 empirical wall).
    Add a 30% safety factor for tail seeds + checkpoint+backtest overhead.
    Floor at 30h so small cells (500K, 1M) still get a generous deadline if
    an SFTP-deploy stragges.

    Examples (with sps=62, safety=1.3, floor=30):
        500K  -> floor 30h  (raw 2.9h)
        2M    -> floor 30h  (raw 11.6h)
        4M    -> floor 30h  (raw 23.3h)
        8M    -> 46.6h
        16M   -> 93.2h
    """
    raw_h = total_timesteps / sps * safety_factor / 3600.0
    return max(floor_h, raw_h)


def write_derived_config(base: dict, total_timesteps: int, label: str,
                         study_id: str, runtime_dir: Path,
                         filename_stem: str) -> Path:
    """Materialize a derived config with overridden total_timesteps."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("training", {})["total_timesteps"] = total_timesteps
    # Tag each run with budget + unique study id for downstream filtering.
    tags = list(cfg.setdefault("wandb", {}).setdefault("tags", []))
    for t in (f"budget-{label}", f"study-id-{study_id}"):
        if t not in tags:
            tags.append(t)
    cfg["wandb"]["tags"] = tags

    runtime_dir.mkdir(parents=True, exist_ok=True)
    out = runtime_dir / f"{filename_stem}_{label}.yaml"
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
                  timestamp: str, dry_run: bool,
                  prefix_base: str = DEFAULT_PREFIX) -> tuple[int, float]:
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
    prefix = f"{prefix_base}-{label}"
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
        # Use separate WandB runs per seed (one parent-run per seed) instead of
        # consolidated parent. The consolidated mode has a multi-writer summary
        # race where each python's wandb session, on finish(), flushes its
        # local summary cache (which contains stale reads of OTHER seeds'
        # values) and stomps the run-wide summary. Separate runs eliminate
        # the race entirely — each seed run has a single writer. Bonus:
        # `--collect` polling now works (each subprocess polls its own run
        # id, which transitions cleanly without the parent-writer deadlock).
        "--separate_runs",
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
    p.add_argument("--base_config", type=Path, default=DEFAULT_BASE_CONFIG,
                   help="Base config to derive per-budget cells from. "
                        "Default: configs/gmgp1_volume_study.yaml")
    p.add_argument("--gates_file", type=Path, default=DEFAULT_GATES_FILE,
                   help="Pre-registered hypothesis gates yaml (drives step_budgets, seeds). "
                        "Default: configs/gmgp1_volume_study.gates.yaml")
    p.add_argument("--prefix", type=str, default=None,
                   help="Run-name prefix used for derived configs + WandB run names. "
                        "If unset, derived from --base_config stem (lowercase, "
                        "underscores → hyphens). GMGP1 v1 uses 'gmgp1-volume'.")
    p.add_argument("--results_subdir", type=str, default=DEFAULT_RESULTS_SUBDIR,
                   help="Sub-directory under results/ for the manifest. "
                        "Default: 'volume_study' → results/volume_study_<ts>/")
    p.add_argument("--study_id", type=str, default=None,
                   help="Override the study_id used in the WandB 'study-id-<ID>' "
                        "tag. Default: auto-generated timestamp. Use this to "
                        "RESUME a study where some cells already finished — pass "
                        "the original study_id so 4M/8M cells share the tag with "
                        "the existing 2M cell for clean analysis-time merging.")
    p.add_argument("--dry_run", action="store_true",
                   help="Generate + validate configs, print launch commands, do not spawn")
    args = p.parse_args()

    base_config: Path = args.base_config
    gates_file: Path = args.gates_file
    if not base_config.exists():
        log.error("base config not found: %s", base_config)
        return 2
    if not gates_file.exists():
        log.error("gates file not found: %s", gates_file)
        return 2
    if not LAUNCHER.exists():
        log.error("launcher not found: %s", LAUNCHER)
        return 2

    # Derive prefix + filename stem from base_config if not supplied. Backward
    # compat: when both base_config and gates_file are the GMGP1 v1 defaults
    # AND --prefix was not passed, restore the historical 'gmgp1-volume' prefix
    # so existing WandB run-name patterns / randd_log greps keep matching.
    base_stem = base_config.stem  # e.g. 'sg1_xauusd_volume_study_v2'
    if args.prefix is not None:
        prefix_base = args.prefix
    elif base_config == DEFAULT_BASE_CONFIG:
        prefix_base = DEFAULT_PREFIX  # 'gmgp1-volume' (back-compat)
    else:
        prefix_base = base_stem.replace("_", "-")
    # Back-compat: GMGP1 v1 historical runtime dir was '_volume_study_runtime'
    # (no workstream prefix). Preserve to keep existing on-disk artifacts.
    if base_config == DEFAULT_BASE_CONFIG:
        runtime_dir = DEFAULT_RUNTIME_DIR
    else:
        runtime_dir = PROJECT_ROOT / "configs" / f"_{base_stem}_runtime"

    def _load_gates() -> dict:
        with gates_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    gates = _load_gates()
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
    # study_id is the WandB-side study tag (study-id-<ID>); defaults to the
    # current timestamp but can be overridden to resume an interrupted study
    # so newly-launched cells share the tag with already-finished cells.
    study_id = args.study_id or timestamp
    results_dir = PROJECT_ROOT / "results" / f"{args.results_subdir}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Volume Study Dispatcher (workstream=%s)",
             gates.get("workstream", "<unset>"))
    log.info("  base config: %s", base_config.name)
    log.info("  gates file:  %s", gates_file.name)
    log.info("  prefix:      %s", prefix_base)
    log.info("  runtime dir: %s", runtime_dir.relative_to(PROJECT_ROOT))
    log.info("  study_id:    %s%s", study_id,
             " (override)" if args.study_id else " (auto)")
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

    with base_config.open("r", encoding="utf-8") as f:
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
        cfg_path = write_derived_config(base_cfg, steps, label, study_id,
                                        runtime_dir=runtime_dir,
                                        filename_stem=base_stem)
        derived_paths[steps] = cfg_path
        if not validate(cfg_path):
            log.error("aborting: validator rejected %s", cfg_path.name)
            return 3

    # Phase 2: dispatch budgets sequentially.
    # The launcher fires off all 5 deploys with --no_collect and returns within
    # ~25 min (5 × stagger). Training itself runs for hours afterwards. To
    # know when a budget is actually done — so we don't pile concurrent
    # budgets onto gpuhub-1 — we poll wandb directly for that budget's runs.
    import wandb as _wandb
    project = base_cfg.get("wandb", {}).get("project", "FinRL-Pro-DS")
    entity = base_cfg.get("wandb", {}).get("entity", "bigcan-chiwin-technology")

    def wait_runs_finished(label_: str, expected_n: int,
                           poll_every_s: int = 120, max_wait_h: float = 30.0) -> int:
        """Block until `expected_n` runs tagged for this budget are all in a
        terminal state. Returns count of seeds that reached `finished`.

        S537-cont fix: instantiate a FRESH wandb.Api per poll iteration. The
        prior implementation created a single Api outside the loop and got
        stuck in May 2026 returning stale 'all running' state for hours after
        runs had actually finished server-side. The Api.runs() public collection
        caches GraphQL pages on the underlying client; recreating the Api each
        poll forces a clean session and avoids the cache.
        """
        deadline = time.time() + max_wait_h * 3600
        last_log_state = None
        consecutive_errors = 0
        while time.time() < deadline:
            try:
                api = _wandb.Api(timeout=60)
                runs = list(api.runs(f"{entity}/{project}", filters={
                    "$and": [
                        {"tags": {"$in": ["volume-study"]}},
                        {"tags": {"$in": [f"budget-{label_}"]}},
                        {"tags": {"$in": [f"study-id-{study_id}"]}},
                    ],
                }))
                consecutive_errors = 0
            except Exception as exc:  # transient network / 5xx / auth
                consecutive_errors += 1
                log.warning("wandb poll error #%d on budget=%s: %s",
                            consecutive_errors, label_, exc)
                if consecutive_errors >= 10:
                    log.error("aborting wait: %d consecutive wandb api errors on budget=%s",
                              consecutive_errors, label_)
                    return -1
                time.sleep(poll_every_s)
                continue
            states = [r.state for r in runs]
            counts = {s: states.count(s) for s in set(states)}
            terminal = sum(1 for s in states if s in ("finished", "crashed", "failed"))
            cur_state = (len(runs), terminal, tuple(sorted(counts.items())))
            if cur_state != last_log_state:
                log.info("waiting on budget=%s: runs=%d  terminal=%d/%d  by_state=%s",
                         label_, len(runs), terminal, expected_n, counts)
                last_log_state = cur_state
            if len(runs) >= expected_n and terminal >= expected_n:
                ok = states.count("finished")
                log.info("budget=%s wandb-side complete: %d/%d finished, others: %s",
                         label_, ok, expected_n,
                         {k: v for k, v in counts.items() if k != "finished"})
                return ok
            time.sleep(poll_every_s)
        log.error("wandb wait timed out after %.1fh on budget=%s", max_wait_h, label_)
        return -1

    for steps in budgets:
        label = _budget_label(steps)
        cfg_path = derived_paths[steps]
        rc, elapsed = launch_budget(
            cfg_path, label, seeds,
            args.instance, args.gpu, args.concurrent,
            args.slots,
            timestamp, args.dry_run,
            prefix_base=prefix_base,
        )

        # Wait for the actual training+backtest to finish before next budget.
        # The launcher only spawns deploys; runs continue on remote afterwards.
        finished_ok = -2  # not waited
        if rc == 0 and not args.dry_run:
            cell_max_wait_h = _compute_max_wait_h(steps)
            log.info("budget=%s wait deadline: %.1fh (steps=%d @ ~62 SPS × 1.3 safety, floor 30h)",
                     label, cell_max_wait_h, steps)
            finished_ok = wait_runs_finished(label, expected_n=len(seeds),
                                             max_wait_h=cell_max_wait_h)

        manifest["runs"].append({
            "budget": steps,
            "label": label,
            "seeds": seeds,
            "config": str(cfg_path.relative_to(PROJECT_ROOT)),
            "exit_code": rc,
            "elapsed_s": round(elapsed, 1),
            "wandb_finished_ok": finished_ok,
        })
        with (results_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if rc != 0:
            log.error("aborting subsequent budgets: budget=%s launcher exit=%d", label, rc)
            return 1
        if finished_ok < 0:
            log.error("aborting subsequent budgets: budget=%s wandb wait failed (%d)", label, finished_ok)
            return 1
        if finished_ok < len(seeds):
            log.warning("budget=%s: only %d/%d seeds finished cleanly; continuing",
                        label, finished_ok, len(seeds))

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
