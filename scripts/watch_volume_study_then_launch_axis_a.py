#!/usr/bin/env python
"""Volume Study 5M-arm watcher -> Axis A auto-launch.

Polls WandB for the Volume Study 5M-budget runs tagged with the manifest's
study-id. When all expected seeds reach a terminal state, optionally runs
the volume-study analyzer (best-effort, log-only on failure), then
synchronously launches `scripts/run_data_window_study.py --full ...` to
start Axis A on the recommended slot.

Designed to be launched once and left to run for hours. Logs every poll
cycle to a file under C:/tmp/ so the user can tail it. Failsafe: aborts
after 12h with a clear log message.

Pre-conditions:
  * `data-window-study` configs + runner already validated (S508 evening).
  * Volume Study manifest exists with budget=5_000_000 included AND the
    dispatcher is still running (or already finished) so the 5M runs will
    eventually appear in WandB tagged with `study-id-<ts>`.

Usage:
    # Auto-discover newest manifest
    python scripts/watch_volume_study_then_launch_axis_a.py

    # Explicit manifest
    python scripts/watch_volume_study_then_launch_axis_a.py \
        --manifest results/volume_study_20260428_114420/manifest.json \
        --axis_a_instance gpuhub-1 --axis_a_gpu 0 \
        --axis_a_concurrent 5

    # Skip analyzer step (just launch Axis A on completion)
    python scripts/watch_volume_study_then_launch_axis_a.py --skip_analyze

    # Dry-run launch (writes derived configs but doesn't dispatch)
    python scripts/watch_volume_study_then_launch_axis_a.py --axis_a_dry_run
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path("C:/tmp")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(study_id: str) -> Path:
    log_path = LOG_DIR / f"watch_axis_a_{study_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] watch_axis_a - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def auto_discover_manifest() -> Path | None:
    """Pick the newest results/volume_study_*/manifest.json that includes 5M budget."""
    results = PROJECT_ROOT / "results"
    candidates = sorted(results.glob("volume_study_*/manifest.json"),
                        key=lambda p: p.parent.name, reverse=True)
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            if 5_000_000 in m.get("budgets", []):
                return path
        except Exception:
            continue
    return None


def poll_wandb_5m(api, entity: str, project: str, study_id: str) -> tuple[int, int, dict]:
    """Returns (n_runs_total, n_terminal, state_counts) for the 5M cell.

    Terminal = states {'finished', 'crashed', 'failed'}. Wandb API uses tag
    array filters identical to the dispatcher's wait_runs_finished.
    """
    runs = list(api.runs(f"{entity}/{project}", filters={
        "$and": [
            {"tags": {"$in": ["volume-study"]}},
            {"tags": {"$in": ["budget-5m"]}},
            {"tags": {"$in": [f"study-id-{study_id}"]}},
        ],
    }))
    states = [r.state for r in runs]
    counts = {s: states.count(s) for s in set(states)}
    terminal = sum(1 for s in states if s in ("finished", "crashed", "failed"))
    return len(runs), terminal, counts


def run_analyzer(manifest_path: Path) -> int:
    """Best-effort analyze_volume_study.py; log if it fails but don't abort."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_volume_study.py"),
           "--manifest", str(manifest_path)]
    logging.info("running analyzer: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=300)
        logging.info("analyzer stdout (last 30 lines):")
        for line in proc.stdout.splitlines()[-30:]:
            logging.info("  %s", line)
        if proc.returncode != 0:
            logging.warning("analyzer exited with code %d (continuing to Axis A launch)",
                            proc.returncode)
        return proc.returncode
    except Exception as e:
        logging.warning("analyzer raised %s — continuing to Axis A launch anyway", e)
        return -1


def launch_axis_a(instance: str, gpu: str, concurrent: int,
                  slots: str | None, dry_run: bool) -> int:
    """Synchronous launch of run_data_window_study.py --full. Blocks until
    the dispatcher returns (which itself will block per-cell through wandb
    polling). Logs go to the Bash subprocess output, captured by the
    background launch.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_data_window_study.py"),
           "--full"]
    if slots:
        cmd.extend(["--slots", slots])
    else:
        cmd.extend(["--instance", instance, "--gpu", str(gpu),
                    "--concurrent", str(concurrent)])
    if dry_run:
        cmd.append("--dry_run")

    logging.info("LAUNCHING AXIS A: %s", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if proc.returncode == 0:
        logging.info("Axis A dispatcher finished cleanly (%.0fs)", elapsed)
    else:
        logging.error("Axis A dispatcher exit=%d (%.0fs)", proc.returncode, elapsed)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=None,
                   help="Path to Volume Study manifest. If omitted, auto-discover "
                        "newest matching one in results/.")
    p.add_argument("--entity", default="bigcan-chiwin-technology")
    p.add_argument("--project", default="FinRL-Pro-DS")
    p.add_argument("--expected_seeds", type=int, default=5,
                   help="Number of seeds expected for the 5M cell (default 5).")
    p.add_argument("--poll_interval_s", type=int, default=300,
                   help="Seconds between WandB polls (default 300 = 5 min).")
    p.add_argument("--max_wait_h", type=float, default=12.0,
                   help="Maximum hours to wait before aborting (default 12).")
    p.add_argument("--skip_analyze", action="store_true",
                   help="Skip analyze_volume_study.py before Axis A launch.")
    p.add_argument("--axis_a_instance", default="gpuhub-1")
    p.add_argument("--axis_a_gpu", default="0")
    p.add_argument("--axis_a_concurrent", type=int, default=5)
    p.add_argument("--axis_a_slots", default=None,
                   help="Cross-GPU slot pool for Axis A (e.g. 'gpuhub-1:0,gpuhub-1:1').")
    p.add_argument("--axis_a_dry_run", action="store_true",
                   help="Forward --dry_run to the Axis A dispatcher.")
    args = p.parse_args()

    manifest_path = args.manifest or auto_discover_manifest()
    if manifest_path is None or not manifest_path.exists():
        print("ERROR: no Volume Study manifest with budget=5M found", file=sys.stderr)
        return 2
    manifest_path = manifest_path.resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    study_id = manifest["timestamp"]
    expected = args.expected_seeds

    log_path = setup_logging(study_id)
    logging.info("=" * 70)
    logging.info("Volume Study 5M -> Axis A watcher")
    logging.info("  manifest:        %s", manifest_path.relative_to(PROJECT_ROOT))
    logging.info("  study_id:        %s", study_id)
    logging.info("  budgets:         %s", manifest.get("budgets"))
    logging.info("  expected seeds:  %d", expected)
    logging.info("  poll interval:   %ds", args.poll_interval_s)
    logging.info("  failsafe:        %.1fh", args.max_wait_h)
    logging.info("  skip analyzer:   %s", args.skip_analyze)
    logging.info("  axis A target:   %s",
                 args.axis_a_slots or
                 f"{args.axis_a_instance} (gpu={args.axis_a_gpu}, concurrent={args.axis_a_concurrent})")
    logging.info("  axis A dry_run:  %s", args.axis_a_dry_run)
    logging.info("  watcher log:     %s", log_path)
    logging.info("=" * 70)

    import wandb
    api = wandb.Api(timeout=60)

    deadline = time.time() + args.max_wait_h * 3600
    last_state = None
    while time.time() < deadline:
        try:
            n_runs, terminal, counts = poll_wandb_5m(
                api, args.entity, args.project, study_id)
        except Exception as e:
            logging.warning("WandB poll raised %s — retrying in %ds", e, args.poll_interval_s)
            time.sleep(args.poll_interval_s)
            continue

        cur = (n_runs, terminal, tuple(sorted(counts.items())))
        if cur != last_state:
            logging.info("poll: 5M-arm runs=%d terminal=%d/%d by_state=%s",
                         n_runs, terminal, expected, counts)
            last_state = cur

        if n_runs >= expected and terminal >= expected:
            ok = counts.get("finished", 0)
            crashed = counts.get("crashed", 0) + counts.get("failed", 0)
            logging.info("=" * 70)
            logging.info("5M arm REACHED TERMINAL: %d finished, %d crashed/failed (of %d expected)",
                         ok, crashed, expected)
            logging.info("=" * 70)
            if not args.skip_analyze:
                run_analyzer(manifest_path)
            rc = launch_axis_a(args.axis_a_instance, args.axis_a_gpu,
                               args.axis_a_concurrent, args.axis_a_slots,
                               args.axis_a_dry_run)
            return rc

        time.sleep(args.poll_interval_s)

    logging.error("FAILSAFE: %.1fh elapsed without 5M-arm reaching terminal state. "
                  "Aborting WITHOUT launching Axis A. Investigate dispatcher state.",
                  args.max_wait_h)
    return 1


if __name__ == "__main__":
    sys.exit(main())
