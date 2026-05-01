#!/usr/bin/env python
"""GMGP1 Training-Data-Window Study (Axis A) — dispatcher.

Sweeps `data.train_start_date` x `training.total_timesteps` on GMGP1 SAC
BTC 15-min:
  4 windows  {A_short, A_base, A_medium, A_long}
  2 budgets  {500K, 2M}
  5 seeds    [123, 456, 789, 1024, 2026]
  = 40 runs total.

Twin to scripts/run_volume_study.py (which sweeps budget at fixed window).
HPs frozen from v5 HPO `6xluh826` T29 (same agent, swap data axis).

Per Protocol v2 Stage 2: each (window, budget) cell launches via
launch_l1_multiseed.py with --separate_runs (per-seed wandb runs). Cells
are sequential to avoid GPU contention. Within a cell, seeds are
fanned across slots (or concurrent on a single slot).

Pre-registered hypothesis gates (G1-G6) in
configs/gmgp1_data_window_study.gates.yaml — read by
scripts/analyze_data_window_study.py post-run.

Usage:
    # Smoke test (1 seed at A_base / 500K on gpuhub-1:0)
    python scripts/run_data_window_study.py --smoke --instance gpuhub-1 --gpu 0

    # Full 40-run sweep, single-instance (sequential cells, concurrent seeds)
    python scripts/run_data_window_study.py --full --instance gpuhub-1 --gpu 0 \
        --concurrent 5

    # Full 40-run sweep, cross-GPU fan-out (gpuhub-1 dual-GPU after Volume
    # Study clears; seeds within each cell distributed across slots)
    python scripts/run_data_window_study.py --full \
        --slots gpuhub-1:0,gpuhub-1:1

    # Subset (e.g., only A_long at 2M for quick re-check)
    python scripts/run_data_window_study.py --full \
        --windows A_long --budgets 2000000

    # Dry run: prints commands + writes derived configs but does not launch
    python scripts/run_data_window_study.py --full --dry_run

Manifest: writes results/data_window_study_<timestamp>/manifest.json with
(window, budget, seed, run_name, status, elapsed_s) entries for the
analysis script.
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
BASE_CONFIG = PROJECT_ROOT / "configs" / "gmgp1_data_window_study.yaml"
GATES_FILE = PROJECT_ROOT / "configs" / "gmgp1_data_window_study.gates.yaml"
RUNTIME_DIR = PROJECT_ROOT / "configs" / "_data_window_study_runtime"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_l1_multiseed.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_config.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] data_window_study - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("data_window_study")


def _budget_label(steps: int) -> str:
    """Human-readable budget tag, e.g. 500000 -> '500k', 2_000_000 -> '2m'."""
    if steps >= 1_000_000:
        return f"{steps // 1_000_000}m" if steps % 1_000_000 == 0 else f"{steps / 1_000_000:.1f}m"
    return f"{steps // 1000}k"


def _window_slug(window_name: str) -> str:
    """Map gates-file window names to short, regex-safe slugs for run prefixes.

    naming.py validate_run_name is `^[a-z0-9-]+_\\d{8}_\\d{6}$` — keep slugs
    lowercase + hyphens only.
    """
    return {
        "A_short": "short",
        "A_base": "base",
        "A_medium": "med",
        "A_long": "long",
    }.get(window_name, window_name.lower().replace("_", "-"))


def load_gates() -> dict:
    with GATES_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_derived_config(base: dict, window_name: str, train_start_date: str,
                         total_timesteps: int, study_id: str) -> Path:
    """Materialize a derived config with overridden train_start_date AND
    total_timesteps. One file per (window, budget) cell.
    """
    cfg = copy.deepcopy(base)
    cfg.setdefault("data", {})["train_start_date"] = train_start_date
    cfg.setdefault("training", {})["total_timesteps"] = total_timesteps

    win_slug = _window_slug(window_name)
    bud_label = _budget_label(total_timesteps)

    # Tag each run with window + budget + unique study id for downstream
    # filtering. Mirrors Volume Study tagging convention.
    tags = list(cfg.setdefault("wandb", {}).setdefault("tags", []))
    for t in (f"window-{win_slug}", f"budget-{bud_label}",
              f"study-id-{study_id}"):
        if t not in tags:
            tags.append(t)
    cfg["wandb"]["tags"] = tags

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNTIME_DIR / f"gmgp1_data_window_{win_slug}_{bud_label}.yaml"
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


def launch_cell(cfg_path: Path, window_slug: str, budget_label: str,
                seeds: list[int],
                instance: str | None, gpu: str | None, concurrent: int,
                slots: str | None,
                timestamp: str, dry_run: bool) -> tuple[int, float]:
    """Invoke launch_l1_multiseed.py for one (window, budget) cell.
    Blocks until all seeds done.

    See run_volume_study.launch_budget for the rationale on --separate_runs
    + relative config paths + naming-regex constraints.
    """
    seeds_csv = ",".join(str(s) for s in seeds)
    prefix = f"gmgp1-dwin-{window_slug}-{budget_label}"
    cfg_rel = cfg_path.relative_to(PROJECT_ROOT).as_posix()
    cmd = [
        sys.executable, str(LAUNCHER),
        "--config", cfg_rel,
        "--seeds", seeds_csv,
        "--run_name_prefix", prefix,
        "--separate_runs",
    ]
    if slots:
        cmd.extend(["--slots", slots])
        target_str = f"slots={slots}"
    else:
        cmd.extend([
            "--instance", instance or "gpuhub-1",
            "--gpu", gpu or "0",
            "--concurrent", str(concurrent),
        ])
        target_str = f"{instance} (gpu={gpu}, concurrent={concurrent})"
    if dry_run:
        cmd.append("--dry_run")

    log.info("launching cell window=%s budget=%s seeds=%s -> %s",
             window_slug, budget_label, seeds_csv, target_str)
    log.info("  cmd: %s", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("cell window=%s budget=%s FAILED (exit=%d, %.0fs)",
                  window_slug, budget_label, proc.returncode, elapsed)
    else:
        log.info("cell window=%s budget=%s OK (%.0fs)",
                 window_slug, budget_label, elapsed)
    return proc.returncode, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help="Smoke test: 1 seed (123) at A_base / 500K only")
    mode.add_argument("--full", action="store_true",
                      help="Full 40-run sweep (4 windows x 2 budgets x 5 seeds)")
    p.add_argument("--instance", default="gpuhub-1",
                   help="Single-slot fallback target instance "
                        "(default: gpuhub-1, free post-Volume-Study). "
                        "Ignored if --slots is set.")
    p.add_argument("--gpu", default="0",
                   help="Single-slot CUDA_VISIBLE_DEVICES (default 0). "
                        "Ignored if --slots is set.")
    p.add_argument("--concurrent", type=int, default=5,
                   help="Single-slot within-cell seed concurrency (default 5 — "
                        "matches N=5 seeds, all parallel; reduce on VRAM-bound "
                        "instances). Ignored if --slots is set.")
    p.add_argument("--slots", default=None,
                   help="Cross-GPU slot pool: comma-separated host:gpu pairs "
                        "(e.g. 'gpuhub-1:0,gpuhub-1:1'). When set, "
                        "each cell fans seeds out across these slots; "
                        "concurrency = len(slots). Cells remain sequential.")
    p.add_argument("--windows", type=str, default=None,
                   help="Comma-separated subset of window names "
                        "(e.g., 'A_long' or 'A_base,A_long'). "
                        "Default: all 4 from gates file.")
    p.add_argument("--budgets", type=str, default=None,
                   help="Comma-separated subset of budgets "
                        "(e.g., '500000' or '500000,2000000'). "
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
    all_windows: dict[str, dict] = dict(gates["windows"])
    all_budgets: list[int] = list(gates["step_budgets"])
    all_seeds: list[int] = list(gates["seeds"])

    if args.windows:
        wanted = [w.strip() for w in args.windows.split(",")]
        windows = {k: all_windows[k] for k in wanted if k in all_windows}
        if len(windows) != len(wanted):
            missing = [w for w in wanted if w not in all_windows]
            log.error("unknown window(s): %s (known: %s)", missing, list(all_windows))
            return 2
    elif args.smoke:
        windows = {"A_base": all_windows["A_base"]}
    else:
        windows = all_windows

    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",")]
    elif args.smoke:
        budgets = [500_000]
    else:
        budgets = all_budgets

    seeds = [all_seeds[0]] if args.smoke else all_seeds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"data_window_study_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("GMGP1 Data-Window Study (Axis A) Dispatcher")
    log.info("  base config: %s", BASE_CONFIG.name)
    log.info("  windows:     %s", list(windows.keys()))
    log.info("  budgets:     %s", budgets)
    log.info("  seeds:       %s", seeds)
    log.info("  total cells: %d  total runs: %d",
             len(windows) * len(budgets), len(windows) * len(budgets) * len(seeds))
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
        "windows": {k: v for k, v in windows.items()},
        "budgets": budgets,
        "seeds": seeds,
        "dry_run": args.dry_run,
        "runs": [],
    }

    # Phase 1: generate + validate all derived configs upfront. Cheap fail-fast.
    derived_paths: dict[tuple[str, int], Path] = {}
    for window_name, window_meta in windows.items():
        train_start_date = window_meta["train_start_date"]
        for steps in budgets:
            cfg_path = write_derived_config(
                base_cfg, window_name, train_start_date, steps, timestamp,
            )
            derived_paths[(window_name, steps)] = cfg_path
            if not validate(cfg_path):
                log.error("aborting: validator rejected %s", cfg_path.name)
                return 3

    # Phase 2: dispatch cells sequentially. Volume Study precedent: poll wandb
    # for terminal state before advancing — avoids stacking concurrent cells
    # onto the same GPU once deploy_bare_metal returns from setup phase.
    # Each poll runs in a fresh subprocess to dodge the long-lived api.runs()
    # hang that bit S508/S509 (volume study, watcher). See randd_log S509.
    project = base_cfg.get("wandb", {}).get("project", "FinRL-Pro-DS")
    entity = base_cfg.get("wandb", {}).get("entity", "bigcan-chiwin-technology")

    _POLL_PY = (
        "import json, sys, wandb\n"
        "api = wandb.Api(timeout=30)\n"
        "entity, project, win, bud, sid = sys.argv[1:6]\n"
        "runs = list(api.runs(f'{entity}/{project}', filters={'$and': [\n"
        "    {'tags': {'$in': ['data-window-study']}},\n"
        "    {'tags': {'$in': [f'window-{win}']}},\n"
        "    {'tags': {'$in': [f'budget-{bud}']}},\n"
        "    {'tags': {'$in': [f'study-id-{sid}']}},\n"
        "]}))\n"
        "states = [r.state for r in runs]\n"
        "counts = {s: states.count(s) for s in set(states)}\n"
        "terminal = sum(1 for s in states if s in ('finished','crashed','failed'))\n"
        "print(json.dumps({'n': len(runs), 'terminal': terminal,\n"
        "                  'counts': counts, 'finished': states.count('finished')}))\n"
    )

    def wait_runs_finished(window_slug_: str, budget_label_: str,
                           expected_n: int,
                           poll_every_s: int = 120,
                           max_wait_h: float = 8.0) -> int:
        """Block until `expected_n` runs tagged for this cell are all in a
        terminal state. Returns count of seeds that reached `finished`.

        Per artifact compute estimate: 2M-step run ~25 min, 500K-step run
        ~6 min. 8h ceiling = 16-20x slack on the worst cell.

        Fresh subprocess per poll — `wandb.Api()` reuse hangs after hours.
        """
        deadline = time.time() + max_wait_h * 3600
        last_log_state = None
        while time.time() < deadline:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _POLL_PY,
                     entity, project, window_slug_, budget_label_, timestamp],
                    capture_output=True, text=True, timeout=180,
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    log.warning("poll subprocess failed (rc=%d): %s",
                                proc.returncode, proc.stderr[-500:])
                    time.sleep(poll_every_s)
                    continue
                data = json.loads(proc.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                log.warning("poll subprocess hit 180s timeout — retrying")
                time.sleep(poll_every_s)
                continue
            except Exception as e:
                log.warning("poll subprocess raised %s — retrying", e)
                time.sleep(poll_every_s)
                continue

            n_runs = data["n"]
            terminal = data["terminal"]
            counts = data["counts"]
            cur_state = (n_runs, terminal, tuple(sorted(counts.items())))
            if cur_state != last_log_state:
                log.info("waiting on cell window=%s budget=%s: "
                         "runs=%d terminal=%d/%d by_state=%s",
                         window_slug_, budget_label_, n_runs,
                         terminal, expected_n, counts)
                last_log_state = cur_state
            if n_runs >= expected_n and terminal >= expected_n:
                ok = data["finished"]
                log.info("cell window=%s budget=%s wandb-side complete: "
                         "%d/%d finished, others: %s",
                         window_slug_, budget_label_, ok, expected_n,
                         {k: v for k, v in counts.items() if k != "finished"})
                return ok
            time.sleep(poll_every_s)
        log.error("wandb wait timed out after %.1fh on cell window=%s budget=%s",
                  max_wait_h, window_slug_, budget_label_)
        return -1

    # Outer loop: budgets (fast first — 500K cells finish in ~6 min each so
    # the 2M cells overlap with their analysis). Inner loop: windows.
    # This ordering means any partial-completion still produces a complete
    # 500K row of the PF surface, which alone answers G1 at the cheaper
    # budget level.
    for steps in budgets:
        budget_label = _budget_label(steps)
        for window_name, window_meta in windows.items():
            window_slug = _window_slug(window_name)
            cfg_path = derived_paths[(window_name, steps)]
            rc, elapsed = launch_cell(
                cfg_path, window_slug, budget_label, seeds,
                args.instance, args.gpu, args.concurrent,
                args.slots,
                timestamp, args.dry_run,
            )

            finished_ok = -2  # not waited
            if rc == 0 and not args.dry_run:
                finished_ok = wait_runs_finished(
                    window_slug, budget_label, expected_n=len(seeds),
                )

            manifest["runs"].append({
                "window": window_name,
                "window_slug": window_slug,
                "train_start_date": window_meta["train_start_date"],
                "train_window_months": window_meta.get("train_window_months"),
                "budget": steps,
                "budget_label": budget_label,
                "seeds": seeds,
                "config": str(cfg_path.relative_to(PROJECT_ROOT)),
                "exit_code": rc,
                "elapsed_s": round(elapsed, 1),
                "wandb_finished_ok": finished_ok,
            })
            with (results_dir / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            if rc != 0:
                log.error("aborting subsequent cells: window=%s budget=%s "
                          "launcher exit=%d", window_name, budget_label, rc)
                return 1
            # finished_ok == -2 means "not waited" (dry-run path) — not a failure.
            # Only treat actual wandb wait timeouts (-1) as fatal.
            if finished_ok == -1:
                log.error("aborting subsequent cells: window=%s budget=%s "
                          "wandb wait timed out", window_name, budget_label)
                return 1
            if finished_ok < len(seeds):
                log.warning("cell window=%s budget=%s: only %d/%d seeds "
                            "finished cleanly; continuing",
                            window_name, budget_label, finished_ok, len(seeds))

    fails = [r for r in manifest["runs"] if r["exit_code"] != 0]
    log.info("=" * 70)
    log.info("Dispatcher complete: %d/%d cells OK",
             len(manifest["runs"]) - len(fails), len(manifest["runs"]))
    log.info("Manifest: %s", (results_dir / "manifest.json").relative_to(PROJECT_ROOT))
    log.info("Next: scripts/analyze_data_window_study.py --manifest %s",
             results_dir / "manifest.json")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
