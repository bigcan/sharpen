#!/usr/bin/env python
"""SG-1-BTC Velotrade Stage 3 WF dispatcher — DECAY-01 variant (S542, top-3 N=3).

Mirrors `scripts/launch_sg1_btc_extended_wf_multiseed.py`. For each fold (8
emitted by splitter on 2024-01-01 → 2026-04-30 envelope at 18mo train / 1mo
val / 1mo test / 1mo step):
  1. Materialize a derived L1-style config with explicit train/val/test dates.
  2. Validate via `scripts/validate_config.py --stage l1-multiseed`.
  3. Launch via `scripts/launch_l1_multiseed.py` with --seeds 456,42,2025
     and the user-supplied --slots pool (or single-instance fallback).
Cells run sequentially; within a cell, seeds fan out across slots.

After this dispatcher completes (all 24 train runs done), run:
  python scripts/sg1_btc_velotrade_ensemble_eval.py \
      --wf_config configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml \
      --gates_file configs/sg1_btc_velotrade_ensemble.gates.yaml
to produce per-fold ensemble verdicts via `run_wf_ensemble()`.

Usage:
    # Dry-run: materialize + validate, no launch
    python scripts/launch_sg1_btc_decay01_wf_multiseed.py \
        --config configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml --dry_run

    # Single-instance launch (gpuhub-2:0, concurrent=3 → ~24h wall)
    python scripts/launch_sg1_btc_decay01_wf_multiseed.py \
        --config configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml \
        --instance gpuhub-2 --gpu 0 --concurrent 3

    # Multi-slot pool (faster wall time, more concurrent VRAM pressure)
    python scripts/launch_sg1_btc_decay01_wf_multiseed.py \
        --config configs/sg1_btc_velotrade_decay01_wf_multiseed.yaml \
        --slots gpuhub-2:0,gpuhub-2:0,gpuhub-2:0

Manifest: writes `results/sg1_btc_decay01_wf_<timestamp>/manifest.json` with
per-fold (study_id, seeds, exit_code, wandb_finished_ok) entries for the
post-run analyzer. WandB tags per cell:
  `sg-1`, `BTC`, `wf-stage3`, `decay01`, `s542`, `fold-<i>`, `study-id-<ts>`.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_l1_multiseed.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_config.py"

sys.path.insert(0, str(PROJECT_ROOT))
from sharpen.data.splitter import RollingWindowSplitter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] sg1_btc_decay01_wf - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sg1_btc_decay01_wf")


def _date_only(ts: str) -> str:
    """Trim 'YYYY-MM-DD HH:MM:SS' → 'YYYY-MM-DD' for config dates."""
    return ts[:10]


def parse_base_prefix(checkpoint_pattern: str) -> str:
    """Derive the run-name prefix from ensemble.checkpoint_pattern.

    'checkpoints/sg1-btc-x1-wf-fold{fold}-seed{seed}_*/checkpoint_final.pth'
        -> 'sg1-btc-x1-wf'  (the text between 'checkpoints/' and '-fold{')
    Config-driven so decay01 and x1 (and future workstreams) share one launcher
    instead of hardcoding seeds/prefix. Returns the prefix or raises ValueError.
    """
    m = re.search(r"checkpoints/(.+?)-fold\{fold", checkpoint_pattern)
    if not m:
        raise ValueError(
            f"cannot parse run-name prefix from checkpoint_pattern={checkpoint_pattern!r}; "
            "expected 'checkpoints/<prefix>-fold{fold}-seed{seed}_*/...'")
    return m.group(1)


def write_fold_config(base: dict, fold_idx: int, fold: dict, study_id: str,
                      runtime_dir: Path, fold_basename: str) -> Path:
    """Materialize a single-fold L1-style config with explicit windows."""
    cfg = copy.deepcopy(base)
    cfg.pop("splitter", None)
    cfg["data"]["train_start_date"] = _date_only(str(fold["train"].start))
    cfg["data"]["train_end_date"] = _date_only(str(fold["train"].end))
    cfg["data"]["val_start_date"] = _date_only(str(fold["val"].start))
    cfg["data"]["val_end_date"] = _date_only(str(fold["val"].end))
    cfg["data"]["test_start_date"] = _date_only(str(fold["test"].start))
    cfg["data"]["test_end_date"] = _date_only(str(fold["test"].end))
    cfg["data"].pop("start_date", None)
    cfg["data"].pop("end_date", None)

    tags = list(cfg.setdefault("wandb", {}).setdefault("tags", []))
    for t in (f"fold-{fold_idx}", f"study-id-{study_id}"):
        if t not in tags:
            tags.append(t)
    cfg["wandb"]["tags"] = tags

    runtime_dir.mkdir(parents=True, exist_ok=True)
    out = runtime_dir / f"{fold_basename}_fold_{fold_idx}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def validate(cfg_path: Path) -> bool:
    cmd = [sys.executable, str(VALIDATOR),
           "--config", str(cfg_path), "--stage", "l1-multiseed"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if "status=FAIL" in (proc.stdout + proc.stderr):
        log.error("validator FAIL on %s\n%s", cfg_path.name, proc.stdout)
        return False
    log.info("validator %s on %s",
             "WARN" if "status=WARN" in proc.stdout else "PASS", cfg_path.name)
    return True


def launch_cell(cfg_path: Path, fold_idx: int, slots: str | None,
                instance: str | None, gpu: str | None, concurrent: int,
                dry_run: bool, seeds: list[int], base_prefix: str) -> tuple[int, float]:
    """Invoke launch_l1_multiseed.py for one fold's N-seed cell."""
    seeds_csv = ",".join(str(s) for s in seeds)
    prefix = f"{base_prefix}-fold{fold_idx}"
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
            "--instance", instance or "gpuhub-2",
            "--gpu", gpu or "0",
            "--concurrent", str(concurrent),
        ])
        target_str = f"{instance} (gpu={gpu}, concurrent={concurrent})"
    if dry_run:
        cmd.append("--dry_run")

    log.info("launching fold %d cell seeds=%s -> %s", fold_idx, seeds_csv, target_str)
    log.info("  cmd: %s", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("fold %d cell FAILED (exit=%d, %.0fs)", fold_idx, proc.returncode, elapsed)
    else:
        log.info("fold %d cell launcher OK (%.0fs)", fold_idx, elapsed)
    return proc.returncode, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path,
                   help="WF config (must have data.start_date, data.end_date, splitter:)")
    p.add_argument("--instance", default="gpuhub-2",
                   help="Single-slot fallback instance (ignored if --slots set)")
    p.add_argument("--gpu", default="0",
                   help="Single-slot CUDA_VISIBLE_DEVICES (ignored if --slots set)")
    p.add_argument("--concurrent", type=int, default=3,
                   help="Single-slot within-cell seed concurrency (ignored if --slots set)")
    p.add_argument("--slots", default=None,
                   help="Cross-GPU slot pool: comma-separated host:gpu pairs.")
    p.add_argument("--folds", type=str, default=None,
                   help="Comma-separated fold subset (e.g., '0' for smoke, '0,1' for two)")
    p.add_argument("--dry_run", action="store_true",
                   help="Materialize + validate configs + print launch cmds; do not spawn")
    args = p.parse_args()

    if not args.config.exists():
        log.error("config not found: %s", args.config); return 2
    args.config = args.config.resolve()
    if not LAUNCHER.exists():
        log.error("launcher not found: %s", LAUNCHER); return 2

    with args.config.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Config-driven seeds + run-name prefix + runtime/results paths (no hardcoding).
    # Empty ensemble.seeds => fail loud (placeholder guard) rather than run leaked seeds.
    ens = base_cfg.get("ensemble", {})
    seeds = list(ens.get("seeds") or [])
    if not seeds:
        log.error("ensemble.seeds is empty in %s — fill from the upstream L1 "
                  "seed_report.json (top3_by_val_argmax_pf) before launching; "
                  "refusing to run placeholder/leaked seeds", args.config.name)
        return 2
    try:
        base_prefix = parse_base_prefix(ens.get("checkpoint_pattern", ""))
    except ValueError as e:
        log.error("%s", e)
        return 2
    pfx_us = base_prefix.replace("-", "_")
    runtime_dir = PROJECT_ROOT / "configs" / f"_{pfx_us}_runtime"

    data = base_cfg.get("data", {})
    spl = base_cfg.get("splitter", {})
    if not spl or "start_date" not in data or "end_date" not in data:
        log.error("config missing splitter: or data.{start_date,end_date}"); return 2

    splitter = RollingWindowSplitter(
        train_months=spl.get("train_months", 18),
        val_months=spl.get("val_months", 1),
        test_months=spl.get("test_months", 1),
        step_months=spl.get("step_months", 1),
        buffer_days=spl.get("buffer_days", 0),
    )
    folds = splitter.split(data["start_date"], data["end_date"])
    log.info("Splitter emitted %d folds:", len(folds))
    for i, f in enumerate(folds):
        log.info("  fold %d: train %s -> %s | val %s -> %s | test %s -> %s",
                 i, _date_only(str(f["train"].start)), _date_only(str(f["train"].end)),
                 _date_only(str(f["val"].start)), _date_only(str(f["val"].end)),
                 _date_only(str(f["test"].start)), _date_only(str(f["test"].end)))

    wf_windows_required = base_cfg.get("gates", {}).get("wf_windows", 8)
    if len(folds) < wf_windows_required:
        log.error("splitter emitted %d folds < gates.wf_windows %d — abort",
                  len(folds), wf_windows_required); return 3

    if args.folds:
        wanted = [int(x) for x in args.folds.split(",")]
        folds_subset = [(i, folds[i]) for i in wanted if 0 <= i < len(folds)]
    else:
        folds_subset = list(enumerate(folds))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = PROJECT_ROOT / "results" / f"{pfx_us}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("SG-1-BTC Stage 3 WF Dispatcher — config-driven (prefix=%s)", base_prefix)
    log.info("  config:      %s", args.config.name)
    log.info("  folds:       %s of %d total", [i for i, _ in folds_subset], len(folds))
    log.info("  seeds:       %s", seeds)
    log.info("  total runs:  %d", len(folds_subset) * len(seeds))
    if args.slots:
        log.info("  slots:       %s (cross-GPU)", args.slots)
    else:
        log.info("  instance:    %s (gpu=%s, concurrent=%d)",
                 args.instance, args.gpu, args.concurrent)
    log.info("  results dir: %s", results_dir.relative_to(PROJECT_ROOT))
    log.info("  dry_run:     %s", args.dry_run)
    log.info("=" * 70)

    manifest = {
        "timestamp": timestamp,
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "instance": args.instance,
        "gpu": args.gpu,
        "concurrent": args.concurrent,
        "slots": args.slots,
        "seeds": seeds,
        "base_prefix": base_prefix,
        "n_folds": len(folds_subset),
        "dry_run": args.dry_run,
        "folds": [],
    }

    # Phase 1: materialize + validate all derived configs upfront
    derived: dict[int, Path] = {}
    for fold_idx, fold in folds_subset:
        cfg_path = write_fold_config(base_cfg, fold_idx, fold, timestamp,
                                     runtime_dir, pfx_us)
        derived[fold_idx] = cfg_path
        if not validate(cfg_path):
            log.error("aborting: validator rejected %s", cfg_path.name); return 3

    # Phase 2: dispatch fold cells sequentially. `launch_l1_multiseed.py` exits
    # after deploys finish (~5-9 min), NOT after training finishes. Without
    # WandB polling between cells, all 8 folds would land on the slot pool
    # simultaneously causing VRAM OOM.
    project = base_cfg.get("wandb", {}).get("project", "FinRL-Pro-DS")
    entity = base_cfg.get("wandb", {}).get("entity", "bigcan-chiwin-technology")

    _POLL_PY = (
        "import json, sys, wandb\n"
        "api = wandb.Api(timeout=30)\n"
        "entity, project, fid, sid = sys.argv[1:5]\n"
        "runs = list(api.runs(f'{entity}/{project}', filters={'$and': [\n"
        "    {'tags': {'$in': ['wf-stage3']}},\n"
        "    {'tags': {'$in': [f'fold-{fid}']}},\n"
        "    {'tags': {'$in': [f'study-id-{sid}']}},\n"
        "]}))\n"
        "states = [r.state for r in runs]\n"
        "counts = {s: states.count(s) for s in set(states)}\n"
        "terminal = sum(1 for s in states if s in ('finished','crashed','failed'))\n"
        "print(json.dumps({'n': len(runs), 'terminal': terminal,\n"
        "                  'counts': counts, 'finished': states.count('finished')}))\n"
    )

    def wait_runs_finished(fold_idx_: int, expected_n: int,
                           poll_every_s: int = 120, max_wait_h: float = 14.0) -> int:
        """Block until all `expected_n` runs tagged for this fold cell are
        terminal. Returns count finished cleanly. Fresh subprocess per poll
        (S509 INV: api.runs() reuse hangs after hours)."""
        deadline = time.time() + max_wait_h * 3600
        last_log = None
        while time.time() < deadline:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _POLL_PY,
                     entity, project, str(fold_idx_), timestamp],
                    capture_output=True, text=True, timeout=180,
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    log.warning("poll failed (rc=%d): %s", proc.returncode, proc.stderr[-300:])
                    time.sleep(poll_every_s); continue
                data = json.loads(proc.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                log.warning("poll subprocess hit 180s timeout — retrying"); time.sleep(poll_every_s); continue
            except Exception as e:
                log.warning("poll subprocess raised %s — retrying", e); time.sleep(poll_every_s); continue

            n_runs, terminal, counts = data["n"], data["terminal"], data["counts"]
            cur = (n_runs, terminal, tuple(sorted(counts.items())))
            if cur != last_log:
                log.info("waiting on fold %d: runs=%d terminal=%d/%d by_state=%s",
                         fold_idx_, n_runs, terminal, expected_n, counts)
                last_log = cur
            if n_runs >= expected_n and terminal >= expected_n:
                ok = data["finished"]
                log.info("fold %d wandb-side complete: %d/%d finished, others: %s",
                         fold_idx_, ok, expected_n,
                         {k: v for k, v in counts.items() if k != "finished"})
                return ok
            time.sleep(poll_every_s)
        log.error("wandb wait timed out after %.1fh on fold %d", max_wait_h, fold_idx_)
        return -1

    for fold_idx, fold in folds_subset:
        cfg_path = derived[fold_idx]
        rc, elapsed = launch_cell(
            cfg_path, fold_idx, args.slots,
            args.instance, args.gpu, args.concurrent, args.dry_run,
            seeds, base_prefix,
        )

        finished_ok = -2
        if rc == 0 and not args.dry_run:
            finished_ok = wait_runs_finished(fold_idx, expected_n=len(seeds))

        manifest["folds"].append({
            "fold_idx": fold_idx,
            "train_start": _date_only(str(fold["train"].start)),
            "train_end":   _date_only(str(fold["train"].end)),
            "val_start":   _date_only(str(fold["val"].start)),
            "val_end":     _date_only(str(fold["val"].end)),
            "test_start":  _date_only(str(fold["test"].start)),
            "test_end":    _date_only(str(fold["test"].end)),
            "config":      str(cfg_path.relative_to(PROJECT_ROOT)),
            "study_id":    timestamp,
            "exit_code":   rc,
            "elapsed_s":   round(elapsed, 1),
            "wandb_finished_ok": finished_ok,
        })
        with (results_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if rc != 0:
            log.error("aborting: fold %d launcher exit=%d", fold_idx, rc); return 1
        if finished_ok == -1:
            log.error("aborting: fold %d wandb wait timed out", fold_idx); return 1
        if finished_ok != -2 and finished_ok < len(seeds):
            log.warning("fold %d: only %d/%d seeds finished cleanly; continuing",
                        fold_idx, finished_ok, len(seeds))

    log.info("=" * 70)
    log.info("Dispatcher complete: %d/%d folds OK", len(folds_subset), len(folds_subset))
    log.info("Manifest: %s", (results_dir / "manifest.json").relative_to(PROJECT_ROOT))
    log.info("Next: python scripts/sg1_btc_velotrade_ensemble_eval.py "
             "--wf_config %s --gates_file configs/sg1_btc_velotrade_ensemble.gates.yaml",
             args.config.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
