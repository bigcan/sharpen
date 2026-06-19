#!/usr/bin/env python
"""GMGP1-BTC Feature-Engineering Ablation — dispatcher.

Sweeps the FEATURE AXIS (which features + scales the agent sees) across 6 rungs
at N=5 seeds each (30 runs total) on the clean-Bybit-BTC / de-leaked /
cost-corrected substrate. HPs, data, costs, budget and eval protocol are FROZEN
(canary HPO trial #5) — the feature set is the only independent variable.

Rung table + hypotheses live in configs/gmgp1_btc_feature_ablation.gates.yaml
(pre-registered). Each rung derives a config from
configs/gmgp1_btc_feature_ablation_base.yaml by overriding only the feature
axis, then launches scripts/launch_l1_multiseed.py.

Per Protocol v2 this is a pure Stage-3 (l1-multiseed) sweep — NO HPO, no fused
pipeline. Each rung = one decision artifact (5 seed runs).

Fleet utilization: rungs run SEQUENTIALLY; within each rung the 5 seeds fan out
across all slots via the launcher's `--slots` mode, which serializes per-host
setup internally (the S551 shared-workspace pip-race guard). Running separate
launchers per GPU on the same host is UNSAFE (concurrent unzip/pip on the shared
remote workspace) — do not do that; use --slots.

Usage:
    # Smoke: r0 only, 1 seed (123), short budget, single GPU — de-risks the
    # never-run single-feature path end-to-end on the fleet.
    python scripts/run_feature_ablation.py --smoke --instance gpuhub-2 --gpu 0

    # Full ladder, all 3 GPUs, rungs sequential / seeds fanned per rung:
    python scripts/run_feature_ablation.py --full \
        --slots gpuhub-1:0,gpuhub-1:1,gpuhub-2:0

    # Subset of rungs (e.g. resume):
    python scripts/run_feature_ablation.py --full --rungs r3,r4,r5 \
        --slots gpuhub-1:0,gpuhub-1:1,gpuhub-2:0 --study_id 20260616_171000

    # Dry run: generate + validate configs, print launch commands, no spawn.
    python scripts/run_feature_ablation.py --full --dry_run

Manifest: writes results/feature_ablation_<ts>/manifest.json with
(rung_id, label, feature_indices, scales, seeds, run_name_prefix, status,
elapsed_s) entries for scripts/analyze_feature_ablation.py.
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
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "gmgp1_btc_feature_ablation_base.yaml"
DEFAULT_GATES_FILE = PROJECT_ROOT / "configs" / "gmgp1_btc_feature_ablation.gates.yaml"
DEFAULT_PREFIX = "gmgp1-btc-featabl"
RUNTIME_DIR = PROJECT_ROOT / "configs" / "_feature_ablation_runtime"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_l1_multiseed.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_config.py"

# Smoke uses a short budget so the on-fleet end-to-end check is cheap (~minutes
# of train, not hours). Real rungs use the base config's frozen 2M budget.
SMOKE_STEPS = 20_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] feat_ablation - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("feat_ablation")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_rung_config(base: dict, rung: dict, study_id: str,
                      runtime_dir: Path, smoke: bool) -> Path:
    """Materialize a derived config overriding ONLY the feature axis for `rung`.

    Overrides: features.{feature_indices,scales,features_per_scale},
    env.{scales,features_per_scale}, network.scale_encoder.input_size. n_scales
    is auto-derived from len(scales) by sac_trainer.py — do NOT set it here.
    """
    cfg = copy.deepcopy(base)
    fidx = [int(i) for i in rung["feature_indices"]]
    scales = [int(s) for s in rung["scales"]]
    n_feat = len(fidx)

    cfg.setdefault("features", {})
    cfg["features"]["feature_indices"] = fidx
    cfg["features"]["scales"] = scales
    cfg["features"]["features_per_scale"] = n_feat

    cfg.setdefault("env", {})
    cfg["env"]["scales"] = scales
    cfg["env"]["features_per_scale"] = n_feat

    cfg.setdefault("network", {}).setdefault("scale_encoder", {})
    cfg["network"]["scale_encoder"]["input_size"] = n_feat

    if smoke:
        cfg.setdefault("training", {})["total_timesteps"] = SMOKE_STEPS

    tags = list(cfg.setdefault("wandb", {}).setdefault("tags", []))
    for t in (f"rung-{rung['id']}", f"study-id-{study_id}"):
        if t not in tags:
            tags.append(t)
    cfg["wandb"]["tags"] = tags

    runtime_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    out = runtime_dir / f"{DEFAULT_PREFIX}_{rung['id']}_{rung['label']}{suffix}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def validate(cfg_path: Path) -> bool:
    """Run scripts/validate_config.py --stage l1-multiseed on a derived config."""
    cmd = [sys.executable, str(VALIDATOR),
           "--config", str(cfg_path), "--stage", "l1-multiseed"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    blob = (proc.stdout or "") + (proc.stderr or "")
    if "status=FAIL" in blob or proc.returncode != 0:
        log.error("validator FAIL on %s\n%s", cfg_path.name, blob)
        return False
    log.info("validator %s on %s",
             "WARN" if "status=WARN" in blob else "PASS", cfg_path.name)
    return True


def launch_rung(cfg_path: Path, rung: dict, seeds: list[int],
                instance: str | None, gpu: str | None, concurrent: int,
                slots: str | None, dry_run: bool) -> tuple[int, float]:
    """Invoke launch_l1_multiseed.py for one rung. Blocks until all seeds done.

    When `slots` is set the launcher fans seeds across GPUs and serializes
    per-host setup internally; instance/gpu/concurrent are ignored.
    """
    seeds_csv = ",".join(str(s) for s in seeds)
    # run_name_prefix must be lowercase/digits/hyphens (naming.py regex). The
    # launcher appends _<ts> and -seed{N}; study disambiguation is via wandb tag.
    prefix = f"{DEFAULT_PREFIX}-{rung['id']}"
    cfg_rel = cfg_path.relative_to(PROJECT_ROOT).as_posix()
    cmd = [
        sys.executable, str(LAUNCHER),
        "--config", cfg_rel,
        "--seeds", seeds_csv,
        "--run_name_prefix", prefix,
        "--separate_runs",   # per-seed runs (avoids consolidated-summary writer race)
    ]
    if slots:
        cmd.extend(["--slots", slots])
        target = f"slots={slots}"
    else:
        cmd.extend(["--instance", instance or "gpuhub-2",
                    "--gpu", gpu or "0",
                    "--concurrent", str(concurrent)])
        target = f"{instance} (gpu={gpu}, concurrent={concurrent})"
    if dry_run:
        cmd.append("--dry_run")

    log.info("launching rung=%s (%s) feats=%s scales=%s seeds=%s -> %s",
             rung["id"], rung["label"], rung["feature_indices"],
             rung["scales"], seeds_csv, target)
    log.info("  cmd: %s", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if proc.returncode != 0:
        log.error("rung=%s FAILED (exit=%d, %.0fs)", rung["id"], proc.returncode, elapsed)
    else:
        log.info("rung=%s OK (%.0fs)", rung["id"], elapsed)
    return proc.returncode, elapsed


_SEED_RE = re.compile(r"-seed(\d+)_")
WORKSTREAM_TAG = "feature-ablation"


def _poll_rung_status(project: str, entity: str, rung_id: str, study_id: str,
                      seeds: list[int]) -> dict[int, str]:
    """Fresh-Api WandB poll of one rung's seeds. Returns {seed: status} where
    status in {good, running, bad, missing}. good = a run for that seed is
    `finished` AND has backtest_test/profit_factor (a step-2 crash that wandb
    marks `finished` has no PF -> bad). missing = never appeared (deploy EOF)."""
    import wandb
    api = wandb.Api(timeout=60)  # FRESH per poll (S537: stale public-collection cache)
    runs = list(api.runs(f"{entity}/{project}", filters={"$and": [
        {"tags": {"$in": [WORKSTREAM_TAG]}},
        {"tags": {"$in": [f"rung-{rung_id}"]}},
        {"tags": {"$in": [f"study-id-{study_id}"]}},
    ]}))
    rank = {"good": 3, "running": 2, "bad": 1}
    by_seed: dict[int, str] = {}
    for r in runs:
        m = _SEED_RE.search(r.name)
        if not m or int(m.group(1)) not in seeds:
            continue
        sd = int(m.group(1))
        summ = dict(r.summary_metrics or {})
        if r.state == "finished" and "backtest_test/profit_factor" in summ:
            cur = "good"
        elif r.state == "running":
            cur = "running"
        else:
            cur = "bad"
        if sd not in by_seed or rank[cur] > rank[by_seed[sd]]:
            by_seed[sd] = cur  # best status across a seed's (possibly retried) runs
    return {sd: by_seed.get(sd, "missing") for sd in seeds}


def wait_for_rung(project: str, entity: str, rung_id: str, study_id: str,
                  seeds: list[int], poll_s: int = 120,
                  max_wait_h: float = 12.0) -> dict[int, str] | None:
    """Block until none of `seeds` is still `running` (deploy-and-detach means the
    launcher returns early; THIS is the real per-rung barrier — without it all
    rungs launch at once and oversubscribe the GPUs). Returns final status map."""
    deadline = time.time() + max_wait_h * 3600
    errs = 0
    last = None
    while time.time() < deadline:
        try:
            st = _poll_rung_status(project, entity, rung_id, study_id, seeds)
            errs = 0
        except Exception as exc:  # transient wandb/network
            errs += 1
            log.warning("wandb poll error #%d rung=%s: %s", errs, rung_id, exc)
            if errs >= 10:
                log.error("aborting wait: 10 consecutive wandb errors rung=%s", rung_id)
                return None
            time.sleep(poll_s)
            continue
        snap = tuple(sorted(st.items()))
        if snap != last:
            log.info("rung=%s status: %s", rung_id, {k: st[k] for k in sorted(st)})
            last = snap
        if not any(v == "running" for v in st.values()):
            return st  # all terminal (good/bad/missing)
        time.sleep(poll_s)
    log.error("rung=%s barrier timed out after %.1fh", rung_id, max_wait_h)
    return _poll_rung_status(project, entity, rung_id, study_id, seeds)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                      help=f"Smoke: r0 only, 1 seed, {SMOKE_STEPS} steps")
    mode.add_argument("--full", action="store_true",
                      help="Full ladder (all rungs from gates file unless --rungs)")
    p.add_argument("--instance", default="gpuhub-2",
                   help="Single-slot fallback instance (default gpuhub-2). Ignored if --slots.")
    p.add_argument("--gpu", default="0",
                   help="Single-slot CUDA_VISIBLE_DEVICES (default 0). Ignored if --slots.")
    p.add_argument("--concurrent", type=int, default=2,
                   help="Single-slot within-rung seed concurrency (default 2 — VRAM-bound).")
    p.add_argument("--slots", default=None,
                   help="Cross-GPU slot pool, e.g. 'gpuhub-1:0,gpuhub-1:1,gpuhub-2:0'. "
                        "Each rung fans its seeds across these slots; rungs stay sequential.")
    p.add_argument("--rungs", default=None,
                   help="Comma-separated rung-id subset (e.g. 'r3,r4,r5'). Default: all.")
    p.add_argument("--base_config", type=Path, default=DEFAULT_BASE_CONFIG)
    p.add_argument("--gates_file", type=Path, default=DEFAULT_GATES_FILE)
    p.add_argument("--study_id", default=None,
                   help="Override study-id wandb tag (pass original to RESUME a study).")
    p.add_argument("--max_retry", type=int, default=2,
                   help="Per-rung retries for seeds that crash or fail to deploy "
                        "(EOFError) — only the missing/bad seeds are re-launched.")
    p.add_argument("--max_wait_h", type=float, default=12.0,
                   help="Per-rung barrier deadline (hours) waiting on WandB completion.")
    p.add_argument("--dry_run", action="store_true",
                   help="Generate + validate configs, print launch commands, do not spawn.")
    args = p.parse_args()

    for f in (args.base_config, args.gates_file, LAUNCHER, VALIDATOR):
        if not Path(f).exists():
            log.error("required file not found: %s", f)
            return 2

    base = load_yaml(args.base_config)
    gates = load_yaml(args.gates_file)
    all_rungs: list[dict] = list(gates["rungs"])
    all_seeds: list[int] = [int(s) for s in gates["seeds"]]

    if args.smoke:
        rungs = [next(r for r in all_rungs if r["id"] == "r0")]
        seeds = [all_seeds[0]]
    else:
        if args.rungs:
            want = [s.strip() for s in args.rungs.split(",")]
            by_id = {r["id"]: r for r in all_rungs}
            missing = [w for w in want if w not in by_id]
            if missing:
                log.error("unknown rung ids: %s (have %s)", missing, list(by_id))
                return 2
            rungs = [by_id[w] for w in want]
        else:
            rungs = all_rungs
        seeds = all_seeds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_id = args.study_id or timestamp
    results_dir = PROJECT_ROOT / "results" / f"feature_ablation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 78)
    log.info("Feature-Ablation Dispatcher (workstream=%s)", gates.get("workstream"))
    log.info("  base config: %s", args.base_config.name)
    log.info("  gates file:  %s", args.gates_file.name)
    log.info("  study_id:    %s   smoke=%s  dry_run=%s", study_id, args.smoke, args.dry_run)
    log.info("  rungs:       %s", [r["id"] for r in rungs])
    log.info("  seeds:       %s", seeds)
    target = f"slots={args.slots}" if args.slots else \
        f"{args.instance}:gpu{args.gpu} concurrent={args.concurrent}"
    log.info("  target:      %s", target)
    log.info("=" * 78)

    # ---- Phase 1: generate + validate ALL rung configs up front (fail fast) ----
    derived: list[tuple[dict, Path]] = []
    for rung in rungs:
        cfg_path = write_rung_config(base, rung, study_id, RUNTIME_DIR, args.smoke)
        if not validate(cfg_path):
            log.error("aborting: validation FAILED for rung=%s", rung["id"])
            return 3
        derived.append((rung, cfg_path))
    log.info("all %d rung config(s) generated + validated", len(derived))

    # ---- Phase 2: launch rungs sequentially ----
    manifest: dict = {
        "study_id": study_id,
        "workstream": gates.get("workstream"),
        "base_config": str(args.base_config),
        "gates_file": str(args.gates_file),
        "seeds": seeds,
        "smoke": args.smoke,
        "dry_run": args.dry_run,
        "target": target,
        "rungs": [],
    }
    project = base.get("wandb", {}).get("project", "FinRL-Pro-DS")
    entity = base.get("wandb", {}).get("entity", "bigcan-chiwin-technology")
    overall_rc = 0
    for rung, cfg_path in derived:
        rung_start = time.time()
        final: dict[int, str] = {}
        # RESUME / idempotent top-up: skip seeds already `good` in this study
        # (a completed 2M+eval run is step-determined valid regardless of the
        # GPU contention it trained under). With a fresh study_id every seed is
        # `missing` -> full run; with --study_id <prior> only the gaps re-launch.
        if not args.dry_run:
            try:
                existing = _poll_rung_status(project, entity, rung["id"], study_id, seeds)
                for sd, v in existing.items():
                    if v == "good":
                        final[sd] = "good"
            except Exception as exc:
                log.warning("resume-poll failed rung=%s (%s) — launching all seeds",
                            rung["id"], exc)
        todo = [sd for sd in seeds if final.get(sd) != "good"]
        if final:
            log.info("rung=%s resume: %d/%d seeds already good -> launching %s",
                     rung["id"], len(final), len(seeds), todo or "none")
        for attempt in range(args.max_retry + 1):
            if not todo:
                break
            log.info("rung=%s attempt %d/%d: launching seeds %s",
                     rung["id"], attempt, args.max_retry, todo)
            launch_rung(cfg_path, rung, todo, args.instance, args.gpu,
                        args.concurrent, args.slots, args.dry_run)
            if args.dry_run:
                break
            # BARRIER: launch_l1_multiseed is deploy-and-detach — block here until
            # these seeds finish on the GPUs, else the next rung oversubscribes.
            st = wait_for_rung(project, entity, rung["id"], study_id, todo,
                               max_wait_h=args.max_wait_h)
            if st is None:  # wandb wait gave up — record and stop retrying this rung
                break
            final.update(st)
            todo = [sd for sd, v in st.items() if v in ("bad", "missing")]
            if todo:
                log.warning("rung=%s seeds %s bad/missing after attempt %d -> retry",
                            rung["id"], todo, attempt)
        good = sorted(sd for sd, v in final.items() if v == "good")
        status = ("OK" if len(good) >= len(seeds)
                  else "PARTIAL" if good else "FAIL")
        manifest["rungs"].append({
            "rung_id": rung["id"],
            "label": rung["label"],
            "feature_indices": rung["feature_indices"],
            "scales": rung["scales"],
            "config": cfg_path.relative_to(PROJECT_ROOT).as_posix(),
            "run_name_prefix": f"{DEFAULT_PREFIX}-{rung['id']}",
            "seeds": seeds,
            "good_seeds": good,
            "seed_status": {str(k): v for k, v in sorted(final.items())},
            "status": status,
            "elapsed_s": round(time.time() - rung_start, 1),
        })
        with (results_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log.info("rung=%s %s (%d/%d good seeds)", rung["id"], status, len(good), len(seeds))
        if status != "OK":
            overall_rc = 1

    log.info("=" * 78)
    log.info("dispatcher done. manifest: %s", results_dir / "manifest.json")
    log.info("  results: %s", " ".join(
        f"{r['rung_id']}={r['status']}" for r in manifest["rungs"]))
    log.info("=" * 78)
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
