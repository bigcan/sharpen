"""Launch the FULL gmgp1-spx500 A/B HPO fleet in one deterministic shot, then verify it.

THE SYMMETRY IS THE POINT. Variant A (long/short) and variant B (long-only) must each get
the SAME number of Optuna workers, because worker count is a nuisance variable on the search
itself: TPE started with more concurrency has more trials in flight before earlier results
land, which is effectively more random exploration. A 2-vs-1 split would confound the A/B
with search parallelism. So the plan below places exactly one A worker and one B worker on
every GPU.

WHY ONE SHOT AND WHY IT VERIFIES. Hand-launching these accumulated duplicates twice: a
shell for-loop around deploy_bare_metal.py was reaped mid-flight and silently left a second
`ls-w2` on gpuhub-2, and a retry added a second worker to gpuhub-1 GPU0. Both times the
fleet ended asymmetric — the one thing that invalidates the comparison. This script is
idempotent-by-refusal: it aborts if any run_full_pipeline process is already alive, and
after launching it re-reads the fleet and asserts the inventory matches the plan exactly.

CONCURRENCY IS FREE HERE, measured: a single run uses ~1.15 of 208 cores and 13-18% of one
4090. Doubling to two runs per GPU lifted utilization to 78-98% with no per-run slowdown,
so the A/B costs one wall-clock window rather than two.

Usage:
    python scripts/launch_spx500_ab_fleet.py --trials-per-worker 10
    python scripts/launch_spx500_ab_fleet.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import paramiko
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/DeepScalper"

# (instance, gpu) slots. One A worker and one B worker land on each.
SLOTS = [("gpuhub-1", "0"), ("gpuhub-1", "1"), ("gpuhub-2", "0")]
VARIANTS = {"ls": "configs/gmgp1_spx500_ls_hpo.yaml",
            "lo": "configs/gmgp1_spx500_lo_hpo.yaml"}


def connect(inst: dict) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(inst["host"], port=int(inst["port"]), username="root",
                password=inst["password"], timeout=30)
    return ssh


def run(ssh: paramiko.SSHClient, cmd: str) -> str:
    _in, out, _err = ssh.exec_command(cmd)
    return out.read().decode()


def inventory(conns: dict[str, paramiko.SSHClient]) -> Counter:
    """Map run_name -> count of live pipeline processes, across the fleet."""
    c: Counter = Counter()
    for name, ssh in conns.items():
        txt = run(ssh, "ps -eo args | grep 'python -u scripts/run_full_pipeline' | grep -v grep")
        for line in txt.splitlines():
            m = re.search(r"--run_name (\S+)", line)
            if m:
                c[f"{name}:{m.group(1)}"] += 1
    return c


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-per-worker", type=int, default=10)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    reg = json.loads((ROOT / "instances.json").read_text(encoding="utf-8"))["instances"]
    db_url = os.environ["DISTRIBUTED_HPO_DB_URL"]
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    conns = {n: connect(reg[n]) for n in {i for i, _ in SLOTS}}

    if args.verify_only:
        inv = inventory(conns)
        for k, v in sorted(inv.items()):
            print(f"  {k}  x{v}")
        print(f"total distinct workers: {len(inv)}")
        return 0

    live = inventory(conns)
    if live:
        print("REFUSING TO LAUNCH — pipeline processes already running:", file=sys.stderr)
        for k, v in sorted(live.items()):
            print(f"  {k} x{v}", file=sys.stderr)
        print("Kill them first (pkill -f run_full_pipeline) so the fleet stays symmetric.",
              file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    planned = []
    for wi, (inst, gpu) in enumerate(SLOTS):
        for variant, cfg in VARIANTS.items():
            run_name = f"spx500-{variant}-hpo-w{wi}_{stamp}"
            log = f"run_{stamp}_{variant}_w{wi}.log"
            inner = (f"python -u scripts/run_full_pipeline.py --config {shlex.quote(cfg)}"
                     f" --agent sac --stage hpo --trials {args.trials_per_worker}"
                     f" --run_name {shlex.quote(run_name)}"
                     f" --hpo_storage {shlex.quote(db_url)}")
            cmd = (f"cd {REMOTE} && export PATH=/root/miniconda3/bin:$PATH"
                   f" && export CUDA_VISIBLE_DEVICES={gpu}"
                   + (f" && export WANDB_API_KEY={shlex.quote(wandb_key)}" if wandb_key else "")
                   + f" && (ulimit -n 65535 || true)"
                   f" && nohup {inner} > {log} 2>&1 & echo OK")
            run(conns[inst], cmd)
            planned.append(f"{inst}:{run_name}")
            print(f"launched {inst} gpu{gpu} variant={variant} run_name={run_name}")
            time.sleep(2)

    print("\nwaiting 60s for processes to register...")
    time.sleep(60)
    inv = inventory(conns)
    print("\n=== FLEET INVENTORY ===")
    for k in sorted(inv):
        print(f"  {k}  (x{inv[k]} procs)")

    missing = [p for p in planned if p not in inv]
    extra = [k for k in inv if k not in planned]
    a = sum(1 for k in inv if "-ls-" in k)
    b = sum(1 for k in inv if "-lo-" in k)
    print(f"\nvariant A workers={a}  variant B workers={b}  (must be equal)")
    if missing or extra or a != b:
        print(f"FAIL: missing={missing} extra={extra}", file=sys.stderr)
        return 1
    print("OK: fleet symmetric and matches plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
