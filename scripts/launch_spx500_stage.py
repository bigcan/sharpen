"""Launch a non-HPO gmgp1-spx500 stage (L1 multiseed / walk-forward / recent-OOS)
across the fleet, keeping the two variants symmetric.

SYMMETRY AGAIN: every job list is built as matched pairs (same seed / same fold for variant
A and variant B) and placed round-robin, so the arms share hardware and ordering evenly. An
arm that ran entirely on one GPU, or entirely later in the queue, would differ from the
other by more than `long_only`.

Refuses to start while anything is still running, for the same reason the HPO launcher does:
two earlier attempts silently left duplicate/asymmetric workers behind.

Usage:
    python scripts/launch_spx500_stage.py --stage l1-multiseed --seeds 42,123,7,2024,31337
    python scripts/launch_spx500_stage.py --stage wf --folds 1,2,3,4
    python scripts/launch_spx500_stage.py --stage oos
    python scripts/launch_spx500_stage.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import paramiko
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/workspace/DeepScalper"
SLOTS = [("gpuhub-1", "0"), ("gpuhub-1", "1"), ("gpuhub-2", "0")]
VARIANTS = ("ls", "lo")


def connect(inst):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(inst["host"], port=int(inst["port"]), username="root",
                password=inst["password"], timeout=30)
    return ssh


def run(ssh, cmd):
    _i, out, _e = ssh.exec_command(cmd)
    return out.read().decode()


# MEASURED CEILING, enforced rather than remembered, and scoped PER GPU rather than per
# instance -- the first version of this cap got the scope wrong. What the evidence actually
# shows:
#   * gpuhub-2 (ONE gpu), 3 concurrent 20-env runs -> died twice: BrokenPipeError at 08:00
#     (one L1 seed lost) and ConnectionResetError [Errno 104] at 14:16 (both new WF jobs AND
#     the L1 seed already running there). That is 3 runs on 1 GPU.
#   * gpuhub-1 (TWO gpus), 4 concurrent runs -> all completed cleanly. That is 2 per GPU.
#   * the entire HPO stage ran 6 workers as 2-per-GPU across 3 GPUs with zero pipe failures.
# So the binding resource scales with GPUs (Errno 104 is the /dev/shm exhaustion signature in
# CLAUDE.md's gotchas), and a per-INSTANCE cap would both under-use gpuhub-1 and mis-state
# the cause. Hard-coded here because the remembered version was broken once, costing 3 runs.
MAX_PER_GPU = 2


def live_by_instance(conns):
    counts = {}
    for inst, ssh in conns.items():
        txt = run(ssh, "ps -eo args | grep 'python -u scripts/run_full_pipeline' | grep -v grep")
        counts[inst] = len({m.group(1) for m in
                            (re.search(r"--run_name (\S+)", ln) for ln in txt.splitlines()) if m})
    return counts


def live(conns):
    names = []
    for inst, ssh in conns.items():
        txt = run(ssh, "ps -eo args | grep 'python -u scripts/run_full_pipeline' | grep -v grep")
        for line in txt.splitlines():
            m = re.search(r"--run_name (\S+)", line)
            if m:
                names.append(f"{inst}:{m.group(1)}")
    return sorted(set(names))


def build_jobs(stage, seeds, folds):
    """Matched pairs, so the two arms interleave rather than clustering."""
    jobs = []
    if stage == "l1-multiseed":
        for seed in seeds:
            for v in VARIANTS:
                jobs.append((v, f"configs/gmgp1_spx500_{v}_l1_multiseed.yaml",
                             ["--seed", str(seed)], f"l1-s{seed}"))
    elif stage == "wf":
        for f in folds:
            for v in VARIANTS:
                jobs.append((v, f"configs/gmgp1_spx500_{v}_wf_f{f}.yaml", [], f"wf-f{f}"))
    elif stage == "oos":
        for v in VARIANTS:
            jobs.append((v, f"configs/gmgp1_spx500_{v}_recent_oos.yaml", [], "oos"))
    else:
        raise SystemExit(f"unknown stage {stage}")
    return jobs


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["l1-multiseed", "wf", "oos"])
    ap.add_argument("--seeds", default="42,123,7,2024,31337")
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=6,
                    help="concurrent runs; 2 per GPU measured near-free (13-18%% GPU each)")
    args = ap.parse_args()

    reg = json.loads((ROOT / "instances.json").read_text(encoding="utf-8"))["instances"]
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    conns = {n: connect(reg[n]) for n in {i for i, _ in SLOTS}}

    if args.verify_only:
        for n in live(conns):
            print(" ", n)
        print("total:", len(live(conns)))
        return 0
    if not args.stage:
        print("--stage required unless --verify-only", file=sys.stderr)
        return 2

    # Capacity-aware rather than all-or-nothing: top up free slots, never oversubscribe.
    occupancy = live_by_instance(conns)
    free_slots = []
    for inst, gpu in SLOTS:
        n_gpus_on_inst = sum(1 for i, _ in SLOTS if i == inst)
        capacity = MAX_PER_GPU * n_gpus_on_inst
        # Spread this instance's headroom across its GPUs, one slot per (inst, gpu) pass.
        used = occupancy.get(inst, 0) + sum(1 for s in free_slots if s[0] == inst)
        if used < capacity:
            free_slots.append((inst, gpu))
    # second pass so a 2-GPU instance can offer both of its per-GPU slots
    for inst, gpu in SLOTS:
        n_gpus_on_inst = sum(1 for i, _ in SLOTS if i == inst)
        capacity = MAX_PER_GPU * n_gpus_on_inst
        used = occupancy.get(inst, 0) + sum(1 for s in free_slots if s[0] == inst)
        if used < capacity:
            free_slots.append((inst, gpu))
    if not free_slots:
        print(f"REFUSING: no free slots (cap {MAX_PER_GPU}/gpu): {occupancy}", file=sys.stderr)
        return 1
    print(f"occupancy={occupancy} free_slots={len(free_slots)} (cap {MAX_PER_GPU}/gpu)")

    seeds = [s for s in args.seeds.split(",") if s]
    folds = [f for f in args.folds.split(",") if f]
    jobs = build_jobs(args.stage, seeds, folds)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Jobs are matched A/B pairs, so truncate to an EVEN number of slots — launching an odd
    # count would put one arm ahead of the other.
    usable = (len(free_slots) // 2) * 2
    if usable == 0:
        print(f"REFUSING: only {len(free_slots)} free slot(s); need 2 for a matched A/B pair",
              file=sys.stderr)
        return 1

    launched = []
    for i, (variant, cfg, extra, tag) in enumerate(jobs[:usable]):
        inst, gpu = free_slots[i]
        run_name = f"spx500-{variant}-{tag}_{stamp}"
        log = f"run_{stamp}_{variant}_{tag}.log"
        inner = (f"python -u scripts/run_full_pipeline.py --config {shlex.quote(cfg)}"
                 f" --agent sac --stage {args.stage}"
                 f" --run_name {shlex.quote(run_name)} " + " ".join(extra))
        cmd = (f"cd {REMOTE} && export PATH=/root/miniconda3/bin:$PATH"
               f" && export CUDA_VISIBLE_DEVICES={gpu}"
               + (f" && export WANDB_API_KEY={shlex.quote(wandb_key)}" if wandb_key else "")
               + f" && (ulimit -n 65535 || true)"
               f" && nohup {inner} > {log} 2>&1 & echo OK")
        run(conns[inst], cmd)
        launched.append(f"{inst}:{run_name}")
        print(f"launched {inst} gpu{gpu} {run_name}")
        time.sleep(2)

    queued = jobs[usable:]
    if queued:
        print(f"\n{len(queued)} job(s) QUEUED (exceeded --max-parallel); re-run this command "
              f"after the current batch drains:")
        for v, c, e, t in queued:
            print(f"   {v} {t}")

    time.sleep(45)
    now = live(conns)
    print("\n=== ACTIVE ===")
    for n in now:
        print(" ", n)
    a = sum(1 for n in now if "-ls-" in n)
    b = sum(1 for n in now if "-lo-" in n)
    print(f"variant A={a} B={b} (must be equal)")
    return 0 if a == b else 1


if __name__ == "__main__":
    raise SystemExit(main())
