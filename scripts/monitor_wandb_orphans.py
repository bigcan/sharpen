#!/usr/bin/env python3
"""
Monitor GPU runs whose WandB connection died (401 Unauthorized on Apr 6 23:05).

Checks progress via SSH log tails, detects completion, and auto-syncs wandb.
Designed to be called repeatedly (e.g. via /loop).
"""
import sys
import io
import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import paramiko

# ── Instance Registry ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_FILE = PROJECT_ROOT / "instances.json"
STATE_FILE = PROJECT_ROOT / "results" / "wandb_orphan_state.json"

WANDB_KEY = "<REDACTED_WANDB_KEY>"
WANDB_BIN = "/root/miniconda3/bin/wandb"

# Orphaned runs: WandB shows "finished" but training is still active on GPU
ORPHAN_RUNS = {
    "gpuhub-1": [
        {
            "wandb_id": "0ugd38uo",
            "name": "GMGP2-BTC HPO",
            "pid": 546779,
            "log": "/workspace/DeepScalper/run_20260405_224719.log",
            "wandb_dir": "/workspace/DeepScalper/wandb/run-20260405_224721-0ugd38uo",
        },
        {
            "wandb_id": "ljx1r5up",
            "name": "SG-1 XAUUSD FTMO HPO",
            "pid": 561043,
            "log": "/workspace/DeepScalper/run_20260406_081959.log",
            "wandb_dir": "/workspace/DeepScalper/wandb/run-20260406_082002-ljx1r5up",
        },
    ],
    "gpuhub-2": [
        {
            "wandb_id": "yqxebqx3",
            "name": "MM v4 dYdX HPO",
            "pid": 188306,
            "log": "/workspace/DeepScalper/run_20260405_085857.log",
            "wandb_dir": "/workspace/DeepScalper/wandb/run-20260405_085901-yqxebqx3",
        },
        {
            "wandb_id": "9eedvvgk",
            "name": "Funding-Arb W15-W21",
            "pid": 199832,
            "log": "/workspace/DeepScalper/run_20260405_144701.log",
            "wandb_dir": "/workspace/DeepScalper/wandb/run-20260405_144701-9eedvvgk",
        },
    ],
    "gpuhub-3": [
        {
            "wandb_id": "axuehi6y",
            "name": "CMGP1 Crypto 2H HPO",
            "pid": 98619,
            "log": "/workspace/DeepScalper/run_20260405_134326.log",
            "wandb_dir": "/workspace/DeepScalper/wandb/run-20260405_134344-axuehi6y",
        },
        {
            "wandb_id": "oo48k2yy",
            "name": "SG-1 BTC Velotrade HPO",
            "pid": 124530,
            "log": "/workspace/DeepScalper/run_20260406_084425.log",
            "wandb_dir": "/workspace/DeepScalper/wandb/run-20260406_084428-oo48k2yy",
        },
    ],
}


def load_instances():
    """Load SSH credentials from instances.json."""
    with open(INSTANCES_FILE, encoding="utf-8") as f:
        return json.load(f)["instances"]


def ssh_exec(ssh, cmd, timeout=30):
    """Execute command via SSH, return stdout."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace")


def connect(inst_cfg):
    """Create SSH connection."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        inst_cfg["host"],
        port=inst_cfg["port"],
        username="root",
        password=inst_cfg["password"],
        timeout=15,
    )
    return ssh


def load_state():
    """Load persisted state (tracks which runs have been synced)."""
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"synced": {}, "last_check": None}


def save_state(state):
    """Persist state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_check"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def check_run(ssh, run_info):
    """Check a single run's status. Returns dict with status info."""
    pid = run_info["pid"]
    log = run_info["log"]
    wandb_dir = run_info["wandb_dir"]
    wid = run_info["wandb_id"]

    # Check if process is alive
    alive_out = ssh_exec(ssh, f"kill -0 {pid} 2>&1 && echo ALIVE || echo DEAD")
    is_alive = "ALIVE" in alive_out

    # Get last 5 log lines
    log_tail = ssh_exec(ssh, f"tail -5 {log} 2>/dev/null").strip()

    # Parse progress from log
    progress = None
    sps = None
    for line in reversed(log_tail.split("\n")):
        if "[HPO] step" in line:
            # e.g. [HPO] step 580000/750000 (77%) | SPS=651 | buf=100000
            try:
                parts = line.split("step ")[1]
                frac = parts.split(" ")[0]  # 580000/750000
                current, total = frac.split("/")
                pct = int(current) / int(total) * 100
                progress = f"{int(current):,}/{int(total):,} ({pct:.0f}%)"
                if "SPS=" in line:
                    sps = line.split("SPS=")[1].split(" ")[0].split("|")[0].strip()
            except (IndexError, ValueError):
                pass
            break
        elif "Window" in line and "Training Phase" in line:
            # Funding-arb: === Window 18: Full Training Phase ===
            import re
            m = re.search(r"Window (\d+).*Training Phase", line)
            progress = f"Window {m.group(1)} training" if m else "window training"
            break
        elif "Best params" in line:
            # Funding-arb trial completed
            progress = "trial done, next window"
            break
        elif "PropFirm" in line and "PASSED" in line:
            progress = "PropFirm PASSED (trial running)"
            break
        elif "WandB run finalized" in line or "HPO walk-forward COMPLETED" in line:
            progress = "COMPLETED"
            break
        elif "FAILED" in line and "PropFirm" not in line:
            progress = "ERROR"
            break

    # Check .wandb file size
    wandb_size = ssh_exec(ssh, f"ls -lh {wandb_dir}/run-{wid}.wandb 2>/dev/null | awk '{{print $5}}'").strip()

    return {
        "alive": is_alive,
        "progress": progress or "unknown",
        "sps": sps,
        "wandb_size": wandb_size or "?",
        "log_tail": log_tail,
    }


def sync_run(ssh, run_info):
    """Sync a completed run's wandb data to cloud."""
    wandb_dir = run_info["wandb_dir"]
    wid = run_info["wandb_id"]
    print(f"    Syncing {wid} to WandB cloud...")
    out = ssh_exec(
        ssh,
        f"cd /workspace/DeepScalper && WANDB_API_KEY={WANDB_KEY} {WANDB_BIN} sync {wandb_dir} "
        f"--entity bigcan-chiwin-technology --project FinRL-Pro-DS 2>&1",
        timeout=120,
    )
    success = "done." in out
    print(f"    {'OK' if success else 'FAILED'}: {out.strip()}")
    return success


def main():
    instances = load_instances()
    state = load_state()
    now = datetime.now(timezone(timedelta(hours=8)))

    print(f"\n{'='*72}")
    print(f"  WandB Orphan Monitor — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+8")
    print(f"{'='*72}")

    all_done = True
    summary = []

    for inst_name, runs in ORPHAN_RUNS.items():
        inst_cfg = instances[inst_name]
        print(f"\n  [{inst_name}]")

        try:
            ssh = connect(inst_cfg)
        except Exception as e:
            print(f"    SSH FAILED: {e}")
            all_done = False
            for r in runs:
                summary.append(f"  {r['wandb_id']}  {r['name']:<28} UNREACHABLE")
            continue

        for run_info in runs:
            wid = run_info["wandb_id"]
            name = run_info["name"]

            # Skip already-synced runs
            if state["synced"].get(wid):
                summary.append(f"  {wid}  {name:<28} SYNCED (done)")
                continue

            status = check_run(ssh, run_info)

            if status["alive"]:
                all_done = False
                sps_str = f"SPS={status['sps']}" if status["sps"] else ""
                line = f"  {wid}  {name:<28} RUNNING  {status['progress']:<24} {sps_str:<10} .wandb={status['wandb_size']}"
                print(f"    {wid} RUNNING  {status['progress']}  {sps_str}  .wandb={status['wandb_size']}")
            else:
                # Process died — either completed or crashed
                print(f"    {wid} FINISHED — attempting wandb sync...")
                ok = sync_run(ssh, run_info)
                if ok:
                    state["synced"][wid] = now.isoformat()
                line = f"  {wid}  {name:<28} {'SYNCED' if status.get('synced') else 'DONE'}  {status['progress']}"
                print(f"    Last log lines:")
                for l in status["log_tail"].split("\n")[-3:]:
                    print(f"      {l}")

            summary.append(line)

        ssh.close()

    # Summary table
    print(f"\n{'─'*72}")
    print(f"  SUMMARY")
    print(f"{'─'*72}")
    for line in summary:
        print(line)

    remaining = sum(1 for s in summary if "RUNNING" in s)
    synced = sum(1 for s in summary if "SYNCED" in s)
    print(f"\n  Active: {remaining}  |  Synced: {synced}  |  Total: {len(summary)}")

    if all_done:
        print("\n  ALL ORPHAN RUNS COMPLETED AND SYNCED.")
    else:
        print(f"\n  Next check: waiting for {remaining} runs to complete...")

    save_state(state)
    return 0 if all_done else 1


if __name__ == "__main__":
    sys.exit(main())
