#!/usr/bin/env python3
"""
DeepScalper Monitoring Script
=============================
Standalone monitor for remote training runs on GPUHub.
Detects:
  - Process death (via SSH check)
  - WandB run failures/crashes
  - Training stalls (no new steps)
  - Q-value divergence (instability)
  - NaN loss

Triggers `collect_run.py` on completion/failure unless --no_collect is set.

Usage:
    python scripts/monitor_run.py --run_id <ID>
    python scripts/monitor_run.py --run_id <ID> --poll 60 --no_collect
"""

import sys
import os
import time
import argparse
import wandb
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.remote_cmd import remote_cmd
from scripts.collect_run import collect_run

def check_remote_pid(pid):
    """Check if remote PID is running via SSH."""
    if not pid:
        return False
    try:
        # ps -p PID returns non-zero exit code if not found
        out = remote_cmd(f"ps -p {pid} > /dev/null && echo 'RUNNING' || echo 'DEAD'", timeout=10)
        return "RUNNING" in out
    except Exception:
        return False  # Assume dead/unreachable on error

def monitor_run(run_id, pid=None, poll_interval=120, max_wait=7200, no_collect=False):
    """
    Monitor a specific WandB run until completion or failure.
    
    Args:
        run_id: WandB run ID
        pid: Remote process ID (optional, for stricter checking)
        poll_interval: Seconds between checks
        max_wait: Max seconds to monitor
        no_collect: If True, do not trigger collect_run.py on finish
    """
    print(f"\n{'='*60}")
    print(f"  Monitoring Run: {run_id}")
    print(f"  Poll Interval: {poll_interval}s | Max Wait: {max_wait/3600:.1f}h")
    print(f"{'='*60}\n")
    
    api = wandb.Api()
    project = "bigcan-chiwin-technology/FinRL-Pro-DS"
    run_path = f"{project}/{run_id}"
    
    start_time = time.time()
    last_step = -1
    last_step_time = time.time()
    
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"[ERROR] Could not find run {run_id}: {e}")
        return

    while time.time() - start_time < max_wait:
        try:
            # Refresh run data
            run = api.run(run_path)
            state = run.state
            
            # 1. Check Run State
            if state in ['finished', 'crashed', 'failed']:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Run ended: {state.upper()}")
                if not no_collect and state == 'finished':
                    print("\n[COLLECT] Triggering collection...")
                    collect_run(run_id)
                elif state in ['crashed', 'failed']:
                     print(f"[WARN] Run {state} -- collection optional scan recommended.")
                     # We still collect on crash to get logs/traceback
                     if not no_collect:
                         print("\n[COLLECT] Triggering collection (logs match)...")
                         collect_run(run_id, skip_report=True) # Skip report for crashed runs
                return

            # 2. Check Process (if PID provided)
            if pid:
                is_alive = check_remote_pid(pid)
                if not is_alive:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEAD] Remote process {pid} DIED but WandB is {state}")
                    print("  This usually means a hard crash or OOM kill.")
                    if not no_collect: collect_run(run_id, skip_report=True)
                    return

            # 3. Check Metrics (Live History)
            # Scan last few rows
            history = list(run.scan_history(keys=['_step', 'train/loss', 'train/q_mean'], page_size=10))[-10:]
            if history:
                latest = history[-1]
                step = latest.get('_step', 0)
                loss = latest.get('train/loss', 0)
                q_mean = latest.get('train/q_mean', 0)
                
                # Stall Detection
                if step > last_step:
                    last_step = step
                    last_step_time = time.time()
                elif time.time() - last_step_time > (poll_interval * 5): # 5 cycles no step
                    print(f"[WARN] STALL DETECTED: No new steps for {(time.time()-last_step_time)/60:.1f} min")

                # NaN Detection
                if loss is not None and str(loss).lower() == 'nan':
                     print(f"[ERROR] NaN DETECTED in loss at step {step}")
                     # Could kill process here? For now, just alert.

                # Q-Divergence
                if q_mean and abs(q_mean) > 1e6:
                    print(f"[WARN] Q-DIVERGENCE: Mean Q > 1e6 ({q_mean:.2e}) at step {step}")

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {state} | Step: {step} | Loss: {loss:.4f} | Q: {q_mean:.2f}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {state} (Waiting for metrics...)")

        except Exception as e:
            print(f"Monitor error (retrying): {e}")

        time.sleep(poll_interval)
    
    print(f"\n[TIMEOUT] Monitoring timed out after {max_wait/3600:.1f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor a DeepScalper run")
    parser.add_argument("--run_id", required=True, help="WandB Run ID")
    parser.add_argument("--pid", help="Remote PID (optional, for process check)")
    parser.add_argument("--poll", type=int, default=120, help="Poll interval (seconds)")
    parser.add_argument("--max_wait", type=int, default=28800, help="Max wait time (seconds), default 8h")
    parser.add_argument("--no_collect", action="store_true", help="Disable auto-collection on finish")
    args = parser.parse_args()
    
    monitor_run(args.run_id, args.pid, args.poll, args.max_wait, args.no_collect)
