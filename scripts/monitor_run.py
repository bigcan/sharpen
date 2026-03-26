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

import argparse
import os
import sys
import time
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Thresholds
Q_DIVERGENCE_THRESHOLD = 1e4

def check_remote_pid(pid):
    """Check if remote PID is running via SSH."""
    from scripts.remote_cmd import remote_cmd

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
    import wandb
    from scripts.collect_run import collect_run

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
                    if not no_collect:
                        collect_run(run_id, skip_report=True)
                    return

            # 3. Check Metrics (via run.summary — fast, no scan_history)
            summary = run.summary._json_dict
            step = summary.get('_step', summary.get('step', 0))
            loss = summary.get('agent/loss_total')
            q_mean = summary.get('agent/q_value/mean', summary.get('agent/q_qty_mean'))

            # Stall Detection
            if step > last_step:
                last_step = step
                last_step_time = time.time()
            elif time.time() - last_step_time > (poll_interval * 5):
                print(f"[WARN] STALL DETECTED: No new steps for {(time.time()-last_step_time)/60:.1f} min")

            # NaN Detection
            if loss is not None and (
                str(loss).lower() in ('nan', 'inf')
                or (isinstance(loss, float) and (loss != loss or abs(loss) == float('inf')))
            ):
                print(f"[ERROR] NaN/Inf DETECTED in loss at step {step}")

            # Q-Divergence
            if q_mean is not None and isinstance(q_mean, (int, float)) and abs(q_mean) > Q_DIVERGENCE_THRESHOLD:
                print(f"[WARN] Q-DIVERGENCE: |q_mean| = {abs(q_mean):.2e} > {Q_DIVERGENCE_THRESHOLD:.0e}")

            # Safe formatting
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) and loss == loss else str(loss)
            q_str = f"{q_mean:.2f}" if isinstance(q_mean, (int, float)) else str(q_mean)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {state} | Step: {step:,} | Loss: {loss_str} | Q: {q_str}")

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
