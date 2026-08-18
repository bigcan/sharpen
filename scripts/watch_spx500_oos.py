"""Emit one line per gmgp1-spx500 recent-OOS run as it reaches a terminal state, then exit.

Built for the Monitor tool: stdout is the event stream. Reports FAILED/CRASHED as well as
finished — a watcher that greps only for success is silent through a crash, which looks
exactly like "still running" and is how a dead stage goes unnoticed for hours.
"""
from __future__ import annotations

import argparse
import sys
import time

import wandb

PROJECT = "bigcan-chiwin-technology/FinRL-Pro-DS"
TERMINAL = {"finished", "crashed", "failed", "killed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", type=int, default=2)
    ap.add_argument("--poll", type=int, default=300)
    args = ap.parse_args()

    reported: set[str] = set()
    while True:
        try:
            api = wandb.Api()
            runs = [r for r in api.runs(PROJECT, order="-created_at")[:20]
                    if "-oos" in (r.name or "") and (r.name or "").startswith("spx500-")]
        except Exception as e:                      # transient WandB/network blip
            print(f"wandb query failed: {type(e).__name__}", flush=True)
            time.sleep(args.poll)
            continue

        for r in runs:
            if r.name in reported or r.state not in TERMINAL:
                continue
            reported.add(r.name)
            s = dict(r.summary)
            pf = s.get("backtest_test/profit_factor")
            if r.state == "finished" and pf is not None:
                print(f"OOS DONE {r.name}: PF={pf:.4f} "
                      f"ret={s.get('backtest_test/total_return', float('nan')):+.4%} "
                      f"dd={s.get('backtest_test/max_drawdown', float('nan')):+.2%}", flush=True)
            else:
                print(f"OOS {r.state.upper()} {r.name} (no usable result)", flush=True)

        if len(reported) >= args.expect:
            print("ALL OOS RUNS TERMINAL", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
