#!/usr/bin/env python3
"""Lightweight Optuna poll for Phase 1a leverage HPO.

Emits one stdout line per NEW trial completion (or kill/error). Use with the
Monitor tool — each line becomes a notification.

Tracks: trial number, state, PF (objective value), max_leverage param, and
running-best PF. Compares against the SG-1-XAUUSD baseline (PF 3.455) and
the leverage gate floor (baseline × 1.15 = 3.973) so each event is
self-contained and decision-ready.

Usage (called by Monitor):
    python scripts/_monitor_leverage_phase1a.py \\
        --study sg1_xauusd_leverage_narrow_hpo_20260427 \\
        --baseline_pf 3.455 --uplift 1.15 --poll_seconds 300
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import optuna

PYTHON_FLUSH = True


def _emit(line: str) -> None:
    print(line, flush=PYTHON_FLUSH)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--study", required=True)
    p.add_argument("--baseline_pf", type=float, required=True,
                   help="PF to beat (e.g. 3.455 for SG-1-XAUUSD S488 ensemble)")
    p.add_argument("--uplift", type=float, default=1.15,
                   help="Multiplicative gate (default 1.15 = +15%)")
    p.add_argument("--poll_seconds", type=int, default=300)
    p.add_argument("--target_trials", type=int, default=12,
                   help="Total expected trials (exit when reached)")
    args = p.parse_args()

    db = os.environ.get("DISTRIBUTED_HPO_DB_URL")
    if not db:
        _emit("[FATAL] DISTRIBUTED_HPO_DB_URL not set")
        sys.exit(2)

    gate = args.baseline_pf * args.uplift
    _emit(
        f"[start] study={args.study} baseline_pf={args.baseline_pf:.3f} "
        f"gate={gate:.3f} (×{args.uplift:.2f}) target_trials={args.target_trials}"
    )

    seen: set[int] = set()
    last_best: float | None = None

    while True:
        try:
            study = optuna.load_study(study_name=args.study, storage=db)
        except Exception as e:
            _emit(f"[ERR] load_study failed: {e}")
            time.sleep(args.poll_seconds)
            continue

        trials = study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed = [t for t in trials if t.state in (
            optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED,
        )]
        running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]

        # Per-trial events
        for t in completed:
            if t.number in seen:
                continue
            seen.add(t.number)
            pf = float(t.value) if t.value is not None else float("nan")
            ml = t.params.get("max_leverage")
            ml_s = f"{ml:.3f}" if ml is not None else "n/a"
            verdict = (
                "PASS_GATE" if pf >= gate
                else "ABOVE_BASELINE" if pf >= args.baseline_pf
                else "BELOW_BASELINE"
            )
            _emit(
                f"[trial-done] #{t.number}  PF={pf:.3f}  max_leverage={ml_s}  "
                f"vs gate {gate:.3f} → {verdict}  ({len(completed)}/{args.target_trials} done)"
            )

        for t in failed:
            if t.number in seen:
                continue
            seen.add(t.number)
            _emit(f"[trial-{t.state.name.lower()}] #{t.number}  state={t.state.name}")

        # Best-so-far tracking
        completed_with_value = [t for t in completed if t.value is not None]
        if completed_with_value:
            best = max(completed_with_value, key=lambda t: t.value)
            best_pf = float(best.value)
            if last_best is None or best_pf > last_best + 1e-6:
                ml = best.params.get("max_leverage")
                ml_s = f"{ml:.3f}" if ml is not None else "n/a"
                gap = (best_pf / args.baseline_pf - 1.0) * 100.0
                _emit(
                    f"[best-improved] trial #{best.number}  PF={best_pf:.3f}  "
                    f"max_leverage={ml_s}  uplift_vs_baseline={gap:+.2f}%"
                )
                last_best = best_pf

        # Termination check
        if len(completed) + len(failed) >= args.target_trials:
            _emit(
                f"[done] all trials accounted for: {len(completed)} complete, "
                f"{len(failed)} failed. Final best PF = "
                f"{(last_best if last_best is not None else float('nan')):.3f}"
            )
            return

        if not running and not (len(completed) + len(failed)):
            _emit("[heartbeat] no running/completed trials yet — workers warming up")
        elif not running:
            _emit(
                f"[heartbeat] no RUNNING trials but {len(completed)}/{args.target_trials} "
                f"complete — possibly all workers shut down (target reached?)"
            )

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
