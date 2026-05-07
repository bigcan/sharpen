#!/usr/bin/env python3
"""One-shot WF gate re-evaluation for FU-3 (SG-1-BTC extended-window 8-fold WF).

The aggregator (scripts/sg1_xauusd_ensemble_eval.py::run_wf_ensemble) saved all
per-fold *_metrics.json + summary.csv before crashing in _evaluate_gates on a
hardcoded g4_ftmo_compliance lookup. This script reloads those metrics from
disk, calls the patched _evaluate_gates, and writes verdict.json — avoiding
a 40-min re-backtest of identical trajectories.

Usage:
    python reverdict_fu3.py \
        --results_dir /workspace/DeepScalper/results/sg1_xauusd_ensemble_wf \
        --gates /workspace/DeepScalper/configs/sg1_btc_velotrade_ensemble.gates.yaml \
        --seeds 456 3141 123
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/workspace/DeepScalper")
from scripts.sg1_xauusd_ensemble_eval import _evaluate_gates  # uses patched G4


RULES = ("solo_456", "solo_3141", "solo_123",
         "ens_mean", "ens_median", "ens_agreement", "ens_pf_weighted")


def load_per_fold_metrics(results_dir: Path, seeds):
    rule_names = [f"solo_{s}" for s in seeds] + [
        "ens_mean", "ens_median", "ens_agreement", "ens_pf_weighted",
    ]
    fold_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("fold_"))
    per_fold = []
    status = []
    for fd in fold_dirs:
        fm = {}
        ok = True
        for rn in rule_names:
            mp = fd / f"{rn}_metrics.json"
            if not mp.exists():
                ok = False
                break
            with mp.open() as f:
                fm[rn] = json.load(f)
        per_fold.append(fm if ok else {})
        status.append("ok" if ok else "skipped:missing_metrics")
    return per_fold, status, fold_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None,
                    help="verdict.json output path (default: <results_dir>/verdict.json)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    with open(args.gates, encoding="utf-8") as f:
        gates_cfg = yaml.safe_load(f)

    per_fold, status, fold_dirs = load_per_fold_metrics(results_dir, args.seeds)
    print(f"Loaded {sum(1 for s in status if s == 'ok')}/{len(status)} folds from {results_dir}")
    for fd, st in zip(fold_dirs, status):
        print(f"  {fd.name}: {st}")

    verdict = _evaluate_gates(per_fold, status, gates_cfg, list(args.seeds))

    out_path = Path(args.out) if args.out else results_dir / "verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print("\n========== WF ENSEMBLE VERDICT ==========")
    print(json.dumps(verdict["gates"], indent=2, default=str))
    print(f"\nOVERALL: {verdict['overall']}")
    print(f"DECISION: {verdict.get('decision')}")


if __name__ == "__main__":
    main()
