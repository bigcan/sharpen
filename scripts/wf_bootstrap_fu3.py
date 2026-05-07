#!/usr/bin/env python3
"""WF-aggregated stationary block bootstrap for FU-3 verdict augmentation.

Loads per-fold ens_<rule>_trajectory.parquet and solo_<seed>_trajectory.parquet,
concatenates per-bar returns across folds (preserving pair alignment within each
fold), runs block_bootstrap_pf_mdd from sg1_xauusd_ensemble_eval, applies the
v2.5 _resolve_bootstrap_decision logic, and appends the result to verdict.json
under a `wf_bootstrap` key (does not overwrite legacy G1-G5 gates).

Usage:
    python wf_bootstrap_fu3.py \
        --results_dir results/sg1_xauusd_ensemble_wf \
        --gates configs/sg1_btc_velotrade_ensemble.gates.yaml \
        --ens_rule ens_agreement \
        --solo_seed 456
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/workspace/DeepScalper")
from scripts.sg1_xauusd_ensemble_eval import (  # noqa: E402
    _per_bar_returns,
    block_bootstrap_pf_mdd,
    _resolve_bootstrap_decision,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--ens_rule", required=True,
                    help="Ensemble rule key, e.g. ens_agreement, ens_mean")
    ap.add_argument("--solo_seed", type=int, required=True,
                    help="Solo seed to pair against (best by WF-median PF)")
    ap.add_argument("--n_resamples", type=int, default=10000)
    ap.add_argument("--block_len", type=float, default=None,
                    help="Override sqrt(n) auto block length")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    with open(args.gates, encoding="utf-8") as f:
        gates_cfg = yaml.safe_load(f)
    gates = gates_cfg.get("gates", {})

    ens_returns = []
    solo_returns = []
    fold_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("fold_"))
    fold_n_bars = []
    for fd in fold_dirs:
        ens_p = fd / f"{args.ens_rule}_trajectory.parquet"
        solo_p = fd / f"solo_{args.solo_seed}_trajectory.parquet"
        if not ens_p.exists() or not solo_p.exists():
            print(f"WARN: {fd.name} missing parquets — skipped")
            continue
        ens_df = pd.read_parquet(ens_p)
        solo_df = pd.read_parquet(solo_p)
        e = _per_bar_returns(ens_df)
        s = _per_bar_returns(solo_df)
        n = min(len(e), len(s))
        ens_returns.append(e[:n])
        solo_returns.append(s[:n])
        fold_n_bars.append(int(n))
        print(f"  {fd.name}: n_bars={n}")

    ens_all = np.concatenate(ens_returns) if ens_returns else np.zeros(0)
    solo_all = np.concatenate(solo_returns) if solo_returns else np.zeros(0)
    print(f"\nWF-aggregated: n_bars={ens_all.size}  (per fold: {fold_n_bars})")

    bs = block_bootstrap_pf_mdd(
        ens_all, solo_all,
        n_resamples=args.n_resamples,
        block_len_mean=args.block_len,
    )

    decision = _resolve_bootstrap_decision(bs, gates, legacy_uplift=None)

    augment = {
        "ens_rule": args.ens_rule,
        "solo_seed_paired": args.solo_seed,
        "fold_n_bars": fold_n_bars,
        "bootstrap": bs,
        "v25_decision": decision,
    }

    print("\n========== WF BOOTSTRAP ==========")
    print(json.dumps(augment, indent=2, default=str))

    verdict_path = results_dir / "verdict.json"
    if verdict_path.exists():
        with verdict_path.open() as f:
            verdict = json.load(f)
        verdict["wf_bootstrap"] = augment
        verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
        print(f"\nAppended `wf_bootstrap` to {verdict_path}")
    else:
        out_path = results_dir / "wf_bootstrap.json"
        out_path.write_text(json.dumps(augment, indent=2, default=str))
        print(f"\nWrote {out_path} (verdict.json absent)")


if __name__ == "__main__":
    main()
