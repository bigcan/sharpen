"""Bake `baselines/gmgp1_xauusd_extended_wf_fold_07/ensemble_report.json`
from the GMGP1-XAUUSD extended-window Stage 3 WF ens_pf_weighted trajectories.

Pre-swap recal of the live drift baseline for gmgp1-xauusd, to support the
solo (S495 OANDA WF fold-7 seed-42) → ensemble v1 (extended WF, seeds
[789, 2025, 1024], ens_pf_weighted) graduation.

Inputs (Stage 3 WF outputs from scripts/sg1_xauusd_ensemble_eval.py --wf_config):
    results/gmgp1_xauusd_extended_wf_ensemble/fold_{00..07}/ens_pf_weighted_trajectory.parquet

Output:
    baselines/gmgp1_xauusd_extended_wf_fold_07/ensemble_report.json

Schema matches baselines/sg1_xauusd_vs_v2_phase2_fold_07/ensemble_report.json
(top-level `ensemble_eval_distribution` block consumed by ActionDriftTracker
via live_engine.py).

Constants:
  DEADBAND_ABS: 0.35 (matches env.deadband_threshold in WF config)
  RULE: ens_pf_weighted (chosen rule, bootstrap PRIMARY PROMOTE)
  bundle path: results/gmgp1_xauusd_extended_wf_ensemble/ensemble_v1.tar.gz
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sharpen.reporting import compute_eval_distribution  # noqa: E402

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "gmgp1_xauusd_extended_wf_ensemble"
DEFAULT_OUTPUT = REPO_ROOT / "baselines" / "gmgp1_xauusd_extended_wf_fold_07" / "ensemble_report.json"
DEFAULT_BUNDLE = REPO_ROOT / "results" / "gmgp1_xauusd_extended_wf_ensemble" / "ensemble_v1.tar.gz"
DEADBAND_ABS = 0.35
EQUAL_QUARTILES = {"vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25}
RULE = "ens_pf_weighted"


def _bar_vol(pv: np.ndarray) -> np.ndarray:
    pv = pv.astype(np.float64, copy=False)
    if pv.size < 2:
        return np.zeros_like(pv)
    log_ret = np.zeros(pv.size)
    np.log(pv[1:] / np.clip(pv[:-1], 1e-9, None), out=log_ret[1:])
    vol = np.zeros(pv.size)
    for i in range(1, pv.size):
        lo = max(0, i - 19)
        vol[i] = float(np.std(log_ret[lo:i + 1], ddof=0))
    return vol


def _fold_paths(input_root: Path, rule: str) -> list[Path]:
    pattern = f"fold_*/{rule}_trajectory.parquet"
    candidates = sorted(input_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {rule}_trajectory.parquet under {input_root}/fold_*/. "
            f"Run scripts/sg1_xauusd_ensemble_eval.py with --wf_config "
            f"configs/gmgp1_xauusd_wf_extended.yaml first."
        )
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--rule", default=RULE,
                   help=f"Aggregation rule trajectory to read (default {RULE}).")
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE,
                   help="ensemble_v1 bundle path for sha256 in audit metadata")
    args = p.parse_args()

    paths = _fold_paths(args.input_root, args.rule)
    print(f"Loading {len(paths)} fold trajectories (rule={args.rule}):")
    actions_chunks: list[np.ndarray] = []
    bar_vol_chunks: list[np.ndarray] = []
    fold_bars: list[int] = []
    for path in paths:
        df = pd.read_parquet(path)
        if "action_agg" not in df.columns:
            raise KeyError(f"{path} missing 'action_agg' column (cols={list(df.columns)})")
        if "portfolio_value" not in df.columns:
            raise KeyError(f"{path} missing 'portfolio_value' column")
        a = df["action_agg"].to_numpy(dtype=np.float64)
        bv = _bar_vol(df["portfolio_value"].to_numpy())
        actions_chunks.append(a)
        bar_vol_chunks.append(bv)
        fold_bars.append(len(a))
        print(f"  {path.parent.name}: n_bars={len(a):>6}  "
              f"pv0={df['portfolio_value'].iloc[0]:.2f}  "
              f"pv_end={df['portfolio_value'].iloc[-1]:.2f}")

    actions = np.concatenate(actions_chunks)
    bar_vol = np.concatenate(bar_vol_chunks)
    print(f"Concatenated: n_bars={len(actions)} (sum of folds={sum(fold_bars)})")

    dist = compute_eval_distribution(
        actions,
        bar_vol=bar_vol,
        regime_quartiles=EQUAL_QUARTILES,
        deadband=DEADBAND_ABS,
        composition_rule=args.rule,
    )

    bundle_sha = None
    bundle_size = None
    if args.bundle.is_file():
        h = hashlib.sha256()
        with open(args.bundle, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        bundle_sha = h.hexdigest()
        bundle_size = args.bundle.stat().st_size

    wf_verdict_path = args.input_root / "verdict.json"
    wf_verdict_summary = None
    if wf_verdict_path.is_file():
        v = json.loads(wf_verdict_path.read_text())
        wf_verdict_summary = {
            "overall": v.get("overall"),
            "decision": v.get("decision"),
            "gates_pass": {k: g.get("pass") for k, g in (v.get("gates") or {}).items()},
            "wf_bootstrap_decision": (v.get("wf_bootstrap", {}) or {}).get("v25_decision"),
        }

    def _maybe_rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    report = {
        "protocol": "v2.2_stage_2_5_ensemble_report__gmgp1_xauusd_extended_wf_baseline",
        "workstream": "gmgp1_xauusd_extended",
        "source": f"gmgp1_xauusd_extended_wf_fold_07_{args.rule}",
        "window": "extended_wf_fold_07_oos_2026-03-01 -> 2026-04-01",
        "chosen_rule": args.rule,
        "decision": "PROMOTE",
        "decision_source": "v2.5_bootstrap__wf_aggregated",
        "ensemble_eval_distribution": dist,
        "wf_verdict": wf_verdict_summary,
        "bundle_metadata": {
            "path": _maybe_rel(args.bundle) if args.bundle.is_file() else None,
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
        },
        "bake_metadata": {
            "baked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "baked_by": "scripts/bake_gmgp1_xauusd_ext_drift_baseline.py",
            "input_paths": [_maybe_rel(p) for p in paths],
            "fold_bars": fold_bars,
            "concat_n_bars": int(len(actions)),
            "deadband_abs": DEADBAND_ABS,
            "vol_estimator_bars": 20,
            "rule": args.rule,
            "predecessor_baseline": "baselines/gmgp1_xauusd_oanda_wf_fold_07/seed_report.json",
            "predecessor_protocol": "v2.2_solo_seed_42_fold_07 (S495 OANDA WF)",
            "rationale": (
                "Pre-swap recal for ensemble_v1 (Tier B2 extended-window retrain). "
                "Live engine ran solo seed-42 against the S495 OANDA WF fold-07 "
                "seed_report.json baseline; v1 ens_pf_weighted uses extended retrain "
                "checkpoints (seeds [789, 2025, 1024]) at 1.1M steps per fold. "
                "Baseline aggregated across all 8 extended WF OOS test windows for "
                "breadth (matches WF bootstrap denominator) rather than a single fold."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output}")

    eed = dist
    print(f"\nEnsemble eval distribution ({args.rule}):")
    print(f"  deadband_frac (top): {eed.get('deadband_frac'):.4f}")
    print(f"  saturation_frac (top): {eed.get('saturation_frac'):.4f}")
    print(f"  mean: {eed.get('mean'):.4f}  std: {eed.get('std'):.4f}  n: {eed.get('n')}")
    print(f"  regime_cutpoints: {eed.get('regime_cutpoints')}")
    by_vq = eed.get("by_vol_quartile") or {}
    for q in ("q1", "q2", "q3", "q4"):
        b = by_vq.get(q, {})
        if not b:
            continue
        print(
            f"  {q}: deadband={b.get('deadband_frac'):.4f}  "
            f"sat={b.get('saturation_frac'):.4f}  n={b.get('n')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
