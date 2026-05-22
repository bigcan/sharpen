"""Bake `baselines/sg1_btc_decay01_fold_07_solo_456/ensemble_report.json` from
the DECAY-01 8-fold WF solo_456 trajectories.

Pre-swap recal of the live drift baseline for sg1-btc, to support the
v2 -> solo_v1 swap (ensemble rejected by Stage 3 WF
`reject_ensemble_use_best_solo`; seed 456 anchors solo_v1.tar.gz).

Inputs (local, auto_collect_checkpoints.py-pulled per S548 banner):
    results/sg1_btc_ensemble_wf/fold_{00..07}/solo_456_trajectory.parquet

Output:
    baselines/sg1_btc_decay01_fold_07_solo_456/ensemble_report.json

Schema mirrors bake_sg1_btc_v2_drift_baseline.py (same top-level
`ensemble_eval_distribution` block consumed by ActionDriftTracker via
live_engine.py). composition_rule and source change to reflect solo path.
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

from finrl_pro_ds.reporting import compute_eval_distribution  # noqa: E402

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "sg1_btc_ensemble_wf"
DEFAULT_OUTPUT = REPO_ROOT / "baselines" / "sg1_btc_decay01_fold_07_solo_456" / "ensemble_report.json"
DEFAULT_BUNDLE = REPO_ROOT / "bundles" / "sg1_btc_velotrade_decay01" / "solo_v1.tar.gz"
DEADBAND_ABS = 0.25
EQUAL_QUARTILES = {"vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25}
RULE = "solo_456"


def _bar_vol(pv: np.ndarray) -> np.ndarray:
    """Trailing 20-bar realized vol of log returns of portfolio_value.

    Matches bake_sg1_btc_v2_drift_baseline.py + live tracker's
    `vol_estimator_bars: 20` exactly.
    """
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


def _fold_paths(input_root: Path) -> list[Path]:
    candidates = sorted(input_root.glob("fold_*/solo_456_trajectory.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No solo_456_trajectory.parquet under {input_root}/fold_*/. "
            f"Stage 3 WF should have produced 8 folds."
        )
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE,
                   help="solo_v1 bundle path for sha256 in audit metadata")
    args = p.parse_args()

    paths = _fold_paths(args.input_root)
    print(f"Loading {len(paths)} fold trajectories:")
    actions_chunks: list[np.ndarray] = []
    bar_vol_chunks: list[np.ndarray] = []
    fold_bars: list[int] = []
    for path in paths:
        df = pd.read_parquet(path)
        if "action_456" not in df.columns:
            raise KeyError(f"{path} missing 'action_456' column (cols={list(df.columns)})")
        if "portfolio_value" not in df.columns:
            raise KeyError(f"{path} missing 'portfolio_value' column")
        a = df["action_456"].to_numpy(dtype=np.float64)
        bv = _bar_vol(df["portfolio_value"].to_numpy())
        actions_chunks.append(a)
        bar_vol_chunks.append(bv)
        fold_bars.append(len(a))
        print(f"  {path.parent.name}: n_bars={len(a):>6}  "
              f"pv0={df['portfolio_value'].iloc[0]:.2f}  pv_end={df['portfolio_value'].iloc[-1]:.2f}")

    actions = np.concatenate(actions_chunks)
    bar_vol = np.concatenate(bar_vol_chunks)
    print(f"Concatenated: n_bars={len(actions)} (sum of folds={sum(fold_bars)})")

    dist = compute_eval_distribution(
        actions,
        bar_vol=bar_vol,
        regime_quartiles=EQUAL_QUARTILES,
        deadband=DEADBAND_ABS,
        composition_rule=RULE,
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
            "decision": v.get("decision"),
            "overall": v.get("overall"),
            "folds_completed": v.get("folds_completed"),
            "G2_no_regression_folds_meeting": v.get("gates", {}).get("G2_no_regression", {}).get("folds_meeting"),
            "G3_uplift_ratio": v.get("gates", {}).get("G3_uplift", {}).get("uplift"),
            "G5_cv_pf": v.get("gates", {}).get("G5_dd_stability", {}).get("cv_pf"),
        }

    def _maybe_rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    report = {
        "protocol": "v2.2_stage_2_5_ensemble_report__decay01_wf_solo_456_baseline",
        "workstream": "sg1_btc_velotrade_decay01",
        "source": "decay01_wf_8fold_solo_456",
        "window": "decay01_wf_8fold_oos_2025-09-01 -> 2026-04-01 (per-fold 1mo test windows)",
        "chosen_rule": RULE,
        "decision": "SOLO_BEST",
        "decision_source": "stage_3_wf_5_gate (reject_ensemble_use_best_solo)",
        "ensemble_eval_distribution": dist,
        "wf_verdict": wf_verdict_summary,
        "bundle_metadata": {
            "path": _maybe_rel(args.bundle) if args.bundle.is_file() else None,
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
        },
        "bake_metadata": {
            "baked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "baked_by": "scripts/bake_sg1_btc_decay01_drift_baseline.py",
            "input_paths": [_maybe_rel(p) for p in paths],
            "fold_bars": fold_bars,
            "concat_n_bars": int(len(actions)),
            "deadband_abs": DEADBAND_ABS,
            "vol_estimator_bars": 20,
            "rule": RULE,
            "predecessor_baseline": "baselines/sg1_btc_extended_fold_07_ens_mean/ensemble_report.json",
            "predecessor_protocol": "v2.2_stage_2_5_ensemble_report__fu3_fold07_ens_mean_override (FU-2 ens_mean reference window)",
            "rationale": (
                "Stage 3 WF rejected ensemble (G2 1/8, G3 uplift 0.866x); "
                "solo_v1.tar.gz anchors seed 456 fold_07 checkpoint. New baseline "
                "captures seed 456 action distribution across all 8 DECAY-01 WF OOS "
                "test windows (2025-09-01 -> 2026-04-01) for regime breadth, "
                "matching the WF gate's per-fold coverage."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output}")

    eed = dist
    print(f"\nSolo action distribution (seed 456, rule={RULE}):")
    print(f"  deadband_frac (top): {eed.get('deadband_frac'):.4f}")
    print(f"  saturation_frac (top): {eed.get('saturation_frac'):.4f}")
    print(f"  mean: {eed.get('mean'):.4f}  std: {eed.get('std'):.4f}  n: {eed.get('n')}")
    print(f"  regime_cutpoints: {eed.get('regime_cutpoints')}")
    by_vq = eed.get("by_vol_quartile") or {}
    for q in ("q1", "q2", "q3", "q4"):
        b = by_vq.get(q, {})
        print(
            f"  {q}: deadband={b.get('deadband_frac'):.4f}  "
            f"sat={b.get('saturation_frac'):.4f}  n={b.get('n')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
