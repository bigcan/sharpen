"""Bake `baselines/sg1_btc_extended_fold_07/ensemble_report.json` from the
FU-3 8-fold WF ens_agreement trajectories.

Pre-swap recal of the live drift baseline for sg1-btc, to support the v1->v2
swap (rule change ens_mean -> ens_agreement, see project_sg1_btc_fu3_wf_promote).

Inputs (rsync from gpuhub-1 before running):
    results/sg1_btc_velotrade_extended_ensemble/wf_fold_{00..07}/ens_agreement_trajectory.parquet

Output:
    baselines/sg1_btc_extended_fold_07/ensemble_report.json

Schema matches sg1_xauusd_fold_07/ensemble_report.json (top-level
`ensemble_eval_distribution` block consumed by ActionDriftTracker via
live_engine.py:85-86 and AgreementDecayTracker via live_engine.py:186-188).
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

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "sg1_btc_velotrade_extended_ensemble"
DEFAULT_OUTPUT = REPO_ROOT / "baselines" / "sg1_btc_extended_fold_07" / "ensemble_report.json"
DEADBAND_ABS = 0.25  # matches all sg1-btc configs' env.deadband_threshold
EQUAL_QUARTILES = {"vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25}
RULE = "ens_agreement"


def _bar_vol(pv: np.ndarray) -> np.ndarray:
    """Trailing 20-bar realized vol of log returns of portfolio_value.

    Mirrors `_bar_vol` in scripts/sg1_xauusd_ensemble_eval.py so the baseline
    bucketing matches the live tracker's bucketing path (live also uses
    20-bar vol via `vol_estimator_bars: 20`).
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
    candidates = sorted(input_root.glob("wf_fold_*/ens_agreement_trajectory.parquet"))
    if not candidates:
        candidates = sorted(input_root.glob("fold_*/ens_agreement_trajectory.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No ens_agreement_trajectory.parquet under {input_root}/wf_fold_*/ "
            f"or {input_root}/fold_*/. Rsync FU-3 WF outputs from gpuhub-1 first."
        )
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_INPUT_ROOT / "ensemble_v2.tar.gz",
        help="ensemble_v2 bundle path for sha256 in audit metadata",
    )
    args = p.parse_args()

    paths = _fold_paths(args.input_root)
    print(f"Loading {len(paths)} fold trajectories:")
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
        print(f"  {path.parent.name}: n_bars={len(a):>6}  pv0={df['portfolio_value'].iloc[0]:.2f}  pv_end={df['portfolio_value'].iloc[-1]:.2f}")

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

    fu3_verdict_path = args.input_root / "verdict_fu3_wf.json"
    fu3_verdict_summary = None
    if fu3_verdict_path.is_file():
        v = json.loads(fu3_verdict_path.read_text())
        fu3_verdict_summary = {
            "decision": v.get("decision"),
            "v25_decision": v.get("wf_bootstrap", {}).get("v25_decision", {}).get("decision"),
            "p_pf_ens_better": v.get("wf_bootstrap", {}).get("bootstrap", {}).get("p_pf_ens_better"),
            "ens_pf_q50": v.get("wf_bootstrap", {}).get("bootstrap", {}).get("ens_pf_quantiles", {}).get("q50"),
        }

    def _maybe_rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    report = {
        "protocol": "v2.2_stage_2_5_ensemble_report__fu3_wf_baseline",
        "workstream": "sg1_btc_velotrade_extended",
        "source": "fu3_wf_fold_07_ens_agreement",
        "window": "fu3_wf_fold_07_oos_2026-03-01 -> 2026-04-01",
        "chosen_rule": RULE,
        "decision": "PROMOTE",
        "decision_source": "v2.5_bootstrap",
        "ensemble_eval_distribution": dist,
        "fu3_verdict": fu3_verdict_summary,
        "bundle_metadata": {
            "path": _maybe_rel(args.bundle) if args.bundle.is_file() else None,
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
        },
        "bake_metadata": {
            "baked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "baked_by": "scripts/bake_sg1_btc_v2_drift_baseline.py",
            "input_paths": [_maybe_rel(p) for p in paths],
            "fold_bars": fold_bars,
            "concat_n_bars": int(len(actions)),
            "deadband_abs": DEADBAND_ABS,
            "vol_estimator_bars": 20,
            "rule": RULE,
            "predecessor_baseline": "baselines/sg1_btc_l1/seed_report.json",
            "predecessor_protocol": "v2.2_stage_2_seed_report (per-seed; seed 42 fallback)",
            "rationale": (
                "Pre-swap recal for ensemble_v2 (ens_agreement). Live engine "
                "v1 ran ens_mean against seed-42 fallback baseline; v2 ens_agreement "
                "needs ensemble_eval_distribution block (live_engine.py:85-86, :186-188). "
                "Baseline aggregated across all 8 FU-3 WF OOS test windows for breadth "
                "(matches WF bootstrap n=57028) rather than a single fold to be "
                "robust against single-window regime quirks."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output}")

    # Summary print for operator review
    eed = dist
    print("\nEnsemble eval distribution (ens_agreement):")
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
