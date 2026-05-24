"""Bake `baselines/sg1_eurusd_hg_ensemble_v1/ensemble_report.json` from the
SG-1 EURUSD Hyper Growth 8-fold WF ensemble trajectories.

Pre-deploy build of the live drift baseline for sg1-eurusd, supporting the
ensemble_v1 (ens_pf_weighted, seeds [2025, 3141, 9999]) Bybit demo SOAK
deploy (S550). Stage 3 WF G3 legacy point-estimate FAILed at 1.081 but
Protocol v2.5 bootstrap PROMOTEd (P(PF)=1.0000 / P(vs_best_solo)=0.9994).

Inputs (already collected to local repo from gpuhub-1+2 WF):
    results/sg1_eurusd_hg_ensemble_wf/fold_{00..07}/ens_pf_weighted_trajectory.parquet

Output:
    baselines/sg1_eurusd_hg_ensemble_v1/ensemble_report.json

Schema mirrors bake_sg1_xauusd_vs_v2_phase2_drift_baseline.py /
bake_sg1_btc_v2_drift_baseline.py — same top-level `ensemble_eval_distribution`
block consumed by ActionDriftTracker via live_engine.py.
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

from finrl_pro_ds.reporting import (  # noqa: E402
    compute_challenge_target_hit_rates,
    compute_eval_distribution,
)

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "sg1_eurusd_hg_ensemble_wf"
DEFAULT_OUTPUT = REPO_ROOT / "baselines" / "sg1_eurusd_hg_ensemble_v1" / "ensemble_report.json"
DEFAULT_BUNDLE = REPO_ROOT / "bundles" / "sg1_eurusd_hg_ensemble_v1" / "ensemble_v1.tar.gz"
DEADBAND_ABS = 0.25
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


def _fold_paths(input_root: Path) -> list[Path]:
    candidates = sorted(input_root.glob("fold_*/ens_pf_weighted_trajectory.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No ens_pf_weighted_trajectory.parquet under {input_root}/fold_*/. "
            f"Stage 3 WF should have produced 8 folds."
        )
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE,
                   help="ensemble_v1 bundle path for sha256 in audit metadata")
    args = p.parse_args()

    paths = _fold_paths(args.input_root)
    print(f"Loading {len(paths)} fold trajectories:")
    actions_chunks: list[np.ndarray] = []
    bar_vol_chunks: list[np.ndarray] = []
    pv_chunks: list[np.ndarray] = []
    fold_bars: list[int] = []
    for path in paths:
        df = pd.read_parquet(path)
        if "action_agg" not in df.columns:
            raise KeyError(f"{path} missing 'action_agg' column (cols={list(df.columns)})")
        if "portfolio_value" not in df.columns:
            raise KeyError(f"{path} missing 'portfolio_value' column")
        a = df["action_agg"].to_numpy(dtype=np.float64)
        pv = df["portfolio_value"].to_numpy(dtype=np.float64)
        bv = _bar_vol(pv)
        actions_chunks.append(a)
        bar_vol_chunks.append(bv)
        pv_chunks.append(pv)
        fold_bars.append(len(a))
        print(f"  {path.parent.name}: n_bars={len(a):>6}  "
              f"pv0={pv[0]:.2f}  pv_end={pv[-1]:.2f}")

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

    # Protocol v2.4 amendment (S495 OQ#5, 2026-04-24): per-fold hit-rate of
    # the HG 8% profit target. Each fold's ~1-month OOS window counts as one
    # challenge attempt. Reported at ensemble level (aggregated PV) and per
    # seed (each constituent's solo PV). Validator
    # check_drift_baseline_manifest_schema requires both blocks at paper-deploy.
    def _hit_rate_block(pv_per_fold: list[np.ndarray]) -> dict:
        if not pv_per_fold:
            return {"step1": {"target_pct": 0.08, "n_windows": 0, "n_hits": 0, "hit_rate": 0.0}}
        hits = sum(1 for pv in pv_per_fold if (pv.max() - pv[0]) / pv[0] >= 0.08)
        n_windows = len(pv_per_fold)
        return {
            "step1": {
                "target_pct": 0.08,
                "window_bars": "per_fold",
                "n_windows": n_windows,
                "n_hits": hits,
                "hit_rate": float(hits) / n_windows,
                "max_cum_return_per_fold": [
                    float((pv.max() - pv[0]) / pv[0]) for pv in pv_per_fold
                ],
            },
        }

    ensemble_hit_rate = _hit_rate_block(pv_chunks)

    seed_hit_rates: dict[str, dict] = {}
    for seed in (2025, 3141, 9999):
        seed_pv_per_fold: list[np.ndarray] = []
        for ens_path in paths:
            solo_path = ens_path.parent / f"solo_{seed}_trajectory.parquet"
            if not solo_path.is_file():
                continue
            sdf = pd.read_parquet(solo_path)
            if "portfolio_value" not in sdf.columns:
                continue
            seed_pv_per_fold.append(sdf["portfolio_value"].to_numpy(dtype=np.float64))
        seed_hit_rates[str(seed)] = _hit_rate_block(seed_pv_per_fold)

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
        "protocol": "v2.2_stage_2_5_ensemble_report__sg1_eurusd_hg_wf_baseline",
        "workstream": "sg1_eurusd_hyper_growth",
        "source": "sg1_eurusd_hg_wf_8fold_ens_pf_weighted",
        "window": "hg_wf_8fold_oos (per-fold ~3wk test windows, ~13.5K bars total)",
        "chosen_rule": RULE,
        "decision": "PROMOTE",
        "decision_source": "v2.5_bootstrap__manual (P(PF)=1.0000, P(vs_best_solo)=0.9994; legacy G3 1.081 override)",
        "ensemble_eval_distribution": dist,
        # Protocol v2.4 amendment (S495 OQ#5): deploy-readiness signal.
        # Gate: ensemble step1 hit_rate >= 0.5 to paper-deploy.
        "ensemble_challenge_target_hit_rate": ensemble_hit_rate,
        "challenge_target_hit_rate_by_seed": seed_hit_rates,
        "wf_verdict": wf_verdict_summary,
        "bundle_metadata": {
            "path": _maybe_rel(args.bundle) if args.bundle.is_file() else None,
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
        },
        "bake_metadata": {
            "baked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "baked_by": "scripts/bake_sg1_eurusd_hg_ensemble_drift_baseline.py",
            "input_paths": [_maybe_rel(p) for p in paths],
            "fold_bars": fold_bars,
            "concat_n_bars": int(len(actions)),
            "deadband_abs": DEADBAND_ABS,
            "vol_estimator_bars": 20,
            "rule": RULE,
            "rationale": (
                "Ensemble v1 (seeds [2025, 3141, 9999], aggregation ens_pf_weighted) "
                "baseline captures the ensemble's action distribution across all 8 "
                "Hyper Growth WF OOS test windows for regime breadth. Stage 3 WF "
                "G3 point-estimate FAILed (1.081 < 1.10) but Protocol v2.5 "
                "block-bootstrap PROMOTEd. Live drift monitor compares Bybit demo "
                "trading-time action histogram against this baseline."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.output}")

    eed = dist
    print(f"\nEnsemble action distribution (rule={RULE}):")
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
