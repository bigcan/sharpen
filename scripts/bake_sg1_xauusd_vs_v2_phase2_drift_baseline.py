"""Bake `baselines/sg1_xauusd_vs_v2_phase2_fold_07/ensemble_report.json`
from the SG-1-XAUUSD Volume Study v2 Phase 2 Stage 3 WF ens_mean trajectories.

Pre-swap recal of the live drift baseline for sg1-xauusd, to support the
v1 (S488 OANDA WF, seeds [42, 2025, 3141]) -> v2 (Phase 2, seeds [2025, 42, 9999])
ensemble swap.

Inputs (Stage 3 WF outputs from `scripts/sg1_xauusd_ensemble_eval.py --wf_config`):
    results/sg1_xauusd_vs_v2_phase2_wf_ensemble/fold_{00..07}/ens_mean_trajectory.parquet

Output:
    baselines/sg1_xauusd_vs_v2_phase2_fold_07/ensemble_report.json

Schema matches `baselines/sg1_xauusd_wf_fold_07_ens_mean/ensemble_report.json`
(top-level `ensemble_eval_distribution` block consumed by ActionDriftTracker via
live_engine.py:85-86 and AgreementDecayTracker via live_engine.py:186-188).

Constants vs SG-1-BTC analog (scripts/bake_sg1_btc_v2_drift_baseline.py):
  - DEADBAND_ABS: 0.35 (matches env.deadband_threshold across all SG-1-XAUUSD configs)
  - RULE: ens_mean (chosen by Stage 2.5-R val_argmax_pf; revisit if WF eval flips)
  - input root: results/sg1_xauusd_vs_v2_phase2_wf_ensemble (matches WF eval --output_dir)
  - bundle path: results/sg1_xauusd_volume_study_v2_phase2_stage25r/ensemble_v2.tar.gz
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

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "sg1_xauusd_vs_v2_phase2_wf_ensemble"
DEFAULT_OUTPUT = REPO_ROOT / "baselines" / "sg1_xauusd_vs_v2_phase2_fold_07" / "ensemble_report.json"
DEFAULT_BUNDLE = REPO_ROOT / "results" / "sg1_xauusd_volume_study_v2_phase2_stage25r" / "ensemble_v2.tar.gz"
DEADBAND_ABS = 0.35  # matches all sg1-xauusd configs' env.deadband_threshold
EQUAL_QUARTILES = {"vol_q1": 0.25, "vol_q2": 0.25, "vol_q3": 0.25, "vol_q4": 0.25}
RULE = "ens_mean"


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


def _fold_paths(input_root: Path, rule: str) -> list[Path]:
    pattern = f"fold_*/{rule}_trajectory.parquet"
    candidates = sorted(input_root.glob(pattern))
    if not candidates:
        # Fall back to alt naming used by some older WF runs.
        candidates = sorted(input_root.glob(f"wf_fold_*/{rule}_trajectory.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No {rule}_trajectory.parquet under {input_root}/fold_*/ "
            f"or {input_root}/wf_fold_*/. Run "
            f"`python scripts/sg1_xauusd_ensemble_eval.py "
            f"--wf_config configs/sg1_xauusd_volume_study_v2_phase2_wf_multiseed.yaml "
            f"--gates_file configs/sg1_xauusd_volume_study_v2_phase2_ensemble.gates.yaml "
            f"--output_dir {input_root.relative_to(REPO_ROOT)}` first."
        )
    return candidates


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--rule", default=RULE,
                   help=f"Aggregation rule trajectory to read (default {RULE}). "
                        f"Set to the WF eval's chosen_rule if different.")
    p.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="ensemble_v2 bundle path for sha256 in audit metadata",
    )
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
            "gates_pass": {k: g.get("status") for k, g in (v.get("gates") or {}).items()},
        }

    def _maybe_rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    report = {
        "protocol": "v2.2_stage_2_5_ensemble_report__sg1_xauusd_vs_v2_phase2_wf_baseline",
        "workstream": "sg1_xauusd_volume_study_v2_phase2",
        "source": f"sg1_xauusd_vs_v2_phase2_wf_fold_07_{args.rule}",
        "window": "vs_v2_phase2_wf_fold_07_oos_2026-03-01 -> 2026-04-01",
        "chosen_rule": args.rule,
        "decision": "PROMOTE",
        "decision_source": "v2.5_bootstrap__stage_2_5_r",
        "ensemble_eval_distribution": dist,
        "wf_verdict": wf_verdict_summary,
        "bundle_metadata": {
            "path": _maybe_rel(args.bundle) if args.bundle.is_file() else None,
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
        },
        "bake_metadata": {
            "baked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "baked_by": "scripts/bake_sg1_xauusd_vs_v2_phase2_drift_baseline.py",
            "input_paths": [_maybe_rel(p) for p in paths],
            "fold_bars": fold_bars,
            "concat_n_bars": int(len(actions)),
            "deadband_abs": DEADBAND_ABS,
            "vol_estimator_bars": 20,
            "rule": args.rule,
            "predecessor_baseline": "baselines/sg1_xauusd_wf_fold_07_ens_mean/ensemble_report.json",
            "predecessor_protocol": "v2.2_stage_2_5_ensemble_report__sg1xauusd_fold07_ens_mean_override (S488 OANDA WF)",
            "rationale": (
                "Pre-swap recal for ensemble_v2 (Phase 2 Volume Study v2 retrain). "
                "Live engine v1 ran ens_mean against the S488 OANDA WF fold-07 baseline "
                "(seeds [42, 2025, 3141]); v2 ens_mean uses Phase 2 retrain checkpoints "
                "(seeds [2025, 42, 9999]) at 4M steps. Baseline aggregated across all "
                "8 Phase 2 WF OOS test windows for breadth (matches WF bootstrap "
                "denominator) rather than a single fold to be robust against "
                "single-window regime quirks."
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
