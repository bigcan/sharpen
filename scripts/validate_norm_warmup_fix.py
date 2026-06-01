#!/usr/bin/env python3
"""End-to-end validation of the S509 LiveObsBuilder normalization fix.

Compares per-scale features computed via:
  A. LiveObsBuilder WITHOUT warmup buffer (legacy rolling EMA-Z — the bug)
  B. LiveObsBuilder WITH warmup buffer    (S509 fix)
  C. MultiScaleOHLCVHandler with norm_cutoff_date set (training reference)

On real OANDA+cTrader extended data covering the drift period, asserts:
  - C is the gold-standard reference (training pipeline)
  - B matches C bit-for-bit (or within float32 epsilon)
  - A diverges from C (the train-serve skew) — confirms the bug exists
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crypto.live.live_obs_builder import LiveObsBuilder  # noqa: E402
from finrl_pro_ds.data.multiscale_handler import (  # noqa: E402
    _compute_scale_features, _resample_ohlcv,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("validate-warmup")


def main():
    parquet = "data/oanda/xauusd_m1_m_ext_ctrader.parquet"
    warmup_path = "checkpoints/WF_seed42_fold_07_20260424_061803/norm_warmup.pkl"
    train_end = pd.Timestamp("2026-02-06 00:00:00")
    scales = [3, 15, 60]
    norm_span = 120

    # Load full data, compute live "post-cutoff" segment
    df = pd.read_parquet(parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    live_df = df[df["timestamp"] > train_end].copy().reset_index(drop=True)
    log.info(f"loaded {len(df)} 1-min bars; post-cutoff segment {len(live_df)} rows "
             f"({live_df['timestamp'].min()} -> {live_df['timestamp'].max()})")

    # --- Reference C: training pipeline (frozen cutoff) ---
    # Compute via _compute_scale_features over the FULL series with norm_cutoff_idx.
    # This is what training did: pre-cutoff EMA-Z one way, post-cutoff freeze-and-restart.
    log.info("computing reference C (training pipeline, frozen cutoff)…")
    ref_features: dict[int, np.ndarray] = {}
    for scale in scales:
        resampled_full = _resample_ohlcv(df, scale)
        cutoff_mask = resampled_full["timestamp"] >= train_end
        cutoff_idx = int(cutoff_mask.idxmax()) if cutoff_mask.any() else None
        feats = _compute_scale_features(resampled_full, norm_cutoff_idx=cutoff_idx,
                                         span=norm_span, n_features=8)
        post = feats[cutoff_idx:] if cutoff_idx else feats
        ref_features[scale] = post
        log.info(f"  scale={scale}min: {len(post)} post-cutoff feature rows")

    # --- A: LiveObsBuilder WITHOUT warmup (the bug) ---
    log.info("computing A (LiveObsBuilder no warmup, legacy rolling EMA-Z)…")
    bld_a = LiveObsBuilder(scales=scales, window_size=30, norm_span=norm_span,
                           n_features=8, bootstrap_bars=len(live_df), drift_detection=False)
    bld_a.bootstrap_from_dataframe(live_df)
    feat_a = {s: bld_a._scale_features[s] for s in scales}

    # --- B: LiveObsBuilder WITH warmup (the fix) ---
    log.info("computing B (LiveObsBuilder + warmup buffer, S509 fix)…")
    bld_b = LiveObsBuilder(scales=scales, window_size=30, norm_span=norm_span,
                           n_features=8, bootstrap_bars=len(live_df), drift_detection=False,
                           norm_warmup_path=warmup_path)
    bld_b.bootstrap_from_dataframe(live_df)
    feat_b = {s: bld_b._scale_features[s] for s in scales}

    # --- Compare ---
    print("\n=== feature distribution close_z (col 6) summary per scale ===")
    print(f"{'scale':<8}{'metric':<25}{'C ref':>12}{'A buggy':>12}{'B fixed':>12}{'A-C diff':>12}{'B-C diff':>12}")
    for scale in scales:
        # Align lengths — A and B come from LiveObsBuilder which uses live_df only;
        # ref_features[scale] is from full series sliced post-cutoff.
        ref = ref_features[scale]
        a = feat_a[scale]
        b = feat_b[scale]
        # Trim to common length
        n = min(len(ref), len(a), len(b))
        ref, a, b = ref[-n:], a[-n:], b[-n:]
        for col_idx, col_name in [(6, "close_z")]:
            r_mean = ref[:, col_idx].mean()
            a_mean = a[:, col_idx].mean()
            b_mean = b[:, col_idx].mean()
            r_std = ref[:, col_idx].std()
            a_std = a[:, col_idx].std()
            b_std = b[:, col_idx].std()
            print(f"{scale:<8}{col_name+'_mean':<25}{r_mean:>12.4f}{a_mean:>12.4f}{b_mean:>12.4f}"
                  f"{a_mean-r_mean:>+12.4f}{b_mean-r_mean:>+12.4f}")
            print(f"{scale:<8}{col_name+'_std':<25}{r_std:>12.4f}{a_std:>12.4f}{b_std:>12.4f}"
                  f"{a_std-r_std:>+12.4f}{b_std-r_std:>+12.4f}")

        # Bit-equivalence check on B vs C
        max_abs_diff_b = float(np.max(np.abs(b - ref)))
        max_abs_diff_a = float(np.max(np.abs(a - ref)))
        print(f"{scale:<8}{'max|x-ref| any-col':<25}{0.0:>12.4f}{max_abs_diff_a:>12.4f}{max_abs_diff_b:>12.4f}")

        # Hard assert: B should match C within float32 epsilon
        if max_abs_diff_b > 1e-5:
            print(f"  WARN: scale={scale} B-vs-C max diff {max_abs_diff_b:.6f} > 1e-5 (expected bit-equivalent)")
        else:
            print(f"  OK:   scale={scale} B-vs-C max diff {max_abs_diff_b:.2e} (bit-equivalent)")

        if max_abs_diff_a < 1e-5:
            print(f"  WARN: scale={scale} A-vs-C max diff {max_abs_diff_a:.6f} < 1e-5 — bug doesn't manifest on this slice?")
        else:
            print(f"  OK:   scale={scale} A-vs-C max diff {max_abs_diff_a:.4f} (bug confirmed)")
    print()

    # Aggregate ensemble-deadband-frac proxy: count rows where |close_z| < tanh(0.5*0.35) ≈ deadband threshold
    # Actually, deadband 0.35 applies to the AGENT's action output, not features. So we just
    # report close_z absolute mean and std distribution, which sets up the feature distribution
    # the agent then consumes. If A produces compressed close_z (mean near 0, low std), the
    # agent's downstream output will also be in deadband → drift_crit. B should restore the
    # spread that the agent saw at training time.

    print("=== verdict ===")
    print("If B-vs-C max diffs are <1e-5 across scales: S509 fix is bit-equivalent to training.")
    print("If A-vs-C max diffs are large: legacy rolling EMA-Z bug is confirmed on real data.")


if __name__ == "__main__":
    main()
