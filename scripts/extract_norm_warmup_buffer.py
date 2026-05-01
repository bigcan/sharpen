#!/usr/bin/env python3
"""Extract per-fold normalization warmup buffer (S509 fix).

For each (seed, fold) WF checkpoint directory, save 200 pre-cutoff OHLCV bars
per scale to `norm_warmup.pkl` alongside `checkpoint_final.pth`.

The 200 bars per scale are the last 200 bars (resampled at that scale) ending
at the fold's `train_end_date`. They reproduce the warmup buffer that
training's `_compute_scale_features(arr, norm_cutoff_idx=train_end_idx)`
implicitly used for post-cutoff features. Live engine consumes them via
`compute_features_with_warmup()` to get bit-equivalent EMA-Z behavior.

Usage:
    python scripts/extract_norm_warmup_buffer.py \\
        --config configs/sg1_xauusd_ftmo_rehpo_wf_multiseed.yaml \\
        --seeds 42 2025 3141 --folds 7

Writes one pickle per checkpoint dir with shape:
    {scale_minutes: pd.DataFrame[200 rows x [timestamp,open,high,low,close,volume]]}
"""
from __future__ import annotations
import argparse
import glob
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data.multiscale_handler import _resample_ohlcv  # noqa: E402
from finrl_pro_ds.data.splitter import RollingWindowSplitter  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract-warmup")

WARMUP_BUFFER = 200  # must match multiscale_handler.WARMUP_BUFFER


def fold_train_end(config: dict, fold_idx: int) -> pd.Timestamp:
    data_cfg = config["data"]
    splitter = RollingWindowSplitter(
        train_months=int(data_cfg.get("train_months", 6)),
        val_months=int(data_cfg.get("val_months", 1)),
        test_months=int(data_cfg.get("test_months", 1)),
        step_months=int(data_cfg.get("step_months", 1)),
        buffer_days=int(data_cfg.get("buffer_days", 0)),
    )
    folds = splitter.split(data_cfg["start_date"], data_cfg["end_date"])
    if fold_idx >= len(folds):
        raise ValueError(f"fold_idx={fold_idx} out of range (n_folds={len(folds)})")
    return pd.Timestamp(folds[fold_idx]["train"].end)


def build_warmup_buffer(parquet_path: Path, scales: list[int],
                        train_end: pd.Timestamp) -> dict:
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    log.info(f"loaded {len(df)} 1-min bars span {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # FIX MATH-N03 carry-forward: training reads `arr[norm_cutoff_idx - 200 :
    # norm_cutoff_idx]` per scale. We replicate that exactly.
    out = {}
    for scale in scales:
        resampled = _resample_ohlcv(df, scale)
        # Bars at or before train_end (training-time pre-cutoff slice)
        pre_cutoff = resampled[resampled["timestamp"] <= train_end].reset_index(drop=True)
        if len(pre_cutoff) < WARMUP_BUFFER:
            raise ValueError(
                f"scale={scale}min: only {len(pre_cutoff)} pre-cutoff bars "
                f"available, need {WARMUP_BUFFER}",
            )
        warmup = pre_cutoff.iloc[-WARMUP_BUFFER:].reset_index(drop=True)
        log.info(f"scale={scale}min: {len(warmup)} bars  "
                 f"({warmup['timestamp'].iloc[0]} -> {warmup['timestamp'].iloc[-1]})")
        out[scale] = warmup
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="WF config (e.g. configs/sg1_xauusd_ftmo_rehpo_wf_multiseed.yaml)")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="WF mode: list of fold indices. Mutually exclusive with --train_end_date.")
    ap.add_argument("--train_end_date", default=None,
                    help="L1-multiseed mode: explicit train_end_date (e.g. 2025-06-30). "
                         "Skips fold splitter; one buffer is written per matching checkpoint dir.")
    ap.add_argument("--ckpt_path_glob", default=None,
                    help="L1 mode: explicit glob for checkpoint dirs (e.g. "
                         "'checkpoints/sg1-btc-velotrade-l1-multiseed-seed42_*/checkpoint_final.pth').")
    ap.add_argument("--output_name", default=None,
                    help="Override output filename. Default: norm_warmup_<scale1>-<scale2>-<scale3>.pkl "
                         "(scale-suffixed so configs with different scale tuples can share a checkpoint dir).")
    ap.add_argument("--checkpoint_pattern", default=None,
                    help="Override pattern (default reads from config.ensemble.checkpoint_pattern)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parquet_path = ROOT / config["data"]["file_path"]
    scales = config["features"]["scales"]
    log.info(f"scales={scales}  parquet={parquet_path}")
    out_name = args.output_name or f"norm_warmup_{'-'.join(str(s) for s in scales)}.pkl"

    if args.train_end_date is not None:
        # L1-multiseed mode: explicit train_end + per-seed checkpoint glob
        train_end = pd.Timestamp(args.train_end_date)
        log.info(f"L1 mode: train_end={train_end}")
        warmup = build_warmup_buffer(parquet_path, scales, train_end)
        if args.ckpt_path_glob is None:
            log.error("--train_end_date requires --ckpt_path_glob")
            sys.exit(1)
        for seed in args.seeds:
            pat = args.ckpt_path_glob.format(seed=seed)
            matches = sorted(glob.glob(pat))
            if not matches:
                log.warning(f"seed={seed}: no checkpoint match for {pat}")
                continue
            for ckpt_path in matches:
                ckpt_dir = Path(ckpt_path).parent
                out_path = ckpt_dir / out_name
                with open(out_path, "wb") as f:
                    pickle.dump({
                        "schema_version": "v1",
                        "scales": scales,
                        "n_warmup": WARMUP_BUFFER,
                        "norm_span": int(config["features"].get("norm_span", 120)),
                        "fold": None,
                        "seed": seed,
                        "train_end_date": str(train_end),
                        "buffers": warmup,
                    }, f)
                log.info(f"  wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
        return

    if args.folds is None:
        log.error("Must supply either --folds (WF mode) or --train_end_date (L1 mode)")
        sys.exit(1)

    pattern = args.checkpoint_pattern or config["ensemble"]["checkpoint_pattern"]
    for fold in args.folds:
        train_end = fold_train_end(config, fold)
        log.info(f"fold {fold}: train_end={train_end}")
        warmup = build_warmup_buffer(parquet_path, scales, train_end)

        for seed in args.seeds:
            pat = pattern.format(seed=seed, fold=fold).replace("{fold:02d}", f"{fold:02d}")
            matches = sorted(glob.glob(pat))
            if not matches:
                log.warning(f"seed={seed} fold={fold}: no checkpoint match for {pat}")
                continue
            for ckpt_path in matches:
                ckpt_dir = Path(ckpt_path).parent
                out_path = ckpt_dir / out_name
                with open(out_path, "wb") as f:
                    pickle.dump({
                        "schema_version": "v1",
                        "scales": scales,
                        "n_warmup": WARMUP_BUFFER,
                        "norm_span": int(config["features"].get("norm_span", 120)),
                        "fold": fold,
                        "seed": seed,
                        "train_end_date": str(train_end),
                        "buffers": warmup,
                    }, f)
                log.info(f"  wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
