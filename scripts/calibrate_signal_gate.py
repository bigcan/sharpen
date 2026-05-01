#!/usr/bin/env python
"""Calibrate SG-1 signal_gate thresholds from feature-distribution percentiles.

Computes EMA-Z-normalized ATR, Parkinson volatility, and volume-Z on the training
window of a given SG-1 config, prints p30/p50/p70/p80/p90 percentiles, and suggests
threshold values targeting a ~75% composite pass rate (matches oracle_signal_gate.py
"MARGINAL GO permissive" verdict on CME GC 3-min).

Faster + more targeted than oracle_signal_gate.py for threshold calibration — no DP
backward induction or O(n²) expanding percentile.

Usage:
    python scripts/calibrate_signal_gate.py configs/sg1_xauusd_ftmo_hpo.yaml
    python scripts/calibrate_signal_gate.py configs/sg1_btc_velotrade_hpo.yaml

The suggested thresholds assume `gate_mode: "composite"` (OR of atr/parkinson/volume).
Each signal targets ~p63 (top 37%) so Pr(any open) ≈ 1 - 0.63³ ≈ 75% pass rate
under independence. Re-run after data updates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def compute_signals(path: str, start: str, end: str, resolution: str = "3min") -> dict:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    df = df.resample(resolution).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if "volume" in df.columns:
        df = df[df["volume"] > 0]
    df = df.reset_index()

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    hl_ratio = np.where(low > 0, high / low, 1.0)
    parkinson = np.sqrt(np.log(hl_ratio) ** 2 / (4 * np.log(2)))

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_ema = pd.Series(atr_14).ewm(span=50, adjust=False).mean().values
    atr_ema_sh = np.roll(atr_ema, 1); atr_ema_sh[0] = atr_ema[0]
    atr_std = pd.Series(atr_14).ewm(span=50, adjust=False).std(bias=False).values
    atr_std_sh = np.roll(atr_std, 1); atr_std_sh[0] = atr_std[0]
    atr_norm = np.where(atr_std_sh > 1e-10, (atr_14 - atr_ema_sh) / atr_std_sh, 0.0)

    vol_ema = pd.Series(volume).ewm(span=50, adjust=False).mean().values
    vol_ema_sh = np.roll(vol_ema, 1); vol_ema_sh[0] = vol_ema[0]
    vol_std = pd.Series(volume).ewm(span=50, adjust=False).std(bias=False).values
    vol_std_sh = np.roll(vol_std, 1); vol_std_sh[0] = vol_std[0]
    volume_z = np.where(vol_std_sh > 1e-10, (volume - vol_ema_sh) / vol_std_sh, 0.0)

    return {
        "n_bars": len(df),
        "atr_norm": atr_norm,
        "parkinson": parkinson,
        "abs_volume_z": np.abs(volume_z),
    }


def pct_table(x: np.ndarray, name: str) -> str:
    pcts = [30, 50, 63, 70, 80, 90]
    vals = np.nanpercentile(x, pcts)
    row = "  ".join(f"p{p}={v:+.5f}" if abs(v) < 0.01 else f"p{p}={v:+.3f}" for p, v in zip(pcts, vals))
    return f"{name:18s}  mean={np.nanmean(x):+.4f}  std={np.nanstd(x):.4f}  {row}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", type=Path, help="SG-1 HPO config YAML")
    ap.add_argument("--target-pass-rate", type=float, default=0.75,
                    help="Target composite pass rate (default 0.75)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = cfg["data"]
    path = data["file_path"]
    start = data["train_start_date"]
    end = data["train_end_date"]
    scales = cfg.get("features", {}).get("scales", [3])
    res = f"{scales[0]}min"

    print(f"Config: {args.config.name}")
    print(f"Data:   {path}")
    print(f"Window: {start} → {end}  (resolution {res})")
    print(f"Target composite pass rate: {args.target_pass_rate:.0%}")
    print()

    sig = compute_signals(path, start, end, resolution=res)
    print(f"bars (vol>0): {sig['n_bars']:,}")
    print()
    print(pct_table(sig["atr_norm"], "atr_norm"))
    print(pct_table(sig["parkinson"], "parkinson"))
    print(pct_table(sig["abs_volume_z"], "abs(volume_z)"))
    print()

    # Per-signal pass rate needed for target under independence
    per_signal = 1.0 - (1.0 - args.target_pass_rate) ** (1.0 / 3.0)
    pct_cutoff = (1.0 - per_signal) * 100.0
    atr_thr = float(np.nanpercentile(sig["atr_norm"], pct_cutoff))
    par_thr = float(np.nanpercentile(sig["parkinson"], pct_cutoff))
    vol_thr = float(np.nanpercentile(sig["abs_volume_z"], pct_cutoff))

    print(f"Per-signal pass rate needed: {per_signal:.0%}  (threshold at p{pct_cutoff:.0f})")
    print()
    print("Suggested signal_gate YAML block:")
    print("signal_gate:")
    print("  enabled: true")
    print('  gate_mode: "composite"')
    print(f"  atr_threshold: {atr_thr:.3f}")
    print(f"  parkinson_threshold: {par_thr:.5f}")
    print(f"  volume_threshold: {vol_thr:.3f}")
    current = cfg.get("signal_gate", {})
    if current:
        print()
        print("Current values in config:")
        for k in ("atr_threshold", "parkinson_threshold", "volume_threshold"):
            print(f"  {k}: {current.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
