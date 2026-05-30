#!/usr/bin/env python
"""Build Protocol v2 data manifest for an OHLCV parquet.

Writes `<stem>.manifest.json` alongside the parquet with fields required by
`scripts/validate_config.py`:
    - clean_ohlcv_passed (bool, DATA-CLEAN invariant)
    - nan_count (int, must be 0)
    - last_ts (ISO) / first_ts (ISO)
    - freq_seconds (int, bar frequency)
    - max_gap_bars (int, longest consecutive missing-bar gap)
    - regime_quartiles (dict, volatility-regime coverage fractions)

Usage:
    python scripts/build_data_manifest.py data/foo.parquet --write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_ohlcv import detect_outliers  # noqa: E402  (sibling script in scripts/)


def build_manifest(parquet_path: Path) -> dict:
    df = pd.read_parquet(parquet_path)

    ts_col = None
    for cand in ("timestamp", "ts", "datetime", "date", "time"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        if df.index.name and any(k in df.index.name.lower() for k in ("time", "date", "ts")):
            df = df.reset_index().rename(columns={df.index.name: "timestamp"})
            ts_col = "timestamp"
        else:
            raise ValueError(f"{parquet_path.name}: no timestamp column/index found")

    df = df.sort_values(ts_col).reset_index(drop=True)
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{parquet_path.name}: {int(ts.isna().sum())} unparseable timestamps")

    diffs = ts.diff().dt.total_seconds().dropna()
    freq_seconds = int(diffs.median()) if len(diffs) else 0
    max_gap_bars = int((diffs / freq_seconds - 1).max()) if freq_seconds else 0

    ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    nan_count = int(df[ohlcv_cols].isna().sum().sum()) if ohlcv_cols else 0

    # Regime coverage. NOTE: binning rolling-vol against the data's OWN quartiles
    # is degenerate — each bucket is ~0.25 by construction, so it can NEVER trip
    # the validate_config `regime_quartile < 0.10` coverage gate (audit finding
    # P1-03, sg1_btc_strategy_audit_2026-05-29.md). We instead bin against FIXED
    # multiples of the dataset median vol, which makes the four fractions
    # genuinely informative: a regime-poor (all-calm or all-stress) dataset now
    # surfaces a bucket well below 0.10. `counts.get(k, 0.0)` keeps all four keys
    # so an EMPTY bucket reports 0.0 (and trips the gate) instead of vanishing.
    # (Per-train-window coverage vs global edges remains a validate_config
    # follow-up; see the audit roadmap.)
    regime_quartiles: dict[str, float] = {}
    if "close" in df.columns and len(df) > 100:
        returns = df["close"].astype(float).pct_change().dropna()
        rolling_vol = returns.abs().rolling(30, min_periods=10).mean().dropna()
        med = float(rolling_vol.median())
        if len(rolling_vol) > 4 and med > 0:
            bins = [-np.inf, 0.5 * med, med, 2.0 * med, np.inf]
            labels = pd.cut(rolling_vol, bins=bins, labels=["q1_low", "q2", "q3", "q4_high"])
            counts = labels.value_counts(normalize=True)
            regime_quartiles = {k: float(counts.get(k, 0.0)) for k in ("q1_low", "q2", "q3", "q4_high")}

    # DATA-CLEAN invariant: run the REAL outlier detector (scripts/clean_ohlcv.py)
    # rather than the prior weak high<low / close<=0 self-check (audit finding
    # P1-02), and record provenance so the manifest is an auditable cleaning
    # contract rather than a self-certified flag.
    clean_threshold = 0.05
    n_bad_high = n_bad_low = n_invariant = 0
    have_ohlc = all(c in df.columns for c in ("open", "high", "low", "close"))
    if have_ohlc:
        det = detect_outliers(df, threshold=clean_threshold)
        n_bad_high = int(np.asarray(det["bad_high"]).sum())
        n_bad_low = int(np.asarray(det["bad_low"]).sum())
        n_invariant = int(
            np.asarray(det["inv_high_open"]).sum()
            + np.asarray(det["inv_high_close"]).sum()
            + np.asarray(det["inv_low_open"]).sum()
            + np.asarray(det["inv_low_close"]).sum()
        )
    clean_ohlcv_passed = have_ohlc and (n_bad_high + n_bad_low + n_invariant) == 0
    if "close" in df.columns and (df["close"].astype(float) <= 0).any():
        clean_ohlcv_passed = False

    return {
        "parquet_file": parquet_path.name,
        "row_count": int(len(df)),
        "first_ts": ts.iloc[0].isoformat(),
        "last_ts": ts.iloc[-1].isoformat(),
        "freq_seconds": freq_seconds,
        "max_gap_bars": max_gap_bars,
        "nan_count": nan_count,
        "clean_ohlcv_passed": clean_ohlcv_passed,
        "clean_threshold": clean_threshold,
        "clean_outliers": {
            "bad_high": n_bad_high,
            "bad_low": n_bad_low,
            "invariant_violations": n_invariant,
        },
        "regime_quartiles": regime_quartiles,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--write", action="store_true", help="Write <stem>.manifest.json (else dry-run)")
    args = ap.parse_args()

    manifest = build_manifest(args.parquet)
    print(json.dumps(manifest, indent=2))

    if args.write:
        out = args.parquet.with_suffix(".manifest.json")
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWritten: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
