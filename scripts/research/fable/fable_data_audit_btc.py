"""Fable clean-room audit of data/btc_usdt_1min_bybit.parquet (canary input).

Checks: schema, monotonic timestamps, duplicates, gap census, OHLC invariants,
zero/negative prices, zero-volume runs, extreme 1-min returns, daily-close
extremes vs known history, and a 15-min resample preview for the canary windows.
Writes results/fable_verdict/btc_data_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fable_verdict"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data" / "btc_usdt_1min_bybit.parquet")
report: dict = {"columns": list(df.columns), "n_rows": int(len(df))}

# Normalize timestamp column/index
if "timestamp" in df.columns:
    ts = pd.to_datetime(df["timestamp"])
elif isinstance(df.index, pd.DatetimeIndex):
    ts = df.index.to_series().reset_index(drop=True)
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
else:
    raise SystemExit(f"no timestamp: cols={df.columns}")
df = df.assign(_ts=ts.values).sort_values("_ts").reset_index(drop=True)

report["ts_min"] = str(df["_ts"].min())
report["ts_max"] = str(df["_ts"].max())
report["tz"] = str(getattr(df['_ts'].dt, 'tz', None))
report["dup_ts"] = int(df["_ts"].duplicated().sum())

dt = df["_ts"].diff().dt.total_seconds().dropna()
gap_census = dt.value_counts().head(10)
report["dt_census_sec"] = {str(int(k)): int(v) for k, v in gap_census.items()}
report["n_gaps_gt_60s"] = int((dt > 60).sum())
report["max_gap_sec"] = int(dt.max())
big_gaps = df.loc[dt[dt > 60].index, "_ts"]
report["biggest_gaps"] = [str(x) for x in big_gaps.head(10)]

o, h, l, c = (df[k].astype(float).to_numpy() for k in ["open", "high", "low", "close"])
v = df["volume"].astype(float).to_numpy() if "volume" in df.columns else np.ones(len(df))
report["ohlc_invariant_violations"] = int(
    ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum())
report["nonpositive_prices"] = int((np.stack([o, h, l, c]) <= 0).sum())
report["zero_volume_bars"] = int((v == 0).sum())

# open[t] vs close[t-1] continuity (perp should be near-continuous)
oc_jump = np.abs(o[1:] / c[:-1] - 1)
report["open_vs_prevclose"] = {
    "p99.9": float(np.quantile(oc_jump, 0.999)),
    "max": float(oc_jump.max()),
    "n_gt_1pct": int((oc_jump > 0.01).sum()),
}

r1 = pd.Series(c).pct_change().dropna()
report["one_min_returns"] = {
    "n_abs_gt_2pct": int((r1.abs() > 0.02).sum()),
    "n_abs_gt_5pct": int((r1.abs() > 0.05).sum()),
    "worst": float(r1.abs().max()),
}

# Stale runs: identical closes
stale = (pd.Series(c).diff() == 0)
runs = stale.astype(int).groupby((~stale).cumsum()).sum()
report["max_stale_close_run_min"] = int(runs.max())

# Daily closes for external cross-check + history sanity
d = df.set_index("_ts")["close"].astype(float).resample("1D").last().dropna()
dr = d.pct_change().dropna()
report["daily"] = {
    "n_days": int(len(d)),
    "worst_days": {str(k.date()): round(float(vv), 4) for k, vv in dr.nsmallest(5).items()},
    "best_days": {str(k.date()): round(float(vv), 4) for k, vv in dr.nlargest(5).items()},
    "sample_closes": {str(k.date()): round(float(d.loc[k]), 1)
                      for k in [d.index[0], d.index[len(d)//4], d.index[len(d)//2],
                                d.index[3*len(d)//4], d.index[-1]]},
}

# Canary fold-0 window preview (2025-11-30 .. 2025-12-31), 15-min resample
w = df.set_index("_ts").loc["2025-11-28":"2026-01-02"]
c15 = w["close"].resample("15min", label="left", closed="left").last().dropna()
report["fold0_window_15m"] = {
    "n_bars": int(len(c15)),
    "first": str(c15.index[0]), "last": str(c15.index[-1]),
    "px_range": [float(c15.min()), float(c15.max())],
    "window_return_pct": round(float(c15.iloc[-1] / c15.loc["2025-11-30 16:30"] - 1) * 100, 2)
    if pd.Timestamp("2025-11-30 16:30") in c15.index else None,
}

(OUT / "btc_data_audit.json").write_text(json.dumps(report, indent=2))
for k, vv in report.items():
    if k not in ("columns",):
        print(f"{k}: {vv}")
