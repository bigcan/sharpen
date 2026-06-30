"""Fable clean-room audit of the ETF daily data feeding the TSMOM 'survivor'.

Independent of all project code on purpose. Checks:
  1. prices_daily.parquet (close-only cache the GO verdict ran on): shape, NaN
     pattern, duplicate/non-monotonic dates, extreme daily returns, stale runs.
  2. ohlcv_daily_raw.parquet vs ohlcv_daily.parquet: what exactly did the
     "outlier repair" change (field, magnitude)? Were closes touched?
  3. prices_daily vs ohlcv_daily_raw close: same source downloaded twice at
     different dates -> returns should match away from dividend ex-dates.
Writes JSON report to results/fable_verdict/etf_data_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
XDIR = ROOT / "results" / "xsec_momentum"
OUT = ROOT / "results" / "fable_verdict"
OUT.mkdir(parents=True, exist_ok=True)

report: dict = {}

# ---------------------------------------------------------------- 1. prices_daily
px = pd.read_parquet(XDIR / "prices_daily.parquet")
px = px.sort_index()
report["prices_daily"] = {
    "shape": list(px.shape),
    "date_min": str(px.index.min().date()),
    "date_max": str(px.index.max().date()),
    "index_monotonic": bool(px.index.is_monotonic_increasing),
    "index_duplicates": int(px.index.duplicated().sum()),
    "tickers": list(px.columns),
}

# NaN pattern: leading NaNs are legit (inception); interior NaNs are suspect.
nan_info = {}
for t in px.columns:
    s = px[t]
    first = s.first_valid_index()
    interior_nan = int(s.loc[first:].isna().sum()) if first is not None else -1
    nan_info[t] = {"first_valid": str(first.date()) if first is not None else None,
                   "interior_nans": interior_nan}
report["prices_daily"]["nan_pattern"] = nan_info

# Extreme daily returns + stale runs per ticker.
rets = px.pct_change()
ext = {}
for t in px.columns:
    r = rets[t].dropna()
    big = r[r.abs() > 0.10]
    ext[t] = {
        "n_abs_gt_10pct": int(len(big)),
        "n_abs_gt_20pct": int((r.abs() > 0.20).sum()),
        "worst_days": {str(k.date()): round(float(v), 4)
                       for k, v in r.abs().nlargest(5).items()},
        "max_stale_run": int(
            (px[t].dropna().diff() == 0).astype(int)
            .groupby((px[t].dropna().diff() != 0).cumsum()).sum().max()
        ),
    }
report["prices_daily"]["extremes"] = ext

# Weekend rows (should be none for US ETFs).
report["prices_daily"]["weekend_rows"] = int(px.index.dayofweek.isin([5, 6]).sum())

# ------------------------------------------- 2. raw vs clean OHLCV (repair diff)
raw = pd.read_parquet(XDIR / "ohlcv_daily_raw.parquet")
cln = pd.read_parquet(XDIR / "ohlcv_daily.parquet")
for df in (raw, cln):
    df["date"] = pd.to_datetime(df["date"])
key = ["date", "ticker"]
m = raw.merge(cln, on=key, suffixes=("_raw", "_cln"))
diff_summary = {}
for field in ["open", "high", "low", "close", "volume"]:
    a, b = m[f"{field}_raw"], m[f"{field}_cln"]
    both = a.notna() & b.notna()
    d = (a[both] - b[both]).abs()
    rel = d / a[both].abs().clip(lower=1e-9)
    changed = rel > 1e-9
    rows = m.loc[both].loc[changed.values]
    diff_summary[field] = {
        "n_changed": int(changed.sum()),
        "max_rel_change": float(rel.max()) if len(rel) else 0.0,
        "changed_examples": [
            {"ticker": r["ticker"], "date": str(r["date"].date()),
             "raw": round(float(r[f"{field}_raw"]), 4),
             "clean": round(float(r[f"{field}_cln"]), 4)}
            for _, r in rows.head(8).iterrows()
        ],
    }
report["repair_diff"] = diff_summary

# --------------------------- 3. prices_daily vs ohlcv_daily_raw close (re-download)
cw = cln.pivot(index="date", columns="ticker", values="close").sort_index()
common_t = [t for t in px.columns if t in cw.columns]
common_d = px.index.intersection(cw.index)
r1 = px.loc[common_d, common_t].pct_change()
r2 = cw.loc[common_d, common_t].pct_change()
xchk = {}
for t in common_t:
    a, b = r1[t].dropna(), r2[t].dropna()
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).abs()
    xchk[t] = {
        "ret_corr": round(float(a.loc[idx].corr(b.loc[idx])), 6),
        "n_days_absdiff_gt_10bps": int((d > 0.001).sum()),
        "worst_diff_days": {str(k.date()): round(float(v), 5)
                           for k, v in d.nlargest(3).items()},
    }
report["two_download_return_xcheck"] = xchk

(OUT / "etf_data_audit.json").write_text(json.dumps(report, indent=2))

# Console digest
print(f"prices_daily: {px.shape}, {report['prices_daily']['date_min']} -> "
      f"{report['prices_daily']['date_max']}, dup={report['prices_daily']['index_duplicates']}, "
      f"weekend={report['prices_daily']['weekend_rows']}")
bad_interior = {t: v for t, v in nan_info.items() if v["interior_nans"] not in (0, -1)}
print("interior NaNs:", bad_interior if bad_interior else "none")
print("\nrepair diff (raw->clean):")
for f, d in diff_summary.items():
    print(f"  {f:7s} changed={d['n_changed']:4d} max_rel={d['max_rel_change']:.4f}")
print("\nworst two-download return mismatches (days >10bps):")
for t, v in sorted(xchk.items(), key=lambda kv: -kv[1]["n_days_absdiff_gt_10bps"])[:6]:
    print(f"  {t:4s} corr={v['ret_corr']:.4f} days>10bps={v['n_days_absdiff_gt_10bps']:4d} "
          f"worst={v['worst_diff_days']}")
print("\nextreme-return spot table (n>10% / n>20% / worst):")
for t, v in ext.items():
    w0 = list(v["worst_days"].items())[0]
    print(f"  {t:4s} {v['n_abs_gt_10pct']:3d} / {v['n_abs_gt_20pct']:2d} / {w0[0]} {w0[1]:+.3f} "
          f"stale_run={v['max_stale_run']}")
