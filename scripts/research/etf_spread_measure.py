"""Independent effective-spread measurement for the N2 instruments.

WHY. N2 pre-committed its cost model to the Corwin-Schultz high-low estimator. That estimator
returned a median 23.7 bp across the commodity ETFs and 16.5 bp for GLD -- one of the most liquid
ETFs in existence, whose quoted spread is a cent or two on a ~$300 share (~0.3-0.7 bp). A number
that is wrong by ~20x on the instrument it is easiest to check is not a usable cost model.

The suspected mechanism is CIRCULAR and worth recording: Corwin-Schultz infers the spread from the
relationship between one-day and two-day high/low ranges, assuming the range is driven by
volatility plus spread. A large OVERNIGHT GAP inflates the two-day range relative to the one-day
ranges and the estimator books that difference as "spread". N2's hypothesis is precisely that these
instruments have large overnight moves -- so the estimator's bias is maximal on exactly the
instruments the hypothesis selects for, and the cost model is contaminated by the effect it is
supposed to price.

This script measures the spread two ways that CANNOT pick up overnight gaps:

  1. ROLL (1984) on INTRADAY 5-minute returns only. S = 2*sqrt(-cov(r_t, r_{t-1})) where the
     first-order autocovariance is negative. Bid-ask bounce makes consecutive trade-price returns
     negatively autocorrelated; the magnitude identifies the effective spread. Overnight returns
     are dropped, so no gap enters.
  2. A LIVE QUOTED bid/ask snapshot from yfinance, which is a direct observation rather than an
     inference (point-in-time, so it is a cross-check, not a history).

Neither is a substitute for a full historical quote series. Both are enough to establish whether
the pre-committed estimator is usable, which is the question that matters for reading N2.

Usage:
    python scripts/research/etf_spread_measure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TICKERS = ["GLD", "SLV", "DBC", "USO", "DBA",
           "IAU", "SGOL", "SIVR", "PPLT", "PALL", "GSG", "DJP", "USCI", "PDBC",
           "UNG", "BNO", "UGA", "CORN", "WEAT", "SOYB", "CPER"]


def roll_spread(px: pd.Series) -> float:
    """Roll (1984) effective spread from intraday returns. Returns proportional round-trip cost."""
    r = px.pct_change().dropna()
    if len(r) < 200:
        return float("nan")
    cov = float(np.cov(r.iloc[1:], r.iloc[:-1])[0, 1])
    return 2.0 * np.sqrt(-cov) if cov < 0 else 0.0


def main() -> int:
    import yfinance as yf

    out = {}
    print(f"{'ticker':7s} {'roll bp':>9s} {'quoted bp':>10s} {'n5m':>7s}  note")
    for t in TICKERS:
        roll_bp, quoted_bp, n = float("nan"), float("nan"), 0
        try:
            h = yf.Ticker(t).history(period="60d", interval="5m", auto_adjust=False)
            if not h.empty:
                h = h.reset_index()
                dt = pd.to_datetime(h.iloc[:, 0])
                # keep only WITHIN-day consecutive pairs: drop the first bar of each session so no
                # overnight gap ever enters the autocovariance
                day = dt.dt.tz_convert("America/New_York").dt.normalize() if dt.dt.tz \
                    else dt.dt.normalize()
                parts = []
                for _, g in h.assign(_d=day).groupby("_d"):
                    if len(g) > 10:
                        parts.append(g["Close"].astype(float).reset_index(drop=True))
                vals, covs = 0, []
                for p in parts:
                    r = p.pct_change().dropna()
                    if len(r) > 20:
                        covs.append(float(np.cov(r.iloc[1:], r.iloc[:-1])[0, 1]))
                        vals += len(r)
                if covs:
                    mc = float(np.mean(covs))
                    roll_bp = (2.0 * np.sqrt(-mc) * 1e4) if mc < 0 else 0.0
                    n = vals
        except Exception as e:                                        # noqa: BLE001
            print(f"{t:7s} intraday failed: {e}")
        try:
            fi = yf.Ticker(t).info
            b, a = fi.get("bid"), fi.get("ask")
            if b and a and a > b > 0:
                quoted_bp = (a - b) / ((a + b) / 2) * 1e4
        except Exception:                                             # noqa: BLE001
            pass
        out[t] = {"roll_bp": roll_bp, "quoted_bp": quoted_bp, "n_5m_returns": n}
        print(f"{t:7s} {roll_bp:>9.2f} {quoted_bp:>10.2f} {n:>7,}")

    d = ROOT / "results" / "commodity_session"
    d.mkdir(parents=True, exist_ok=True)
    (d / "measured_spreads.json").write_text(json.dumps(out, indent=2, default=float),
                                             encoding="utf-8")
    print(f"\nwrote {d / 'measured_spreads.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
