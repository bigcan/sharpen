"""Phase D2 — is the 12-36m momentum spread just the secular tech trade again?

Three checks on the best cell from Phase D (36m lookback / 36m hold):
  1. SUB-PERIOD split: does the spread survive in both halves of the sample?
  2. COMPOSITION: what sits in Q5 through time? If Q5 is permanently tech, the
     "momentum" result is one macro bet with extra steps.
  3. SECTOR-NEUTRAL: re-rank within category. If the spread dies, the signal is
     picking sectors, not funds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/etf_outperformance")
BENCH = "SPY"
TRADING_DAYS = 252
N_Q = 5

EQUITY_CATS = {
    "broad_us", "us_style", "us_midsmall", "sector_spdr", "sector_vanguard",
    "sector_ishares", "industry_thematic", "factor_smartbeta", "dividend",
    "growth_tech_core", "intl_developed", "intl_emerging", "global", "reit",
}
TECHY = {"growth_tech_core", "sector_spdr", "sector_vanguard", "sector_ishares", "industry_thematic"}


def panel(px, tickers, lookback_m, hold_m, neutral_by: pd.Series | None = None):
    m_ends = px.resample("ME").last().index
    lb = int(lookback_m / 12 * TRADING_DAYS)
    hz = int(hold_m / 12 * TRADING_DAYS)
    idx = px.index
    recs = []
    for dt in m_ends:
        pos = idx.searchsorted(dt)
        if pos - lb < 0 or pos + hz >= len(idx):
            continue
        t0, tf, t1 = idx[pos - lb], idx[pos], idx[pos + hz]
        pe = (px.loc[tf, tickers] / px.loc[t0, tickers] - 1.0) - (px.loc[tf, BENCH] / px.loc[t0, BENCH] - 1.0)
        fe = (px.loc[t1, tickers] / px.loc[tf, tickers] - 1.0) - (px.loc[t1, BENCH] / px.loc[tf, BENCH] - 1.0)
        pe, fe = pe.dropna(), fe.dropna()
        both = pe.index.intersection(fe.index)
        if len(both) < 30:
            continue
        pe, fe = pe[both], fe[both]
        if neutral_by is not None:
            grp = neutral_by.reindex(both)
            rk = pe.groupby(grp).rank(pct=True)
        else:
            rk = pe.rank(pct=True)
        q = np.ceil(rk * N_Q).clip(1, N_Q).astype(int)
        recs.append(pd.DataFrame({"date": tf, "ticker": both, "q": q.values, "fwd": fe.values}))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def spread_stats(pan: pd.DataFrame, hold_m: int) -> tuple[float, float, int]:
    g = pan.groupby(["date", "q"]).fwd.mean().unstack()
    sp = (g[N_Q] - g[1]).dropna()
    if len(sp) < 12:
        return np.nan, np.nan, len(sp)
    nw = sm.OLS(sp.values, np.ones(len(sp))).fit(cov_type="HAC", cov_kwds={"maxlags": hold_m})
    return sp.mean(), nw.tvalues[0], len(sp)


def main() -> None:
    px = pd.read_parquet(OUT_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(OUT_DIR / "etf_universe_meta.parquet").set_index("ticker")
    cat = meta["category"]
    eq = [t for t in px.columns if t in cat.index and cat[t] in EQUITY_CATS]

    LB, HOLD = 36, 36
    pan = panel(px, eq, LB, HOLD)
    m, t, n = spread_stats(pan, HOLD)
    print(f"\n=== BASELINE {LB}m/{HOLD}m: Q5-Q1 = {m:.4f}, t_NW = {t:.2f}, n={n} formations ===")

    # 1. sub-period split
    print("\n=== CHECK 1: sub-period stability ===")
    mid = pan.date.quantile(0.5)
    for label, sub in [("first half", pan[pan.date <= mid]), ("second half", pan[pan.date > mid])]:
        m_, t_, n_ = spread_stats(sub, HOLD)
        rng = f"{sub.date.min():%Y-%m} .. {sub.date.max():%Y-%m}"
        print(f"  {label:12s} {rng}  Q5-Q1 = {m_:+.4f}  t_NW = {t_:5.2f}  n={n_}")

    # 2. composition of Q5
    print("\n=== CHECK 2: what is in Q5? (share of Q5 slots by category) ===")
    q5 = pan[pan.q == N_Q].copy()
    q5["category"] = q5.ticker.map(cat)
    q5["era"] = np.where(q5.date <= mid, "first half", "second half")
    counts = q5.groupby(["era", "category"]).size().unstack(level=0).fillna(0)
    comp = (counts / counts.sum()).sort_values("second half", ascending=False)
    print(comp.head(10).to_string(float_format=lambda x: f"{x:,.3f}"))
    techshare = q5.assign(techy=q5.category.isin(TECHY)).groupby("era").techy.mean()
    print(f"\n  tech/sector share of Q5: {techshare.to_dict()}")

    # 3. sector-neutral ranking
    print("\n=== CHECK 3: rank WITHIN category (sector-neutral) ===")
    pan_n = panel(px, eq, LB, HOLD, neutral_by=cat)
    m_n, t_n, n_n = spread_stats(pan_n, HOLD)
    print(f"  sector-neutral Q5-Q1 = {m_n:+.4f}  t_NW = {t_n:.2f}  n={n_n}")
    print(f"  vs unconditional      = {m:+.4f}  t_NW = {t:.2f}")
    print(f"  => {100 * (1 - m_n / m):.0f}% of the spread came from picking CATEGORIES, not funds")

    # 4. long-only leg, annualised, which is what a benchmark-relative investor gets
    g = pan.groupby(["date", "q"]).fwd.mean().unstack()
    print("\n=== CHECK 4: long-only Q5 leg (what you can actually hold) ===")
    for q in range(1, N_Q + 1):
        ann = (1 + g[q].mean()) ** (12 / HOLD) - 1
        print(f"  Q{q}: mean {HOLD}m excess {g[q].mean():+.4f}  => {ann:+.2%}/yr vs SPY (gross)")


if __name__ == "__main__":
    main()
