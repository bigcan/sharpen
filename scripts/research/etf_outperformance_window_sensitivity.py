"""Phase B2 — is the "consistent 20-year outperformer" list a start-date artifact?

Two tests:
  1. START SWEEP: for every start year 2000..2016 (window = start -> 2026-08),
     recompute excess CAGR vs SPY and report how the winner set changes.
  2. DOT-COM INCLUSION: for the ~40 ETFs alive in 2000, measure the full
     26-year record including the 2000-02 bust, and compare each fund's rank
     with and without it.

If the leaders reshuffle with the start date, "consistently outperformed for
10-20 years" is a statement about the sample window, not about the fund.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/etf_outperformance")
TRADING_DAYS = 252
BENCH = "SPY"
END = "2026-08-12"


def cagr(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) < 60:
        return np.nan
    return (v.iloc[-1] / v.iloc[0]) ** (TRADING_DAYS / (len(v) - 1)) - 1


def sharpe(s: pd.Series, rf: pd.Series) -> float:
    r = s.dropna().pct_change().dropna()
    if len(r) < 60 or r.std() == 0:
        return np.nan
    return (r - rf.reindex(r.index).fillna(0.0)).mean() / r.std() * np.sqrt(TRADING_DAYS)


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from etf_outperformance_factors import load_factors

    px = pd.read_parquet(OUT_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(OUT_DIR / "etf_universe_meta.parquet").set_index("ticker")
    rf = load_factors()["RF"]

    pd.set_option("display.width", 240, "display.max_columns", 40)

    # ---- Test 1: start-date sweep -----------------------------------------
    starts = [f"{y}-08-14" for y in range(2000, 2017)]
    sweep_excess: dict[str, pd.Series] = {}
    sweep_rows = []
    for st in starts:
        w = px.loc[st:END]
        if len(w) < 3 * TRADING_DAYS:
            continue
        # require presence within first 5 sessions of the window AND at the end
        head = w.iloc[:5].notna().any()
        tail = w.iloc[-5:].notna().any()
        elig = w.columns[head & tail]
        c = w[elig].apply(cagr)
        exc = c - c[BENCH]
        sweep_excess[st[:4]] = exc
        n = len(elig)
        sweep_rows.append(
            {
                "start_year": st[:4],
                "n_eligible": n,
                "spy_cagr": c[BENCH],
                "n_beat": int((exc > 0).sum()),
                "pct_beat": float((exc > 0).mean()),
                "top1": exc.drop(BENCH, errors="ignore").idxmax(),
                "top1_excess": exc.drop(BENCH, errors="ignore").max(),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    print("\n=== TEST 1: start-date sweep (window = start -> 2026-08) ===")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    exc_mat = pd.DataFrame(sweep_excess)
    exc_mat.to_parquet(OUT_DIR / "start_sweep_excess.parquet")

    # Rank stability of the funds that are leaders in the recent window
    recent_leaders = exc_mat["2011"].dropna().sort_values(ascending=False).head(15).index
    print("\n=== Excess CAGR vs SPY of the 2011-start leaders, by window start year ===")
    show = exc_mat.loc[recent_leaders, ["2000", "2001", "2003", "2005", "2007", "2009", "2011", "2013", "2015"]]
    print(show.to_string(float_format=lambda x: f"{x:,.3f}"))

    # ---- Test 2: the 2000 cohort, full 26-year record ---------------------
    w2000 = px.loc["2000-08-14":END]
    elig2000 = w2000.columns[w2000.iloc[:5].notna().any() & w2000.iloc[-5:].notna().any()]
    print(f"\n=== TEST 2: cohort alive in Aug 2000 (n={len(elig2000)}), record through 2026 ===")
    rows = []
    for t in elig2000:
        s = w2000[t]
        # same fund, but starting after the bust
        s_post = px.loc["2003-03-14":END, t]
        rows.append(
            {
                "ticker": t,
                "category": meta.loc[t, "category"] if t in meta.index else "?",
                "cagr_26y": cagr(s),
                "sharpe_26y": sharpe(s, rf),
                "maxdd_26y": float((s.dropna() / s.dropna().cummax() - 1).min()),
                "cagr_post_bust": cagr(s_post),
            }
        )
    t2 = pd.DataFrame(rows).set_index("ticker")
    t2["excess_26y"] = t2.cagr_26y - t2.loc[BENCH, "cagr_26y"]
    t2["excess_post_bust"] = t2.cagr_post_bust - t2.loc[BENCH, "cagr_post_bust"]
    t2["rank_26y"] = t2.excess_26y.rank(ascending=False).astype(int)
    t2["rank_post_bust"] = t2.excess_post_bust.rank(ascending=False).astype(int)
    t2["rank_shift"] = t2.rank_post_bust - t2.rank_26y
    t2 = t2.sort_values("excess_26y", ascending=False)
    print(
        t2[
            [
                "category", "cagr_26y", "excess_26y", "sharpe_26y", "maxdd_26y",
                "excess_post_bust", "rank_26y", "rank_post_bust", "rank_shift",
            ]
        ].to_string(float_format=lambda x: f"{x:,.3f}")
    )
    t2.to_parquet(OUT_DIR / "cohort_2000_full_record.parquet")

    n_beat_26 = int((t2.excess_26y > 0).sum()) - 1  # exclude SPY itself
    print(
        f"\n2000-cohort beating SPY over 26y: {n_beat_26}/{len(t2) - 1} "
        f"({n_beat_26 / (len(t2) - 1):.1%})"
    )

    # Spearman rank correlation between the two rankings
    from scipy.stats import spearmanr

    rho, p = spearmanr(t2.excess_26y, t2.excess_post_bust)
    print(f"Spearman rank corr (full-26y vs post-bust ordering): rho={rho:.3f} p={p:.4f}")


if __name__ == "__main__":
    main()
