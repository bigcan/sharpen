"""Phase D — was any of it identifiable ex ante?

Everything in Phases B/C is measured with hindsight. The only question that
matters for building something is whether a rule available AT THE TIME picks
the future winners.

Test: at each formation date (month-end), rank every ETF with sufficient history
by trailing excess return vs SPY over lookback L. Sort into quintiles, hold for
horizon H, measure realised excess return vs SPY. Report the Q5-Q1 spread and
the rank IC (Spearman between trailing and forward excess).

Lookbacks span the horizon where this project already has a validated edge
(12m TSMOM, net SR ~0.60) up to the 5-10y horizon implied by "consistently
outperformed for 10-20 years".

Newey-West t-stats on the Q5-Q1 spread account for the overlap induced by
monthly formation with multi-year holds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/etf_outperformance")
BENCH = "SPY"
TRADING_DAYS = 252
N_Q = 5

# Equity-only: a bond/commodity ETF ranking low is not an interesting "loser".
EQUITY_CATS = {
    "broad_us", "us_style", "us_midsmall", "sector_spdr", "sector_vanguard",
    "sector_ishares", "industry_thematic", "factor_smartbeta", "dividend",
    "growth_tech_core", "intl_developed", "intl_emerging", "global", "reit",
}


def run(px: pd.DataFrame, tickers: list[str], lookback_m: int, hold_m: int) -> dict:
    m_ends = px.resample("ME").last().index
    lb = int(lookback_m / 12 * TRADING_DAYS)
    hz = int(hold_m / 12 * TRADING_DAYS)

    q_rets: dict[int, list[float]] = {q: [] for q in range(1, N_Q + 1)}
    ics, dates, spreads = [], [], []

    idx = px.index
    for dt in m_ends:
        pos = idx.searchsorted(dt)
        if pos - lb < 0 or pos + hz >= len(idx):
            continue
        t0, tf = idx[pos - lb], idx[pos]
        t1 = idx[pos + hz]

        past = (px.loc[tf, tickers] / px.loc[t0, tickers] - 1.0)
        fwd = (px.loc[t1, tickers] / px.loc[tf, tickers] - 1.0)
        past_b = px.loc[tf, BENCH] / px.loc[t0, BENCH] - 1.0
        fwd_b = px.loc[t1, BENCH] / px.loc[tf, BENCH] - 1.0

        pe = (past - past_b).dropna()
        fe = (fwd - fwd_b).dropna()
        both = pe.index.intersection(fe.index)
        if len(both) < 30:
            continue
        pe, fe = pe[both], fe[both]

        ranks = pe.rank(pct=True)
        qs = np.ceil(ranks * N_Q).clip(1, N_Q).astype(int)
        for q in range(1, N_Q + 1):
            sel = fe[qs == q]
            if len(sel):
                q_rets[q].append(sel.mean())
        spreads.append(fe[qs == N_Q].mean() - fe[qs == 1].mean())
        ics.append(spearmanr(pe, fe).statistic)
        dates.append(tf)

    if not spreads:
        return {}
    sp = pd.Series(spreads, index=dates)
    # Newey-West on the overlapping spread series
    nw = sm.OLS(sp.values, np.ones(len(sp))).fit(cov_type="HAC", cov_kwds={"maxlags": hold_m})
    ic = pd.Series(ics, index=dates)
    return {
        "lookback_m": lookback_m,
        "hold_m": hold_m,
        "n_formations": len(sp),
        **{f"Q{q}_fwd_excess": np.mean(v) if v else np.nan for q, v in q_rets.items()},
        "Q5_minus_Q1": sp.mean(),
        "spread_t_nw": nw.tvalues[0],
        "mean_rank_ic": ic.mean(),
        "ic_pos_frac": float((ic > 0).mean()),
    }


def main() -> None:
    px = pd.read_parquet(OUT_DIR / "etf_prices_adj.parquet")
    meta = pd.read_parquet(OUT_DIR / "etf_universe_meta.parquet").set_index("ticker")
    eq = [t for t in px.columns if t in meta.index and meta.loc[t, "category"] in EQUITY_CATS]
    log.info("equity ETF universe for predictability: %d", len(eq))

    grid = [
        (12, 12), (12, 36), (24, 12), (24, 36),
        (36, 36), (60, 12), (60, 36), (60, 60), (120, 36), (120, 60),
    ]
    rows = [r for lb, h in grid if (r := run(px, eq, lb, h))]
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "predictability.csv", index=False)

    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("\n=== PREDICTABILITY: does past outperformance predict future outperformance? ===")
    print("(forward excess return vs SPY over the hold, by trailing-excess quintile)\n")
    print(res.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(
        "\nQ5 = best trailing performers, Q1 = worst. A positive Q5-Q1 means "
        "past winners keep winning."
    )


if __name__ == "__main__":
    main()
