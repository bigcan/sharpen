"""Is the mega-cap / growth "edge" a strategy property or an inception-date property?

Operator probes (S553-cont-161): XLG (S&P 500 Top 50) and IWY (Russell Top 200
Growth) both beat SPY, and neither is a sector bet or levered beta -- XLG's beta
is 0.937, BELOW the market, with a smaller drawdown than SPY. If any fund family
has a real structural edge, it should be this one.

The test that separates the two explanations: line every fund implementing the
same idea up by INCEPTION DATE and measure each against SPY on its own window.
A strategy property is roughly constant across the family. An artifact of when
measurement started is monotone in launch date.

Two families:
  concentration -- OEF (S&P 100), XLG (Top 50), MGC (Mega Cap), IWY (Top 200 Gr)
  growth        -- SPYG/IVW (S&P 500 Growth), IWF (Russell 1000 Gr), VUG, MGK,
                   IWY, SCHG

Also reports each fund's excess by window start year, so the gradient is visible
within a single fund as well as across the family.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ETF_DIR = ROOT / "results" / "etf_outperformance"
OUT = ROOT / "results" / "etf_outperformance"
BENCH = "SPY"
TD = 252
END = "2026-08-12"

FAMILIES = {
    "concentration": ["OEF", "XLG", "MGC", "IWY"],
    "growth": ["SPYG", "IWF", "IVW", "VUG", "MGK", "IWY", "SCHG"],
}
START_YEARS = [2006, 2009, 2012, 2015, 2018, 2021, 2022]


def stats(px: pd.DataFrame, t: str, start: str) -> tuple[float, float, float, str]:
    from etf_outperformance_factors import load_factors

    rf = load_factors()["RF"]
    s = px.loc[start:END, t].dropna()
    if len(s) < 250:
        return (np.nan,) * 3 + ("",)
    r = s.pct_change().dropna()
    cagr = (s.iloc[-1] / s.iloc[0]) ** (TD / len(r)) - 1
    sh = (r - rf.reindex(r.index).fillna(0.0)).mean() / r.std() * np.sqrt(TD)
    dd = float((s / s.cummax() - 1).min())
    return float(cagr), float(sh), dd, str(s.index[0].date())


def main() -> int:
    px = pd.read_parquet(ETF_DIR / "etf_prices_adj.parquet")
    out: dict = {}

    for fam, tickers in FAMILIES.items():
        rows = []
        for t in tickers:
            if t not in px.columns:
                continue
            inception = str(px[t].dropna().index[0].date())
            c, sh, dd, d0 = stats(px, t, inception)
            bc, bsh, bdd, _ = stats(px, BENCH, inception)
            rows.append({
                "ticker": t, "inception": d0, "cagr": c, "spy_cagr": bc,
                "excess": c - bc, "sharpe": sh, "spy_sharpe": bsh,
                "sharpe_gap": sh - bsh, "maxdd": dd, "spy_maxdd": bdd,
            })
        df = pd.DataFrame(rows).sort_values("inception").reset_index(drop=True)
        # the diagnostic: correlation between launch date and measured edge
        order = np.arange(len(df))
        rho = float(np.corrcoef(order, df.excess.to_numpy())[0, 1]) if len(df) > 2 else np.nan
        out[fam] = {"table": df.to_dict("records"), "rank_corr_inception_vs_excess": rho}

        print(f"\n{'=' * 92}\nFAMILY: {fam} — each fund vs SPY on its OWN window, ordered by inception")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        print(f"  corr(inception order, excess) = {rho:+.3f}"
              f"  {'=> MONOTONE IN LAUNCH DATE, not a strategy property' if rho > 0.7 else ''}")

    print(f"\n{'=' * 92}\nWITHIN-FUND gradient: excess vs SPY by window start year")
    grid = {}
    for t in sorted({x for v in FAMILIES.values() for x in v}):
        if t not in px.columns:
            continue
        inception = px[t].dropna().index[0]
        row = {}
        for y in START_YEARS:
            start = pd.Timestamp(f"{y}-01-01")
            # A fund that did not exist at `start` has no window here. Comparing
            # its post-inception series against SPY measured from `start` would
            # score two different windows against each other.
            if inception > start:
                row[str(y)] = np.nan
                continue
            c, _, _, _ = stats(px, t, f"{y}-01-01")
            b, _, _, _ = stats(px, BENCH, f"{y}-01-01")
            row[str(y)] = c - b if np.isfinite(c) else np.nan
        grid[t] = row
    g = pd.DataFrame(grid).T
    print(g.to_string(float_format=lambda x: f"{x:+,.4f}"))

    out["within_fund_by_start_year"] = g.to_dict()
    (OUT / "concentration_family_gradient.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT / 'concentration_family_gradient.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
