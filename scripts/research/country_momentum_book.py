"""Is C1's low Sharpe the SIGNAL or the BOOK?

`country_momentum_eval.py` measured cross-sectional country momentum through the funnel's naive
equal-weighted decile book: IC-IR 0.101 (CI excluding zero, control-validated) but net Sharpe only
**+0.081** with a **-37%** drawdown and 5.07x turnover.

This project's own validated edge, cross-asset TSMOM at net SR 0.60, does NOT come from a naive
book — it comes from per-asset vol scaling. `portfolio_frontier.metrics` says so directly, reporting
return/DD only after normalising, because a raw un-vol-targeted book shows "scary 90%+ DDs that are
pure leverage, not strategy risk". Sharpe is leverage-invariant so scaling the whole book cannot
change it, but per-asset vol WEIGHTING changes composition and therefore can.

So this reruns the SAME pre-registered 12-1 cross-sectional signal through the house's OWN validated
cross-sectional builder, `xsec_momentum_falsification.xsmom_weights` (rank 12-1, long top third /
short bottom third, vol-scaled) and the same `backtest` cost machinery. The output is directly
comparable to the TSMOM 0.60 headline.

DISCIPLINE: this is ONE pre-specified construction — the house's already-validated one — not a
sweep over weighting schemes. The signal is unchanged and its IC is already pre-registered and
measured; only the portfolio construction differs. Whatever this prints is the answer, including if
it is worse.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import xsec_momentum_falsification as mom  # noqa: E402
from country_momentum_eval import TICKERS  # noqa: E402

log = logging.getLogger("country_momentum_book")
OUT = ROOT / "results" / "country_momentum"

# Country-ETF round-trip cost models (bps of turnover). The funnel's Taiwan-oriented COST_MODELS do
# not apply: no 0.30% transaction tax here.
COSTS = {"frictionless": 0.0, "etf_2bps": 0.0002, "etf_5bps": 0.0005, "etf_10bps": 0.0010}


def _sharpe(d: pd.Series) -> float:
    d = d.dropna()
    s = d.std()
    return float(d.mean() / s * np.sqrt(mom.ANN)) if s > 0 else 0.0


def _max_dd(d: pd.Series) -> float:
    eq = (1 + d.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import yfinance as yf
    raw = yf.download(TICKERS, start="1996-01-01", end="2026-07-31", progress=False,
                      auto_adjust=True)["Close"]
    close = raw.dropna(how="all").sort_index()
    keep = [t for t in TICKERS if t in close.columns and close[t].notna().sum() >= 750]
    close = close[keep]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    warm = mom.LOOKBACKS[-1] + mom.SKIP + mom.VOL_WIN
    rebal = rebal[rebal >= close.index[warm]]
    log.info("panel: %d tickers, %d days (%s..%s), %d monthly rebalances",
             len(keep), len(close), close.index[0].date(), close.index[-1].date(), len(rebal))

    # The house's OWN validated cross-sectional construction, unchanged.
    w = mom.xsmom_weights(close, rets, rebal, keep)
    gross, cost_daily, turn = mom.backtest(w, rets)

    rows = {}
    for name, bps in COSTS.items():
        dw = w.fillna(0.0).diff().abs().sum(axis=1)
        dw.iloc[0] = w.iloc[0].abs().sum()
        cost = dw.reindex(rets.index).fillna(0.0) * bps
        net = (gross - cost).dropna()
        rows[name] = {"sharpe": round(_sharpe(net), 3),
                      "ann_ret_pct": round(float(net.mean() * mom.ANN * 100), 2),
                      "ann_vol_pct": round(float(net.std() * np.sqrt(mom.ANN) * 100), 2),
                      "max_dd_pct": round(_max_dd(net) * 100, 2),
                      "n_days": int(len(net))}
    # leverage-invariant comparison at a common 10% vol (what portfolio_frontier reports)
    net5 = (gross - (w.fillna(0.0).diff().abs().sum(axis=1).reindex(rets.index).fillna(0.0)
                     * COSTS["etf_5bps"])).dropna()
    v = float(net5.std() * np.sqrt(mom.ANN))
    at10 = net5 * (0.10 / v) if v > 0 else net5

    out = {
        "question": "Is C1's net SR 0.081 the signal or the naive decile book?",
        "construction": "xsec_momentum_falsification.xsmom_weights (house-validated, vol-scaled)",
        "n_tickers": len(keep), "turnover_ann": round(turn, 2),
        "by_cost": rows,
        "at_10pct_vol_etf_5bps": {
            "ann_ret_pct": round(float(at10.mean() * mom.ANN * 100), 2),
            "max_dd_pct": round(_max_dd(at10) * 100, 2)},
        "funnel_naive_book_reference": {"net_sharpe_standard": 0.081, "max_dd": -0.374,
                                        "turnover_ann": 5.07},
        "tsmom_house_edge_reference": {"net_sharpe": 0.60},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "country_momentum_book.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("C1 cross-sectional country momentum — HOUSE vol-scaled book vs naive decile book")
    print("=" * 78)
    print(f"tickers {len(keep)}   turnover/yr {turn:.2f}")
    print(f"{'cost':<14}{'Sharpe':>8}{'ret%':>8}{'vol%':>8}{'maxDD%':>9}")
    for k, r in rows.items():
        print(f"{k:<14}{r['sharpe']:>8.3f}{r['ann_ret_pct']:>8.2f}{r['ann_vol_pct']:>8.2f}"
              f"{r['max_dd_pct']:>9.2f}")
    print(f"\nat 10% vol (etf_5bps): ret {out['at_10pct_vol_etf_5bps']['ann_ret_pct']}%  "
          f"maxDD {out['at_10pct_vol_etf_5bps']['max_dd_pct']}%")
    print("\nreference — funnel naive decile book: SR 0.081, DD -37.4%")
    print("reference — house TSMOM edge:          SR 0.60")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
