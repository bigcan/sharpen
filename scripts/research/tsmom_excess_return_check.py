"""TSMOM survivor, restated in EXCESS of cash — the number the headline 0.601 omits.

``xsec_momentum_falsification.py`` reports the pooled TSMOM book's Sharpe on TOTAL returns: the book's
cash is assumed to earn nothing and no borrow is charged on shorts. A self-financing ETF book with net
exposure ``e_t`` has excess return ``r_book - e_t * rf_t``. This script restates the SAME book (same
signal, weights, costs and cached prices) three ways:

  1. total return (reproduces the 0.601 headline),
  2. in excess of 13-week T-bills (^IRX) on the net exposure,
  3. (2) minus an assumed 50 bp/yr borrow fee on the short leg.

Run ``xsec_momentum_falsification.py`` first (it downloads and caches the prices).

    python scripts/research/tsmom_excess_return_check.py

Recorded 2026-09-18: 0.601 / 0.511 / 0.485; excess-of-cash by era 2006-09 0.455, 2010-15 0.753,
2016-20 0.189, 2021-26 0.609. Separately, the same frozen rule on 32 ETFs never used in development
scored 0.389 (docs/research/fable_verdict_2026-06-11.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import xsec_momentum_falsification as xm  # noqa: E402

BORROW_BPS_PER_YEAR = 50.0      # assumption, stated in the docstring; not a gate


def main() -> int:
    import yfinance as yf

    close = xm.get_prices()[xm.ALL_TICKERS]
    rets = close.pct_change()
    rebal = xm.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(xm.LOOKBACKS) + xm.SKIP + xm.VOL_WIN]]
    w = xm.vol_scaled_weights(xm.tsmom_signal(close, rebal), rets, rebal)
    held = w.reindex(rets.index).ffill().fillna(0).shift(1).fillna(0)   # weights in force on day t
    gross, cost, _ = xm.backtest(w, rets)
    net = gross - cost["standard_2bps"]

    irx = yf.download("^IRX", start=str(close.index[0].date()), progress=False,
                      auto_adjust=False)["Close"].squeeze()
    rf = (irx / 100.0 / 252.0).reindex(rets.index).ffill().fillna(0.0)
    excess = net - held.sum(axis=1) * rf
    short_gross = held.clip(upper=0).abs().sum(axis=1)
    excess_borrow = excess - short_gross * (BORROW_BPS_PER_YEAR / 1e4) / 252.0

    print(f"pooled TSMOM net Sharpe, total return        {xm.sharpe(net):.3f}")
    print(f"  in excess of T-bills on net exposure        {xm.sharpe(excess):.3f}")
    print(f"  ... and {BORROW_BPS_PER_YEAR:.0f} bp/yr borrow on shorts           "
          f"{xm.sharpe(excess_borrow):.3f}")
    k = 0.10 / (net.std() * np.sqrt(252))
    live = held.abs().sum(axis=1) > 0
    print(f"at a 10% vol target: mean gross {float((held.abs().sum(axis=1) * k)[live].mean()):.2f}x, "
          f"mean net long {float((held.sum(axis=1) * k)[live].mean()):.2f}x")
    for a, b in [("2006", "2009"), ("2010", "2015"), ("2016", "2020"), ("2021", "2026")]:
        print(f"  excess-of-cash Sharpe {a}-{b}: {xm.sharpe(excess.loc[a:b]):.3f}")
    return 0


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    raise SystemExit(main())
