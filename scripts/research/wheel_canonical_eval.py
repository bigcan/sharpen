"""Eval-1 for the YouTube "wheel" strategy (Cash Flow Academy / Noah) — the canonical,
ZERO-model-risk real-data read.

The wheel = perpetually sell a cash-secured put; if assigned, sell covered calls; repeat.
By put-call parity a short cash-secured put has the SAME payoff as a covered call, so the
wheel is a *perpetual short-volatility income program* on the underlying. The cleanest
real-data proxy for its two legs already exists with decades of history and REAL option
settlement prices (no Black-Scholes reconstruction, no IV assumption, no leak):

  ^PUT  CBOE S&P 500 PutWrite Index   (since 1996) — the "sell cash-secured put" leg
  ^BXM  CBOE S&P 500 BuyWrite Index   (since 1988) — the "covered call" leg
  PUTW  WisdomTree PutWrite ETF       (since 2016) — investable, net of fees
  XYLD  Global X S&P500 Covered Call  (since 2013) — investable, net of fees

All compared to SPY TOTAL return (divs reinvested). These index/ETF versions are the
BEST CASE for the strategy: diversified underlying (no single-name crash risk), perfect
liquidity, systematic ATM writing, collateral earning T-bill interest. If even this best
case fails to beat buy-and-hold on a risk-adjusted, net basis, the single-name retail
wheel in the video (illiquid options, idiosyncratic tail risk, discretionary screen) cannot.

Run:  python scripts/research/wheel_canonical_eval.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from sharpen.crypto.eval import statistics as st
from sharpen.signals.costs import max_drawdown

PPY = 252
_RF_ANN = None


def _rf_daily(index: pd.Index) -> pd.Series:
    """Daily risk-free (13wk T-bill ^IRX) aligned to ``index`` — for EXCESS-return Sharpe.
    rf=0 Sharpe systematically flatters lower-return strategies (the wheel); excess is correct."""
    global _RF_ANN
    if _RF_ANN is None:
        s = yf.Ticker("^IRX").history(period="max")["Close"] / 100.0
        s.index = pd.to_datetime(s.index).tz_localize(None)
        _RF_ANN = s.clip(lower=0.0)
    return (_RF_ANN.reindex(index, method="ffill").fillna(0.02)) / PPY


def fetch(tickers: list[str]) -> pd.DataFrame:
    """Adjusted-close (total-return) daily series, inner-joined on common dates."""
    out = {}
    for t in tickers:
        h = yf.Ticker(t).history(period="max", auto_adjust=True)
        if len(h):
            out[t] = h["Close"]
    df = pd.DataFrame(out).dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def metrics(rets: pd.Series) -> dict:
    s = rets.dropna()
    r = s.to_numpy(np.float64)
    if r.size < 30:
        return {}
    n_years = r.size / PPY
    cagr = float(np.prod(1.0 + r) ** (1.0 / n_years) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(PPY))
    ex = (s - _rf_daily(s.index)).to_numpy(np.float64)   # EXCESS over T-bill (correct Sharpe)
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": st.sharpe_ratio(ex, periods_per_year=PPY),
        "Sortino": st.sortino_ratio(ex, periods_per_year=PPY),
        "MaxDD": max_drawdown(r),
        "Calmar": cagr / abs(max_drawdown(r)) if max_drawdown(r) < 0 else np.nan,
        "n_days": int(r.size),
    }


def capture(strat: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Up/down capture vs benchmark on common days (ratio of mean returns on up/down bench days)."""
    j = pd.concat([strat, bench], axis=1).dropna()
    s, b = j.iloc[:, 0], j.iloc[:, 1]
    up = b > 0
    dn = b < 0
    uc = s[up].mean() / b[up].mean() if up.any() and b[up].mean() != 0 else np.nan
    dc = s[dn].mean() / b[dn].mean() if dn.any() and b[dn].mean() != 0 else np.nan
    return float(uc), float(dc)


def table(px: pd.DataFrame, cols: list[str], label: str, bench: str = "SPY") -> pd.DataFrame:
    rets = px[cols].pct_change()
    rows = {}
    for c in cols:
        m = metrics(rets[c])
        if c != bench and bench in cols:
            uc, dc = capture(rets[c], rets[bench])
            m["UpCap"] = uc
            m["DnCap"] = dc
        rows[c] = m
    out = pd.DataFrame(rows).T
    print(f"\n{'='*92}\n{label}   ({px[cols].dropna().index.min().date()} -> {px[cols].dropna().index.max().date()})\n{'='*92}")
    fmt = out.copy()
    for col in ["CAGR", "Vol", "MaxDD"]:
        if col in fmt:
            fmt[col] = (fmt[col] * 100).map(lambda x: f"{x:6.2f}%" if pd.notna(x) else "   -")
    for col in ["Sharpe", "Sortino", "Calmar", "UpCap", "DnCap"]:
        if col in fmt:
            fmt[col] = fmt[col].map(lambda x: f"{x:6.2f}" if pd.notna(x) else "   -")
    print(fmt.to_string())
    return out


def main() -> None:
    print("Fetching real total-return series (yfinance, auto_adjust=True) ...")
    px = fetch(["SPY", "^PUT", "^BXM", "PUTW", "XYLD", "QYLD"])
    px = px.ffill(limit=2)

    # --- A. CBOE indices vs SPY, full common history (1996+, the 'sell-put' leg has real
    #        SPX settlement prices; longest clean record) ---
    idx = px[["SPY", "^PUT", "^BXM"]].dropna()
    table(idx, ["SPY", "^PUT", "^BXM"], "A. CBOE PutWrite/BuyWrite vs SPY  (FULL COMMON PERIOD)")

    # Synthetic 'full wheel' = 50/50 daily-rebalanced blend of the put-leg and call-leg
    # (a put-writer who is half the time in cash-secured puts, half assigned writing calls).
    wheel = 0.5 * idx["^PUT"].pct_change() + 0.5 * idx["^BXM"].pct_change()
    blend = pd.DataFrame({"SPY": idx["SPY"].pct_change(), "WHEEL_50_50": wheel}).dropna()
    rows = {c: metrics(blend[c]) for c in blend}
    for c in blend:
        if c != "SPY":
            uc, dc = capture(blend[c], blend["SPY"])
            rows[c]["UpCap"], rows[c]["DnCap"] = uc, dc
    print(f"\n{'='*92}\nA'. 50/50 PUT+BXM 'full wheel' proxy vs SPY  (same period)\n{'='*92}")
    bt = pd.DataFrame(rows).T
    for col in ["CAGR", "Vol", "MaxDD"]:
        bt[col] = (bt[col] * 100).map(lambda x: f"{x:6.2f}%" if pd.notna(x) else "  -")
    for col in ["Sharpe", "Sortino", "Calmar", "UpCap", "DnCap"]:
        bt[col] = bt[col].map(lambda x: f"{x:6.2f}" if pd.notna(x) else "  -")
    print(bt.to_string())

    # --- B. Sub-period regimes (does the premium cushion DDs / does it lag in bulls?) ---
    regimes = {
        "GFC 2007-2009": ("2007-10-01", "2009-03-31"),
        "Bull 2009-2020": ("2009-04-01", "2020-02-19"),
        "COVID crash 2020": ("2020-02-19", "2020-03-23"),
        "2022 bear": ("2022-01-01", "2022-10-12"),
        "2023-2026 recovery": ("2023-01-01", "2026-06-20"),
    }
    print(f"\n{'='*92}\nB. REGIME BREAKDOWN — total return over each window (%)\n{'='*92}")
    reg_rows = {}
    for name, (a, b) in regimes.items():
        seg = idx.loc[a:b]
        if len(seg) < 5:
            continue
        reg_rows[name] = {c: (seg[c].iloc[-1] / seg[c].iloc[0] - 1.0) * 100 for c in idx.columns}
    reg = pd.DataFrame(reg_rows).T
    print(reg.round(2).to_string())

    # --- C. Investable ETFs (net of real fees) vs SPY, post-2013 ---
    etf = px[["SPY", "PUTW", "XYLD", "QYLD"]].dropna()
    table(etf, ["SPY", "PUTW", "XYLD", "QYLD"], "C. Investable covered-call/put-write ETFs (NET OF FEES) vs SPY")

    # --- D. The honest 'cash-flow ROI' deception check: distribution yield vs total return ---
    print(f"\n{'='*92}\nD. NOTE ON 'CASH-FLOW ROI'\n{'='*92}")
    print("XYLD/QYLD pay ~10-12%/yr distributions ('cash flow') yet their NAV total return")
    print("(table C, auto_adjust reinvests those distributions) is what actually matters.")
    print("A high distribution yield financed by capped upside is return-of-capital, not alpha.")


if __name__ == "__main__":
    main()
