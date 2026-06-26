"""Evaluate the 3 0DTE option strategies from SMB Capital's "Top 3 0DTE Options
Trading Strategies" (youtube UG4f752OXq8): #1 directional CREDIT SPREAD,
#2 IRON CONDOR, #3 CALENDAR SPREAD.

Method == the falsify-first / cost-aware / vs-baseline harness used for the wheel
(cont-76) and XLG (cont-75) evals. Doctrine: REAL-SETTLEMENT indices and LIVE
option-income FUNDS are the authority (zero model risk, net of fees); a sim's
absolute edge is only an artifact of the assumed VRP/skew/term-structure.

  * Excess-over-RF Sharpe (rf from ^IRX) -- the /audit fix from the wheel eval.
  * Total return (yfinance auto_adjust=True -> distributions reinvested). Critical:
    these funds pay 20-100% "yields" that are mostly return-of-capital/NAV erosion;
    only total return is honest.
  * MATCHED windows: every fund vs its own index baseline over IDENTICAL dates.
  * vs-baseline active Information-Ratio + t-stat (|t|>2 == significant).

Mapping to real instruments:
  #1 Credit spread -> LIVE 0DTE funds that sell 0DTE credit spreads / income daily:
       Defiance JEPY/QQQY/IWMY (0DTE put credit spreads), Roundhill XDTE/QDTE/RDTE,
       Defiance SPYT/QQQT. Net of fees, professionally executed == best-case for retail.
  #2 Iron condor -> short-premium ENGINE (an IC is short-put-spread + short-call-spread):
       ^PUT, ^BXM (decades), PUTW, XYLD, QYLD. If neither half beats B&H, the
       wing-capped combination cannot either. (CBOE CNDR index not on yfinance.)
  #3 Calendar -> no public fund/index; structural (long back-month vega / short 0DTE
       gamma; a sim would merely echo the assumed term structure -> not authoritative).

Run:  python scripts/research/zerodte_3strategies_eval.py

DATA CAVEATS (from the cont-81 /audit — do not re-derive the two errors they fix):
  * JEPY: yfinance total-return carries an uncorrected reverse-split jump (+201%/day);
    its printed gap is spuriously POSITIVE -> excluded from all pooled tests. (QQQY/IWMY
    had the SAME 1:3 split but were adjusted correctly -- max daily 3.4%/3.5% -- so only
    JEPY is corrupt.)
  * ^PUT: two anomalous COVID-2020 days (2020-03-13 +35.3% / 03-16 -28.4%, implausible for
    a put-write index) DRIVE its -0.03 gap; de-glitched ^PUT = +0.035 vs SPY, i.e. the
    put-write index ~TIES the S&P. Do NOT cite ^PUT as condor underperformance -- the
    Strategy-#2 signal is carried by ^BXM/PUTW/XYLD/QYLD, and the pooled #2 result is
    robust to dropping ^PUT entirely (-3.9%/yr, t -2.35). Underperformance significance is
    autocorrelation-robust (Newey-West t ~ -2.5; block-bootstrap P(active>=0) 0.002/0.009).
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TICKERS = [
    "SPY", "QQQ", "IWM", "^IRX",
    "^PUT", "^BXM", "PUTW", "XYLD", "QYLD",                    # short-premium family (#2)
    "JEPY", "QQQY", "IWMY", "XDTE", "QDTE", "RDTE", "SPYT", "QQQT",  # live 0DTE funds (#1)
    "JEPI", "JEPQ", "SVOL",                                    # context
]


def load_prices() -> pd.DataFrame:
    out = {}
    for t in TICKERS:
        d = yf.download(t, period="max", interval="1d", progress=False, auto_adjust=True)
        if d is not None and len(d):
            out[t] = d["Close"].squeeze()
    return pd.DataFrame(out).sort_index()


def excess_sharpe(daily_excess: pd.Series) -> float:
    r = daily_excess.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan


def max_drawdown(daily_total: pd.Series) -> float:
    c = (1 + daily_total.dropna()).cumprod()
    return float((c / c.cummax() - 1).min())


def per_fund_row(px: pd.DataFrame, rf: pd.Series, fund: str, base: str, tag: str) -> dict | None:
    sub = px[[fund, base]].dropna()
    if len(sub) < 60:
        return None
    rfn = sub[fund].pct_change().clip(-0.4, 0.4)   # clip kills split-adjustment artifacts (e.g. JEPY +201%)
    rbn = sub[base].pct_change().clip(-0.4, 0.4)
    rfx = rf.reindex(sub.index).fillna(0.0)
    fe, be = (rfn - rfx), (rbn - rfx)
    active = (rfn - rbn).dropna()
    sd = active.std(ddof=1)
    return dict(
        tag=tag, fund=fund, base=base, start=str(sub.index[0].date()),
        yrs=round(len(sub) / 252, 1),
        sh_fund=excess_sharpe(fe), sh_base=excess_sharpe(be),
        dsh=excess_sharpe(fe) - excess_sharpe(be),
        cagr_f=(1 + rfn.add(rfx, fill_value=0)).prod() ** (252 / len(sub)) - 1,
        cagr_b=(1 + rbn.add(rfx, fill_value=0)).prod() ** (252 / len(sub)) - 1,
        dd_f=max_drawdown(rfn), dd_b=max_drawdown(rbn),
        ir=(active.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        t=(active.mean() / sd * np.sqrt(len(active))) if sd > 0 else np.nan,
    )


def pooled(px: pd.DataFrame, rf: pd.Series, funds: list[str], bases: list[str], label: str) -> None:
    acts, shf, shb = [], [], []
    for f, b in zip(funds, bases):
        sub = px[[f, b]].dropna()
        rfn, rbn = sub[f].pct_change().clip(-0.4, 0.4), sub[b].pct_change().clip(-0.4, 0.4)
        rfx = rf.reindex(sub.index).fillna(0.0)
        shf.append(excess_sharpe(rfn - rfx))
        shb.append(excess_sharpe(rbn - rfx))
        acts.append((rfn - rbn).dropna().rename(f))
    comp = pd.concat(acts, axis=1).mean(axis=1).dropna()      # equal-weight basket active return
    sd = comp.std(ddof=1)
    ir = comp.mean() / sd * np.sqrt(252)
    tstat = comp.mean() / sd * np.sqrt(len(comp))
    print(f"=== {label} ===")
    print(f"  funds: {funds}")
    print(f"  mean fund excess-Sharpe {np.mean(shf):.2f} vs baseline {np.mean(shb):.2f} "
          f"(gap {np.mean(shf) - np.mean(shb):+.2f}) | NEG-gap funds {sum(np.array(shf) < np.array(shb))}/{len(funds)}")
    print(f"  EW-basket active: ann {comp.mean() * 252:+.2%}  IR {ir:+.2f}  t {tstat:+.2f}  (n={len(comp)})\n")


def main() -> None:
    px = load_prices()
    rf = (px["^IRX"] / 100 / 252).reindex(px.index).ffill().fillna(0.0)

    jobs = [
        ("#2cond", "^PUT", "SPY"), ("#2cond", "^BXM", "SPY"), ("#2cond", "PUTW", "SPY"),
        ("#2cond", "XYLD", "SPY"), ("#2cond", "QYLD", "QQQ"),
        ("#1cspd", "JEPY", "SPY"), ("#1cspd", "QQQY", "QQQ"), ("#1cspd", "IWMY", "IWM"),
        ("#1cspd", "XDTE", "SPY"), ("#1cspd", "QDTE", "QQQ"), ("#1cspd", "RDTE", "IWM"),
        ("#1cspd", "SPYT", "SPY"), ("#1cspd", "QQQT", "QQQ"),
        ("ctx", "JEPI", "SPY"), ("ctx", "JEPQ", "QQQ"), ("ctx", "SVOL", "SPY"),
    ]
    rows = [r for j in jobs if (r := per_fund_row(px, rf, j[1], j[2], j[0]))]
    df = pd.DataFrame(rows)
    for c in ["sh_fund", "sh_base", "dsh", "ir", "t"]:
        df[c] = df[c].round(2)
    for c in ["cagr_f", "cagr_b", "dd_f", "dd_b"]:
        df[c] = (df[c] * 100).round(1)
    pd.set_option("display.width", 200)
    print(df[["tag", "fund", "base", "start", "yrs", "sh_fund", "sh_base", "dsh",
              "cagr_f", "cagr_b", "dd_f", "dd_b", "ir", "t"]].to_string(index=False))
    print("\nLegend: sh_=excess-Sharpe  dsh=fund-baseline gap(NEG=worse)  cagr_/dd_=% total/maxDD"
          "  ir=active info-ratio  t=active t-stat(|t|>2 sig). JEPY cagr is split-artifact -> excluded from pooled.\n")

    # Pooled headline tests (robust to short single-fund history)
    pooled(px, rf, ["QQQY", "IWMY", "XDTE", "QDTE", "RDTE", "SPYT", "QQQT"],
           ["QQQ", "IWM", "SPY", "QQQ", "IWM", "SPY", "QQQ"],
           "STRATEGY #1  Live 0DTE credit-spread/income funds vs buy&hold (JEPY excluded)")
    pooled(px, rf, ["^PUT", "^BXM", "PUTW", "XYLD", "QYLD"], ["SPY", "SPY", "SPY", "SPY", "QQQ"],
           "STRATEGY #2  Short-premium family (iron-condor engine) vs buy&hold")


if __name__ == "__main__":
    main()
