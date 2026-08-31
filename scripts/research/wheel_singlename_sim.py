"""Eval-2 for the YouTube "wheel" strategy — a FAITHFUL single-name simulation that tests
the video's specific mechanics and its headline "15.5% cash-flow ROI" claim on KO.

Mechanics (matched to the video, Cash Flow Academy / Noah):
  * Monthly cycle on the 3rd-Friday option expiries (~30-35 DTE), ~12 trades/yr.
  * CASH state  -> sell a cash-secured PUT.  Strike = ~30-delta (video quoted a 33.8%
                  assignment probability -> 30-delta put). If S_expiry < K -> assigned 100 sh.
  * STOCK state -> sell a covered CALL.  Strike = ~15-delta (video quoted ~15% call prob).
                  Collect dividends while held. If S_expiry > K -> called away.
  * Pricing: Black-Scholes-Merton, IV = trailing-21d realized vol x VRP_MULT (single-name
            VRP is small, so 1.15 is generous to the strategy). Strikes by inverting BS delta.
  * Costs: $0.65/contract commission + a premium-slippage haircut (the video fills at MID,
           = 0% haircut; realistic single-name 35-DTE fills are worse).
  * Cash collateral earns the 13-week T-bill yield (^IRX) — the put-write index does too.

Benchmarks: buy-&-hold the SAME name (total return) and SPY (total return), same start capital.
Headline metric is TOTAL NAV return / Sharpe / MaxDD — NOT "cash-flow ROI" (which books premium
while hiding the capped upside and the long-stock mark-to-market).

Run:  PYTHONPATH=. python scripts/research/wheel_singlename_sim.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from sharpen.crypto.eval import statistics as st
from sharpen.signals.costs import max_drawdown

_SPY_CACHE: dict = {}

# ---------- Black-Scholes-Merton ----------
def _d1(S, K, T, r, q, sig):
    return (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))

def bs_put(S, K, T, r, q, sig):
    d1 = _d1(S, K, T, r, q, sig); d2 = d1 - sig * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_call(S, K, T, r, q, sig):
    d1 = _d1(S, K, T, r, q, sig); d2 = d1 - sig * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_delta(S, K, T, r, q, sig):
    return np.exp(-q * T) * norm.cdf(-_d1(S, K, T, r, q, sig))

def call_delta(S, K, T, r, q, sig):
    return np.exp(-q * T) * norm.cdf(_d1(S, K, T, r, q, sig))

def put_iv(S0, K, base_sig, skew):
    """Strike-dependent put IV with a linear volatility SKEW: OTM puts (K<S0) priced richer.
    skew=0.5 -> +5 vol pts at 10% OTM (typical single-name). Flat model (skew=0) underprices
    the deep-OTM protective puts a hedge BUYS, so this is needed to price bought protection."""
    m = max((S0 - K) / S0, 0.0)            # OTM fraction (puts)
    return base_sig + skew * m


def strike_for_delta(S, T, r, q, sig, target_delta, kind):
    grid = np.arange(round(S * 0.5), round(S * 1.5) + 1, 1.0)
    grid = grid[grid > 0]
    fn = put_delta if kind == "put" else call_delta
    d = np.array([fn(S, k, T, r, q, sig) for k in grid])
    return float(grid[np.argmin(np.abs(d - target_delta))])


def third_fridays(start, end):
    out = []
    for mth in pd.date_range(start, end, freq="MS"):
        fris = pd.date_range(mth, mth + pd.offsets.MonthEnd(0), freq="W-FRI")
        if len(fris) >= 3:
            out.append(fris[2])
    return pd.DatetimeIndex(out)


def load(ticker, start):
    tk = yf.Ticker(ticker)
    h = tk.history(start=start, auto_adjust=False, actions=True)
    h.index = pd.to_datetime(h.index).tz_localize(None)
    tr = tk.history(start=start, auto_adjust=True)["Close"]
    tr.index = pd.to_datetime(tr.index).tz_localize(None)
    return h["Close"], h["Dividends"].fillna(0.0), tr


def spy_tr(start):
    if start not in _SPY_CACHE:
        s = yf.Ticker("SPY").history(start=start, auto_adjust=True)["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        _SPY_CACHE[start] = s
    return _SPY_CACHE[start]


def realized_vol(px, win=21):
    return (np.log(px / px.shift(1)).rolling(win).std() * np.sqrt(252)).bfill()


def cash_yield():
    if "irx" not in _SPY_CACHE:
        irx = yf.Ticker("^IRX").history(period="max")["Close"] / 100.0
        irx.index = pd.to_datetime(irx.index).tz_localize(None)
        _SPY_CACHE["irx"] = irx.clip(lower=0.0)
    return _SPY_CACHE["irx"]


def simulate(ticker, start, put_delta_t=0.30, call_delta_t=0.15, vrp_mult=1.15,
             slip=0.05, commission=0.65, prot_put_otm=None, cash_put_otm=None, put_skew=0.0):
    """Full-history wheel. Returns NAV path + benchmarks + a per-expiry ledger.

    ``prot_put_otm`` (e.g. 0.04) = the video's "secret sauce": while holding stock, BUY a
    protective put ``prot_put_otm`` below spot each cycle (covered-call + long put = a COLLAR).
    ``cash_put_otm`` = the HONEST COMPLETE hedge: also buy a deeper protective put while SHORT
    the cash-secured put (turns it into a bull put SPREAD), closing the assignment-phase coverage
    gap. Both legs are paid at the ASK ⇒ the wheel pays the VRP it sells (in both phases).
    """
    px, divs, tr = load(ticker, start)
    rv = realized_vol(px) * vrp_mult
    irx = cash_yield().reindex(px.index, method="ffill").fillna(0.02)
    exps = third_fridays(px.index.min(), px.index.max())
    exps = pd.DatetimeIndex([px.index[px.index.get_indexer([e], method="nearest")[0]] for e in exps]).unique()
    exps = exps[(exps >= px.index.min()) & (exps <= px.index.max())]

    cap0 = 100.0 * float(px.loc[exps[0]])
    cash = cap0; shares = 0; state = "cash"
    led = []
    for i in range(len(exps) - 1):
        t0, t1 = exps[i], exps[i + 1]
        S0, S1 = float(px.loc[t0]), float(px.loc[t1])
        T = max((t1 - t0).days, 1) / 365.0
        sig = float(max(rv.loc[t0], 0.06)); r = float(irx.loc[t0])
        q = float(divs.loc[t0 - pd.Timedelta(days=365):t0].sum() / S0)
        cash *= (1.0 + r * T)
        prem = 0.0; dvd = 0.0; event = ""
        if state == "cash":
            K = strike_for_delta(S0, T, r, q, sig, put_delta_t, "put")
            prem = bs_put(S0, K, T, r, q, put_iv(S0, K, sig, put_skew)) * (1.0 - slip) * 100.0 - commission
            cash += prem
            if cash_put_otm is not None:                         # buy deeper put -> bull put SPREAD
                Kpc = min(round(S0 * (1.0 - cash_put_otm)), K - 1.0)  # must sit below the short strike
                cost = bs_put(S0, Kpc, T, r, q, put_iv(S0, Kpc, sig, put_skew)) * (1.0 + slip) * 100.0 + commission
                cash -= cost
                cash += 100.0 * max(Kpc - S1, 0.0)               # cash-settled assignment-tail insurance
                prem -= cost
            if S1 < K:
                cash -= 100.0 * K; shares = 100; state = "stock"; event = "assigned"
        else:
            K = strike_for_delta(S0, T, r, q, sig, call_delta_t, "call")
            prem = bs_call(S0, K, T, r, q, sig) * (1.0 - slip) * 100.0 - commission
            cash += prem
            if prot_put_otm is not None:                         # buy protective put (collar)
                Kp = round(S0 * (1.0 - prot_put_otm))
                cost = bs_put(S0, Kp, T, r, q, put_iv(S0, Kp, sig, put_skew)) * (1.0 + slip) * 100.0 + commission
                cash -= cost                                     # pay the ASK
                cash += 100.0 * max(Kp - S1, 0.0)                # long-put payoff at expiry
                prem -= cost                                     # ledger: net option premium
            win = divs.loc[(divs.index > t0) & (divs.index <= t1)]  # half-open (t0,t1] — no boundary double-count
            dvd = float(win[win > 0].sum()) * 100.0
            cash += dvd
            if S1 > K:
                cash += 100.0 * K; shares = 0; state = "cash"; event = "called_away"
        led.append(dict(date=t1, nav=cash + shares * S1, prem=prem, dvd=dvd, event=event))

    ledger = pd.DataFrame(led).set_index("date")
    nav = ledger["nav"]
    bh = tr.reindex(nav.index, method="ffill"); bh = cap0 * bh / float(tr.loc[exps[0]])
    sp = spy_tr(start).reindex(nav.index, method="ffill"); sp = cap0 * sp / float(sp.iloc[0])
    return dict(ticker=ticker, nav=nav, bh=bh, spy=sp, cap0=cap0, ledger=ledger)


def metrics(series):
    ret = series.pct_change().dropna()
    r = ret.to_numpy()
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    # EXCESS-return Sharpe/Sortino (subtract per-period T-bill) — rf=0 flatters the cash-holding wheel
    rf_ann = cash_yield().reindex(ret.index, method="ffill").fillna(0.02)
    dt = pd.Series(ret.index).diff().dt.days.to_numpy()[1:] / 365.0
    ex = r.copy(); ex[1:] = r[1:] - rf_ann.to_numpy()[1:] * dt
    return dict(TotRet=series.iloc[-1] / series.iloc[0] - 1.0,
                CAGR=(series.iloc[-1] / series.iloc[0]) ** (1 / yrs) - 1.0 if yrs > 0 else np.nan,
                Sharpe=st.sharpe_ratio(ex, periods_per_year=12),
                Sortino=st.sortino_ratio(ex, periods_per_year=12),
                MaxDD=max_drawdown(r))


def report(res, title, window=None):
    nav, bh, spy, led = res["nav"], res["bh"], res["spy"], res["ledger"]
    if window:
        a, b = window
        nav, bh, spy = nav.loc[a:b], bh.loc[a:b], spy.loc[a:b]
        led = led.loc[a:b]
        base = float(bh.iloc[0])                       # rebase all to common start at window open
        nav = base * nav / nav.iloc[0]; bh = base * bh / bh.iloc[0]; spy = base * spy / spy.iloc[0]
        cap0 = base
    else:
        cap0 = res["cap0"]
    print(f"\n{'='*90}\n{title}: {res['ticker']}  "
          f"({nav.index.min().date()} -> {nav.index.max().date()}, start ${cap0:,.0f})\n{'='*90}")
    df = pd.DataFrame({"WHEEL": metrics(nav), f"B&H {res['ticker']}": metrics(bh),
                       "B&H SPY": metrics(spy)}).T
    df["TotRet"] = (df["TotRet"] * 100).map(lambda x: f"{x:7.1f}%")
    df["CAGR"] = (df["CAGR"] * 100).map(lambda x: f"{x:6.2f}%")
    df["MaxDD"] = (df["MaxDD"] * 100).map(lambda x: f"{x:6.1f}%")
    for c in ["Sharpe", "Sortino"]:
        df[c] = df[c].map(lambda x: f"{x:5.2f}")
    print(df.to_string())
    prem, dvd = led["prem"].sum(), led["dvd"].sum()
    n_as = (led["event"] == "assigned").sum(); n_ca = (led["event"] == "called_away").sum()
    print(f"\n  'Cash-flow' booked: ${prem + dvd:,.0f} (premium ${prem:,.0f} + div ${dvd:,.0f}) "
          f"= {100*(prem+dvd)/cap0:.1f}% of start capital   <-- the video's headline metric")
    print(f"  TOTAL return:  WHEEL {100*(nav.iloc[-1]/cap0-1):+.1f}%   "
          f"vs  B&H {res['ticker']} {100*(bh.iloc[-1]/cap0-1):+.1f}%   "
          f"vs  SPY {100*(spy.iloc[-1]/cap0-1):+.1f}%   "
          f"(assign={n_as}, called={n_ca}, trades={len(led)})")


def main():
    print("#" * 90)
    print("# PART 1 — The video's KO live window (May-2025 -> May-2026): does the wheel beat holding KO?")
    print("#" * 90)
    report(simulate("KO", "2024-09-01", slip=0.05), "KO video window", window=("2025-05-01", "2026-05-31"))

    print("\n" + "#" * 90)
    print("# PART 2 — Long multi-year wheel vs buy-and-hold (2015-2026), net of realistic cost")
    print("#" * 90)
    for tk in ["KO", "EOG"]:
        report(simulate(tk, "2015-01-01", slip=0.05), "Long wheel")

    print("\n" + "#" * 90)
    print("# PART 3 — Sensitivity (KO 2015-2026): fill quality & vol-risk-premium")
    print("#" * 90)
    bh_ko = metrics(simulate("KO", "2015-01-01")["bh"])
    print(f"{'config':30s} {'WheelTot':>9s} {'WheelShrp':>10s} {'WheelDD':>8s} | "
          f"B&H KO Tot {100*bh_ko['TotRet']:.0f}%  Shrp {bh_ko['Sharpe']:.2f}")
    for slip, vrp, lbl in [(0.0, 1.15, "MID fill (video assumption)"),
                            (0.05, 1.15, "realistic 5% slippage"),
                            (0.10, 1.15, "harsh 10% slippage"),
                            (0.05, 1.00, "no VRP (iv=rv)"),
                            (0.05, 1.30, "rich VRP (iv=1.3*rv)")]:
        mm = metrics(simulate("KO", "2015-01-01", slip=slip, vrp_mult=vrp)["nav"])
        print(f"{lbl:30s} {100*mm['TotRet']:8.1f}% {mm['Sharpe']:10.2f} {100*mm['MaxDD']:7.1f}%")


if __name__ == "__main__":
    main()
