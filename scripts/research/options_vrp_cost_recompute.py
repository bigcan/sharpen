"""V4 audit probe: reproduce verdict.json from cache, then quantify cost-model issues.

Read-only w.r.t. repo: writes nothing into the repo, only prints.
Toggles:
  fix_close_args   : close-fee V computed with CORRECT (sigma, tau) order
  charge_roll_hedge: charge perp taker fee on the hedge transition at roll
                     (|q_new_init - q_old_last|) and on the initial hedge at open
"""
import sys, math, json
sys.path.insert(0, r"C:\FinRL\FinRL-Pro_DS")
import numpy as np

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import options_array_builder as oab
from finrl_pro_ds.crypto.options_pricing import (
    straddle_delta, straddle_price, straddle_vega)

ANN = 365.0

class Cfg:
    premium_frac = 0.05
    roll_days = 21
    entry_tenor_days = 30
    initial_capital = 100_000.0
    option_fee_pct_underlying = 0.0003
    option_fee_cap_pct_premium = 0.125
    perp_taker_fee = 0.0005
    option_spread_vol_pts = 1.0
    frictionless = False

def simulate(spot, iv, funding, cfg, fix_close_args=False, charge_roll_hedge=False):
    T = len(spot)
    equity = cfg.initial_capital
    daily_pnl = np.zeros(T)
    eq_curve = np.full(T, cfg.initial_capital, float)
    n = 0.0; K = 0.0; tau = 0.0; q_prev = 0.0; days_held = 0
    tau_step = 1.0 / ANN
    # instrumentation
    tot = dict(open_fee=0.0, open_spread=0.0, close_fee=0.0, close_spread=0.0,
               rehedge=0.0, funding=0.0, roll_hedge_omitted=0.0, n_rolls=0,
               close_cap_bound=0, close_under_bound=0, aged_absdelta=[])

    def option_fees(nn, S, prem_total):
        if cfg.frictionless: return 0.0, "fri"
        fee_u = 2.0 * nn * cfg.option_fee_pct_underlying * S
        fee_c = cfg.option_fee_cap_pct_premium * prem_total
        return (min(fee_u, fee_c), ("cap" if fee_c < fee_u else "und"))

    def spread_cost(nn, S, sigma, tt):
        if cfg.frictionless: return 0.0
        vega = straddle_vega(S, S, sigma, tt)
        return nn * vega * (cfg.option_spread_vol_pts / 100.0) * 0.5

    def open_straddle(t):
        nonlocal n, K, tau, equity, q_prev
        S, sigma = spot[t], iv[t]
        K = S; tau = cfg.entry_tenor_days / ANN
        unit_prem = straddle_price(S, K, sigma, tau)
        gross = cfg.premium_frac * equity
        n = gross / unit_prem if unit_prem > 0 else 0.0
        prem_total = n * unit_prem
        fee, _ = option_fees(n, S, prem_total)
        spr = spread_cost(n, S, sigma, tau)
        tot["open_fee"] += fee; tot["open_spread"] += spr
        cost = fee + spr
        q_old = q_prev
        q_new = n * straddle_delta(S, K, sigma, tau)
        if charge_roll_hedge and not cfg.frictionless:
            hedge_fee = cfg.perp_taker_fee * abs(q_new - q_old) * S
            cost += hedge_fee
        else:
            tot["roll_hedge_omitted"] += cfg.perp_taker_fee * abs(q_new - q_old) * S
        equity -= cost
        daily_pnl[t] -= cost
        q_prev = q_new

    def close_straddle(t):
        nonlocal n, equity
        if n <= 0: return
        S, sigma = spot[t], iv[t]
        if fix_close_args:
            V = straddle_price(S, K, sigma, max(tau, tau_step))
        else:
            V = straddle_price(S, K, max(tau, tau_step), sigma)  # repo's swapped order
        fee, bind = option_fees(n, S, n * V)
        spr = spread_cost(n, S, sigma, max(tau, tau_step))
        tot["close_fee"] += fee; tot["close_spread"] += spr
        tot["close_cap_bound" if bind == "cap" else "close_under_bound"] += 1
        tot["aged_absdelta"].append(abs(straddle_delta(S, K, sigma, max(tau, tau_step))))
        equity -= fee + spr
        daily_pnl[t] -= fee + spr
        n = 0.0

    t0 = 0
    while t0 < T and (not np.isfinite(spot[t0]) or not np.isfinite(iv[t0]) or iv[t0] <= 0):
        t0 += 1
    open_straddle(t0); days_held = 0
    for t in range(t0, T - 1):
        S_t, sig_t = spot[t], iv[t]
        S_n, sig_n = spot[t + 1], iv[t + 1]
        if not (np.isfinite(S_n) and np.isfinite(sig_n) and sig_n > 0):
            S_n, sig_n = S_t, sig_t
        q_t = n * straddle_delta(S_t, K, sig_t, max(tau, tau_step))
        rehedge_cost = 0.0 if cfg.frictionless else cfg.perp_taker_fee * abs(q_t - q_prev) * S_t
        tot["rehedge"] += rehedge_cost
        V_t = straddle_price(S_t, K, sig_t, max(tau, tau_step))
        tau_next = max(tau - tau_step, tau_step)
        V_n = straddle_price(S_n, K, sig_n, tau_next)
        option_pnl = n * (V_t - V_n)
        hedge_pnl = q_t * (S_n - S_t)
        funding_pnl = 0.0 if cfg.frictionless else -q_t * S_t * funding[t]
        tot["funding"] += funding_pnl
        pnl = option_pnl + hedge_pnl - rehedge_cost + funding_pnl
        daily_pnl[t + 1] += pnl
        equity += pnl
        eq_curve[t + 1] = equity
        q_prev = q_t
        tau = tau_next
        days_held += 1
        if days_held >= cfg.roll_days or tau <= (cfg.entry_tenor_days - cfg.roll_days - 1) / ANN:
            close_straddle(t + 1)
            tot["n_rolls"] += 1
            if equity > 0:
                open_straddle(t + 1)
            days_held = 0
            eq_curve[t + 1] = equity
    return daily_pnl, eq_curve, tot

def sharpe(dr):
    d = dr[np.isfinite(dr)]
    return float(d.mean() / d.std(ddof=1) * math.sqrt(ANN)) if d.std(ddof=1) > 0 else 0.0

def pf(p):
    pos = p[p > 0].sum(); neg = -p[p < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")

def mdd(eq):
    pk = np.maximum.accumulate(eq); return float((1 - eq / pk).max())

def rets(p, eq):
    prev = np.concatenate([[eq[0]], eq[:-1]]); prev = np.where(prev <= 0, np.nan, prev)
    return p / prev

raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
panels = oab.build_panels(raw)
i = panels.assets.index("BTC")
spot, iv, fund = panels.spot_ary[:, i], panels.iv_ary[:, i], panels.funding_ary[:, i]

print("== timestamps sanity ==")
print("dvol idx head:", list(raw.dvol['BTC'].index[:2]), "tail:", raw.dvol['BTC'].index[-1])
print("perp idx head:", list(raw.perp['BTC'].index[:2]), "tail:", raw.perp['BTC'].index[-1])
print("funding idx head:", list(raw.funding['BTC'].index[:2]))
d = panels.dates
print("panel:", d[0], "->", d[-1], "T =", len(d), "max gap days:",
      int(np.diff(d.values).max() / np.timedelta64(1, 'D')))

for label, kw in [("repo-as-is", {}),
                  ("fix close args", dict(fix_close_args=True)),
                  ("fix close + charge roll-hedge fee", dict(fix_close_args=True, charge_roll_hedge=True))]:
    for roll in (21, 30):
        cfg = Cfg(); cfg.roll_days = roll
        p, eq, tot = simulate(spot, iv, fund, cfg, **kw)
        r = rets(p, eq)
        ad = np.array(tot.pop("aged_absdelta"))
        print(f"\n[{label} | roll={roll}] BTC net Sharpe={sharpe(r):.4f} PF={pf(p):.4f} "
              f"ret={eq[-1]/cfg.initial_capital-1:.4%} DD={mdd(eq):.4%}")
        print("   costs $: open_fee={open_fee:,.0f} open_spr={open_spread:,.0f} "
              "close_fee={close_fee:,.0f} close_spr={close_spread:,.0f} "
              "rehedge={rehedge:,.0f} funding_pnl={funding:,.0f} "
              "omitted_roll_hedge_fee={roll_hedge_omitted:,.0f} rolls={n_rolls}".format(**tot))
        print(f"   close fee binding: cap={tot['close_cap_bound']} underlying={tot['close_under_bound']}; "
              f"aged |delta| mean={ad.mean():.3f} p90={np.percentile(ad,90):.3f}")
