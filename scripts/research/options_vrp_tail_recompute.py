"""V6 tail-risk audit recomputation for results/options_vrp/verdict.json (BTC).

Eval-only, CPU. Replicates simulate_asset() from
scripts/research/options_vrp_falsification.py with state tracking, verifies
bit-for-bit parity, then runs adversarial stress analytics.
"""
import sys
import math
import importlib.util
import numpy as np
import pandas as pd

ROOT = r"C:\FinRL\FinRL-Pro_DS"
sys.path.insert(0, ROOT)

from finrl_pro_ds.crypto.data import deribit_options_loader as dol
from finrl_pro_ds.crypto.data import options_array_builder as oab
from finrl_pro_ds.crypto.options_pricing import (
    straddle_price, straddle_delta, straddle_vega, ncdf,
)

spec = importlib.util.spec_from_file_location(
    "ovf_script", ROOT + r"\scripts\research\options_vrp_falsification.py")
ovf = importlib.util.module_from_spec(spec)
sys.modules["ovf_script"] = ovf
spec.loader.exec_module(ovf)

ANN = 365.0

import os
os.chdir(ROOT)  # so relative cache paths resolve

raw = dol.load({"universe": {"assets": ["BTC", "ETH"]}})
panels = oab.build_panels(raw)
dates = panels.dates
i_btc = panels.assets.index("BTC")
spot = panels.spot_ary[:, i_btc]
iv = panels.iv_ary[:, i_btc]
funding = panels.funding_ary[:, i_btc]
T = len(spot)
print("=== panel: T=%d  %s -> %s ===" % (T, dates[0].date(), dates[-1].date()))

cfg = ovf.SimConfig()  # primary: premium_frac=0.05, roll=21
ref = ovf.simulate_asset(spot, iv, funding, cfg)

# ---------------------------------------------------------------- replica
def replica(spot, iv, funding, cfg):
    T = len(spot)
    equity = cfg.initial_capital
    daily_pnl = np.zeros(T)
    eq_curve = np.full(T, cfg.initial_capital, float)
    n = 0.0; K = 0.0; tau = 0.0; q_prev = 0.0; days_held = 0
    tau_step = 1.0 / ANN
    # state records, indexed by step t (position held over t -> t+1)
    st = {k: np.full(T, np.nan) for k in
          ("n", "K", "tau_start", "tau_end", "q", "V_t", "eq_start",
           "roll_flag", "untaxed_perp_jump")}
    fee_close_correct_minus_actual = 0.0
    fee_close_actual_sum = 0.0
    fee_open_sum = 0.0; spread_sum = 0.0; rehedge_sum = 0.0; fund_sum = 0.0

    def open_straddle(t):
        nonlocal n, K, tau, equity, q_prev, fee_open_sum, spread_sum
        S, sigma = spot[t], iv[t]
        K_ = S
        tau_ = cfg.entry_tenor_days / ANN
        unit_prem = straddle_price(S, K_, sigma, tau_)
        gross = cfg.premium_frac * equity
        n_ = gross / unit_prem if unit_prem > 0 else 0.0
        prem_total = n_ * unit_prem
        cost = ovf._option_fees(n_, S, prem_total, cfg) + ovf._spread_cost(n_, S, sigma, tau_, cfg)
        fee_open_sum += ovf._option_fees(n_, S, prem_total, cfg)
        spread_sum += ovf._spread_cost(n_, S, sigma, tau_, cfg)
        equity -= cost
        daily_pnl[t] -= cost
        n, K, tau = n_, K_, tau_
        q_prev = n * straddle_delta(S, K, sigma, tau)

    def close_straddle(t):
        nonlocal n, equity, fee_close_correct_minus_actual, fee_close_actual_sum, spread_sum
        if n <= 0:
            return
        S, sigma = spot[t], iv[t]
        V_actual = straddle_price(S, K, max(tau, tau_step), sigma)      # ARG-SWAPPED as in script line 144
        V_correct = straddle_price(S, K, sigma, max(tau, tau_step))     # correct order
        fee_a = ovf._option_fees(n, S, n * V_actual, cfg)
        fee_c = ovf._option_fees(n, S, n * V_correct, cfg)
        fee_close_correct_minus_actual += (fee_c - fee_a)
        fee_close_actual_sum += fee_a
        sc = ovf._spread_cost(n, S, sigma, max(tau, tau_step), cfg)
        spread_sum += sc
        cost = fee_a + sc
        equity -= cost
        daily_pnl[t] -= cost
        n = 0.0

    t0 = 0
    while t0 < T and (not np.isfinite(spot[t0]) or not np.isfinite(iv[t0]) or iv[t0] <= 0):
        t0 += 1
    open_straddle(t0)
    days_held = 0
    for t in range(t0, T - 1):
        S_t, sig_t = spot[t], iv[t]
        S_n, sig_n = spot[t + 1], iv[t + 1]
        if not (np.isfinite(S_n) and np.isfinite(sig_n) and sig_n > 0):
            S_n, sig_n = S_t, sig_t
        q_t = n * straddle_delta(S_t, K, sig_t, max(tau, tau_step))
        rehedge_cost = 0.0 if cfg.frictionless else cfg.perp_taker_fee * abs(q_t - q_prev) * S_t
        rehedge_sum += rehedge_cost
        V_t = straddle_price(S_t, K, sig_t, max(tau, tau_step))
        tau_next = max(tau - 1.0 / ANN, 1.0 / ANN)
        V_n = straddle_price(S_n, K, sig_n, tau_next)
        option_pnl = n * (V_t - V_n)
        hedge_pnl = q_t * (S_n - S_t)
        funding_pnl = 0.0 if cfg.frictionless else -q_t * S_t * funding[t]
        fund_sum += funding_pnl
        pnl = option_pnl + hedge_pnl - rehedge_cost + funding_pnl
        # record state for stress
        st["n"][t] = n; st["K"][t] = K; st["tau_start"][t] = max(tau, 1.0/ANN)
        st["tau_end"][t] = tau_next; st["q"][t] = q_t; st["V_t"][t] = V_t
        st["eq_start"][t] = equity
        daily_pnl[t + 1] += pnl
        equity += pnl
        eq_curve[t + 1] = equity
        q_prev = q_t
        tau = tau_next
        days_held += 1
        if days_held >= cfg.roll_days or tau <= (cfg.entry_tenor_days - cfg.roll_days - 1) / ANN:
            q_old = q_prev
            close_straddle(t + 1)
            if equity > 0:
                open_straddle(t + 1)
            st["roll_flag"][t + 1] = 1.0
            # perp jump from old hedge to new initial hedge, untaxed in sim
            st["untaxed_perp_jump"][t + 1] = abs(q_prev - q_old) * spot[t + 1] * cfg.perp_taker_fee
            days_held = 0
            eq_curve[t + 1] = equity
    return dict(daily_pnl=daily_pnl, eq_curve=eq_curve, t0=t0, st=st,
                cost_decomp=dict(fee_open=fee_open_sum, fee_close=fee_close_actual_sum,
                                 fee_close_bug_delta=fee_close_correct_minus_actual,
                                 spread=spread_sum, rehedge=rehedge_sum, funding=fund_sum))

rep = replica(spot, iv, funding, cfg)
print("replica parity max|d_eq| = %.3e   max|d_pnl| = %.3e" % (
    np.max(np.abs(rep["eq_curve"] - ref["eq_curve"])),
    np.max(np.abs(rep["daily_pnl"] - ref["daily_pnl"]))))

net_ret = ovf._returns_from_pnl(rep["daily_pnl"], rep["eq_curve"])
sharpe = ovf._sharpe(net_ret); pf = ovf._profit_factor(rep["daily_pnl"])
dd = ovf._max_drawdown(rep["eq_curve"]); tot = rep["eq_curve"][-1]/cfg.initial_capital - 1
print("A) headline BTC net: Sharpe %.4f PF %.4f maxDD %.4f ret %.4f  (verdict: 1.1002/1.1927/0.0538/0.3052)"
      % (sharpe, pf, dd, tot))
print("   cost decomposition over %d bars: %s" % (T, {k: round(v,1) for k,v in rep["cost_decomp"].items()}))

# --------------------------------------------------- B) distribution stats
d = net_ret[np.isfinite(net_ret)]
mu, sd = d.mean(), d.std(ddof=1)
skew = ((d - mu)**3).mean() / sd**3
kurt = ((d - mu)**4).mean() / sd**4
downside = d[d < 0]
sortino = mu / downside.std(ddof=1) * math.sqrt(ANN) if len(downside) > 2 else float("nan")
var95 = np.quantile(d, 0.05); cvar95 = d[d <= var95].mean()
var99 = np.quantile(d, 0.01); cvar99 = d[d <= var99].mean()
print("B) daily net ret: mean %.5f sd %.5f skew %.2f kurt %.1f | Sortino %.3f | "
      "VaR95 %.4f CVaR95 %.4f CVaR99 %.4f | worst day %.4f" %
      (mu, sd, skew, kurt, sortino, var95, cvar95, cvar99, d.min()))
order = np.argsort(net_ret)
print("   10 worst CLOSE-marked days:")
for j in order[:10]:
    print("     %s  ret %+.3f%%  pnl %+.0f" % (dates[j].date(), net_ret[j]*100, rep["daily_pnl"][j]))

# annual CVaR check vs gates.yaml tail.cvar95_floor_pct (-8% — units presumably window/period level)
print("   gates.yaml tail.cvar95_floor_pct=-8.0; daily CVaR95*100 = %.2f (NO python consumer computes this anywhere)" % (cvar95*100))

# --------------------------------------------------- C) intrabar stress
perp_btc = raw.perp["BTC"].copy()
dvol_btc = raw.dvol["BTC"].copy()
perp_btc.index = pd.DatetimeIndex(perp_btc.index).floor("D")
perp_btc = perp_btc[~perp_btc.index.duplicated(keep="last")]
dvol_btc.index = pd.DatetimeIndex(dvol_btc.index).floor("D")
dvol_btc = dvol_btc[~dvol_btc.index.duplicated(keep="last")]
dlow  = perp_btc["low"].reindex(dates).to_numpy(float)
dhigh = perp_btc["high"].reindex(dates).to_numpy(float)
vhigh = (dvol_btc["high"].reindex(dates).to_numpy(float)) / 100.0
vclose= (dvol_btc["close"].reindex(dates).to_numpy(float)) / 100.0

st = rep["st"]
eqc = rep["eq_curve"]
intraday_trough = eqc.copy()
intra_loss = np.zeros(T)
for t in range(T - 1):
    n_, K_, q_, V_t = st["n"][t], st["K"][t], st["q"][t], st["V_t"][t]
    if not np.isfinite(n_) or n_ <= 0:
        continue
    tau_mid = st["tau_end"][t]          # tau during the step (fair, end-of-step)
    S_t = spot[t]
    # path of step t->t+1 lives in perp bar of row t+1; dvol bars t+1 (16h) + t+2 (8h)
    sl, sh = dlow[t + 1], dhigh[t + 1]
    if not (np.isfinite(sl) and np.isfinite(sh)):
        continue
    sig_hi = np.nanmax([vhigh[t + 1], vhigh[t + 2] if t + 2 < T else np.nan, iv[t]])
    worst = -1e18
    for S_x in (sl, sh):
        V_x = straddle_price(S_x, K_, sig_hi, max(tau_mid, 1/ANN))
        loss_x = n_ * (V_x - V_t) - q_ * (S_x - S_t)   # positive = loss for the book
        worst = max(worst, loss_x)
    eq_x = st["eq_start"][t] - worst
    intraday_trough[t + 1] = min(eqc[t + 1], eq_x)
    intra_loss[t + 1] = worst - (-rep["daily_pnl"][t + 1])  # extra loss vs close-marked day pnl (approx; cost included in daily)

peak = np.maximum.accumulate(eqc)
dd_close = (1 - eqc / peak).max()
dd_intra = (1 - intraday_trough / peak).max()
t_worst = int(np.argmax(1 - intraday_trough / peak))
print("C) close-marked maxDD %.2f%%  vs INTRADAY-trough maxDD %.2f%%  (worst at %s)"
      % (dd_close*100, dd_intra*100, dates[t_worst].date()))
exc = (eqc - intraday_trough) / peak
order2 = np.argsort(-exc)
print("   10 largest intraday excursions below close-mark equity (%% of peak):")
for j in order2[:10]:
    print("     %s  extra-below-close %.2f%%  close-day ret %+.2f%%  trough-eq %.0f" %
          (dates[j].date(), exc[j]*100, net_ret[j]*100 if np.isfinite(net_ret[j]) else float('nan'),
           intraday_trough[j]))
# named events
for evd in ("2021-05-19","2021-12-03","2021-12-04","2022-05-09","2022-05-11","2022-05-12",
            "2022-06-13","2022-11-08","2022-11-09","2022-11-10","2024-08-04","2024-08-05",
            "2025-10-10","2025-10-11"):
    try:
        j = int(np.where(dates.date.astype(str) == evd)[0][0])
    except Exception:
        continue
    print("   EVENT %s: close ret %+.2f%% | intraday trough eq %.0f (extra %.2f%%) | n=%.2f K=%.0f q=%.2f"
          % (evd, (net_ret[j]*100 if np.isfinite(net_ret[j]) else float('nan')),
             intraday_trough[j], exc[j]*100,
             st["n"][j-1] if j>0 else float('nan'), st["K"][j-1] if j>0 else float('nan'),
             st["q"][j-1] if j>0 else float('nan')))

# --------------------------------------------------- D) overnight shock re-derivation
print("D) instantaneous shock on the standing book, per day (pct of equity):")
for shock_s, shock_v in ((-0.15, 0.20), (-0.30, 0.40), (0.20, 0.15)):
    losses = []
    for t in range(T - 1):
        n_, K_, q_, V_t = st["n"][t], st["K"][t], st["q"][t], st["V_t"][t]
        if not np.isfinite(n_) or n_ <= 0:
            continue
        S_t = spot[t]
        S_x = S_t * (1 + shock_s)
        sig_x = iv[t] + shock_v
        V_x = straddle_price(S_x, K_, sig_x, max(st["tau_end"][t], 1/ANN))
        loss = n_ * (V_x - V_t) - q_ * (S_x - S_t)
        losses.append(loss / st["eq_start"][t])
    losses = np.array(losses)
    print("   spot %+d%% & IV %+d pts: loss median %.2f%% | p90 %.2f%% | max %.2f%% of equity"
          % (shock_s*100, shock_v*100, np.median(losses)*100, np.quantile(losses,0.9)*100, losses.max()*100))

# vega/gamma/notional profile
mask = np.isfinite(st["n"]) & (st["n"] > 0)
notional = st["n"][mask] * spot[:T][mask] / st["eq_start"][mask]
veg = np.array([st["n"][t] * straddle_vega(spot[t], st["K"][t], iv[t], max(st["tau_end"][t],1/ANN))
                for t in range(T-1) if np.isfinite(st["n"][t]) and st["n"][t] > 0])
veq = veg / st["eq_start"][mask]
print("   book profile: straddle notional/equity median %.2f max %.2f | vega(per 100 vol pts)/equity median %.3f -> per +1pt = %.4f"
      % (np.median(notional), notional.max(), np.median(veq), np.median(veq)/100))

# --------------------------------------------------- E) margin / liquidation probe (Deribit approx)
# short option IM per leg ~ max(0.15 - OTM/S, 0.1)*S + mark ; MM per leg ~ 0.075*S + mark (approx, non-PM)
im_usage = []
for t in range(T - 1):
    n_, K_ = st["n"][t], st["K"][t]
    if not np.isfinite(n_) or n_ <= 0:
        continue
    S_t, sig_t, tau_ = spot[t], iv[t], max(st["tau_end"][t], 1/ANN)
    Vc = straddle_price(S_t, K_, sig_t, tau_)  # both legs combined mark
    im = n_ * ((max(0.15 - max(K_-S_t,0)/S_t, 0.1) + max(0.15 - max(S_t-K_,0)/S_t, 0.1)) * S_t + Vc)
    im_usage.append(im / st["eq_start"][t])
im_usage = np.array(im_usage)
print("E) approx Deribit IM usage: median %.1f%%  max %.1f%% of equity (5%% premium sizing)"
      % (np.median(im_usage)*100, im_usage.max()*100))
# liquidation check under -30%/+40pt: equity after loss vs MM after
worst_liq = 0.0
for t in range(T - 1):
    n_, K_, q_ = st["n"][t], st["K"][t], st["q"][t]
    if not np.isfinite(n_) or n_ <= 0:
        continue
    S_t = spot[t]; S_x = S_t * 0.70; sig_x = iv[t] + 0.40
    V_x = straddle_price(S_x, K_, sig_x, max(st["tau_end"][t],1/ANN))
    eq_x = st["eq_start"][t] - (n_ * (V_x - st["V_t"][t]) - q_ * (S_x - S_t))
    mm = n_ * (0.15 * S_x + V_x) + abs(q_) * S_x * 0.0125
    worst_liq = max(worst_liq, mm / eq_x if eq_x > 0 else 99.0)
print("   worst MM/equity under -30%%/+40pt shock: %.2f (>1 => forced liquidation)" % worst_liq)

# --------------------------------------------------- F) bootstrap + PSR/DSR
rng = np.random.default_rng(7)
dd_ = d.copy(); Tn = len(dd_)
block = 21
n_boot = 10000
sh_boot = np.empty(n_boot)
for b in range(n_boot):
    idx = []
    while len(idx) < Tn:
        s0 = rng.integers(0, Tn)
        idx.extend(((s0 + np.arange(block)) % Tn).tolist())
    x = dd_[np.array(idx[:Tn])]
    sh_boot[b] = x.mean() / x.std(ddof=1) * math.sqrt(ANN)
print("F) circular block bootstrap (b=21,10k): Sharpe CI95 [%.2f, %.2f]  P(SR<0.5)=%.3f  P(SR<0)=%.4f"
      % (np.quantile(sh_boot,0.025), np.quantile(sh_boot,0.975),
         (sh_boot < 0.5).mean(), (sh_boot < 0).mean()))
# PSR vs 0 and 0.5 annual (convert to per-day)
sr_d = mu / sd
for target_ann in (0.0, 0.5):
    tgt = target_ann / math.sqrt(ANN)
    psr = ncdf((sr_d - tgt) * math.sqrt(Tn - 1) / math.sqrt(1 - skew*sr_d + (kurt-1)/4*sr_d**2))
    print("   PSR(SR>%.1f ann) = %.4f" % (target_ann, psr))
# DSR: trials = 8 grid configs x 2 assets = 16 (+ spread grid in cost_robustness ~12 more)
grid_sh = np.array([-0.1507, 0.6405, 0.8081, 1.1554, -0.1513, 0.6499, 0.8171, 1.1568]) / math.sqrt(ANN)
v_sr = grid_sh.var(ddof=1)
gamma = 0.5772156649
for N in (8, 16, 28):
    sr_star = 0  # placeholder; real value computed in the loop below
for N in (8, 16, 28):
    from math import e
    def qnorm(p):
        # inverse normal via binary search on ncdf
        lo, hi = -10, 10
        for _ in range(80):
            mid = (lo+hi)/2
            if ncdf(mid) < p: lo = mid
            else: hi = mid
        return (lo+hi)/2
    sr_star = math.sqrt(v_sr) * ((1-gamma)*qnorm(1-1/N) + gamma*qnorm(1-1/(N*e)))
    dsr = ncdf((sr_d - sr_star) * math.sqrt(Tn - 1) / math.sqrt(1 - skew*sr_d + (kurt-1)/4*sr_d**2))
    print("   DSR (N=%d trials, var from grid): SR*=%.3f/day (%.2f ann) -> DSR=%.4f" %
          (N, sr_star, sr_star*math.sqrt(ANN), dsr))

# --------------------------------------------------- G) VRP-negative window conditioning
logret = np.diff(np.log(spot))
win = 30
rows = []
for a in range(0, len(logret) - win, win):
    rv2 = np.nansum(logret[a:a+win]**2) * (ANN/win)
    iv2 = iv[a]**2
    rows.append((dates[a].date(), iv2 - rv2, math.sqrt(rv2), iv[a]))
neg = [r for r in rows if r[1] <= 0]
print("G) Model-B windows: %d total, %d negative (%.0f%%). Negative windows:" %
      (len(rows), len(neg), 100*len(neg)/len(rows)))
for r in neg:
    print("     start %s  var-spread %+.3f  RV %.2f  IV %.2f" % r)
spreads = np.array([r[1] for r in rows])
print("   mean +%.3f | worst window %+.3f | sum of negatives %.3f vs sum positives %.3f" %
      (spreads.mean(), spreads.min(), spreads[spreads<0].sum(), spreads[spreads>0].sum()))

# --------------------------------------------------- H) close-fee arg-swap quantification
for roll in (21, 30):
    c2 = ovf.SimConfig(roll_days=roll)
    r2 = replica(spot, iv, funding, c2)
    print("H) roll=%dd: close-fee bug delta (correct-actual) = $%.1f over sim; close fees actual $%.1f"
          % (roll, r2["cost_decomp"]["fee_close_bug_delta"], r2["cost_decomp"]["fee_close"]))
    uj = np.nansum(r2["st"]["untaxed_perp_jump"])
    print("   roll=%dd: untaxed roll-day perp hedge-jump fees = $%.1f" % (roll, uj))

# --------------------------------------------------- I) IV-mark staleness on crash rows
print("I) 8h IV/spot misalignment on the 5 worst close days (row IV vs next-row IV):")
for j in order[:5]:
    print("     %s  iv[t]=%.1f -> iv[t+1]=%.1f (pts moved %+0.1f); dvol_high[t+1]=%.1f"
          % (dates[j].date(), iv[j-1]*100 if j>0 else float('nan'), iv[j]*100,
             (iv[j]-iv[j-1])*100 if j>0 else float('nan'),
             vhigh[j]*100 if np.isfinite(vhigh[j]) else float('nan')))
