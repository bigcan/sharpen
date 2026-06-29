"""FULLER validation of the delta-hedged Taiwan TXO VRP scout (cont-87 NEXT plan).

The scout (taiwan_txo_vrp_deltahedge.py) returned a research-tier GO (net Sharpe +0.89) but leaned
on three modeled assumptions. This script strengthens them with data already on disk (NO new
FinMind calls), targeting the scout-tier caveats:

  (1) MODEL-delta dependence: the scout hedges with a BS delta at the CONSTANT entry IV. Here we
      sweep the delta MODEL — entry IV (baseline), +/-20% IV perturbation, a per-leg SMILE-aware
      delta (call-side & put-side IV inverted from the real entry chain), and an RV "oracle"
      diagnostic — and require every CAUSAL variant to keep clearing the Sharpe floor. (The
      gold-standard real-daily-delta mark-to-market still needs the full daily chain; token-gated.)
  (2) HEDGE realism: the scout hedges the untradeable TAIEX SPOT at a modeled 2bps. Here the hedge
      runs on the REAL, TRADEABLE TX-FUTURE daily close (from the cont-86 TX 1-min panel) — which
      carries the real futures basis and converges to settlement — priced at the EMPIRICAL
      tick-level cost (TX min tick = 1 index pt; Corwin-Schultz on the TX daily high-low came back
      below estimator resolution => sub-tick true spread; we use a conservative 1.0-pt half-spread).
  (4) MULTIPLICITY: adds the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) at the project's
      0.95 capital bar, deflating for skew, kurtosis, sample length, and the trial count.

Reuses the audited, Math-verified option-leg machinery from the scout (same entry premium, ATM pick,
cash-settled intrinsic, entry-IV back-out). Hedge instrument and cost are the only things replaced.

Usage:
    python scripts/research/taiwan_txo_vrp_validation.py --selftest
    python scripts/research/taiwan_txo_vrp_validation.py \
        --data data/taiwan_options --tx data/taiwan_intraday/TX_1min.parquet \
        --gates configs/taiwan_txo_vrp_validation.gates.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Reuse the audited helpers from the scouts (DRY; identical formulas).
from scripts.research.taiwan_txo_vrp_scout import (  # noqa: E402
    _BS_ATM, bootstrap_p_mean_le_zero, pick_atm, realized_vol, sharpe,
)
# Reuse the project-canonical, audited multiple-testing controls (promoted from the options-VRP
# sleeve) — do NOT reimplement DSR/PSR here. All Sharpe inputs are PER-PERIOD (per cycle).
from finrl_pro_ds.crypto.eval.statistics import (  # noqa: E402
    deflated_sharpe_ratio, excess_kurtosis, skewness,
)
from finrl_pro_ds.crypto.eval.statistics import sharpe_ratio as _per_period_sharpe  # noqa: E402

log = logging.getLogger("txo_vrp_validation")
_ND = NormalDist()
_N = _ND.cdf

# delta MODELS under test. is_causal=False => uses look-ahead (oracle), reported but NOT gated.
DELTA_MODES = (
    ("entry_iv", True),    # baseline == the scout: BS delta at the constant entry IV
    ("perturb_lo", True),  # entry IV * 0.8  (mis-specified low)
    ("perturb_hi", True),  # entry IV * 1.2  (mis-specified high)
    ("smile", True),       # per-leg call/put IV inverted from the real entry chain
    ("rv_oracle", False),  # delta at the realized vol (look-ahead diagnostic, NOT gated)
)


# --------------------------------------------------------------------------- #
# Black-Scholes call/put + per-leg implied-vol inversion (r=q=0). The future is
# the underlying => this is the Black-76 forward parameterisation (F == S spot
# when r=q=0, so the same d1/d2 with the future price in place of spot).
# --------------------------------------------------------------------------- #
def bs_call_put(s: float, k: float, tau: float, sigma: float) -> tuple[float, float]:
    """(call, put) price in index pts, r=q=0. At/after expiry returns intrinsic."""
    if tau <= 0 or sigma <= 0 or s <= 0:
        return max(s - k, 0.0), max(k - s, 0.0)
    vsqrt = sigma * math.sqrt(tau)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * tau) / vsqrt
    d2 = d1 - vsqrt
    return s * _N(d1) - k * _N(d2), k * _N(-d2) - s * _N(-d1)


def implied_vol_leg(price: float, s: float, k: float, tau: float, is_call: bool) -> float:
    """Bisection BS implied vol of ONE leg (r=q=0). NaN if price <= intrinsic (no time value)."""
    intrinsic = max(s - k, 0.0) if is_call else max(k - s, 0.0)
    if not (price > intrinsic + 1e-9) or tau <= 0 or s <= 0:
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        c, p = bs_call_put(s, k, tau, mid)
        px = c if is_call else p
        if px > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def straddle_delta_legs(f: float, k: float, tau: float, sig_c: float, sig_p: float) -> float:
    """SHORT-straddle hedge ratio H = #future units to hold long to neutralise delta.
    Long-straddle delta = call_delta + put_delta = N(d1_c) + (N(d1_p) - 1); holding that many
    future units offsets the short position. Per-leg IVs allow a smile-aware hedge."""
    if tau <= 0 or f <= 0:
        dc = 1.0 if f > k else 0.0          # call delta indicator at expiry
        dp = -1.0 if f < k else 0.0         # put delta indicator at expiry
        return dc + dp
    def d1(sig: float) -> float:
        vs = sig * math.sqrt(tau)
        return (math.log(f / k) + 0.5 * sig * sig * tau) / vs
    cd = _N(d1(sig_c)) if sig_c > 0 else (1.0 if f > k else 0.0)
    pd_ = (_N(d1(sig_p)) - 1.0) if sig_p > 0 else (-1.0 if f < k else 0.0)
    return cd + pd_


# --------------------------------------------------------------------------- #
# TX-future daily close (real, tradeable hedge instrument) from the 1-min panel.
# --------------------------------------------------------------------------- #
def load_tx_daily(tx_path: Path, session_start: str, session_end: str) -> pd.DataFrame:
    """Regular-session (08:45-13:45) daily close + high/low of the TX future → [date, close, high, low]."""
    tx = pd.read_parquet(tx_path)
    t = tx["timestamp"].dt.time
    reg = tx[(t >= pd.Timestamp(session_start).time()) & (t <= pd.Timestamp(session_end).time())]
    daily = (reg.assign(date=reg["timestamp"].dt.normalize())
             .groupby("date").agg(close=("close", "last"), high=("high", "max"),
                                   low=("low", "min")).reset_index().sort_values("date"))
    return daily.reset_index(drop=True)


def corwin_schultz_halfspread_pts(tx_daily: pd.DataFrame) -> float:
    """Corwin-Schultz (2012) 2-day high-low effective HALF-spread in index pts (diagnostic).
    Median across day-pairs of the positive-clipped estimate. Sub-resolution (negative) => ~0."""
    H = tx_daily["high"].to_numpy(float)
    L = tx_daily["low"].to_numpy(float)
    lvl = tx_daily["close"].to_numpy(float)
    if len(H) < 3:
        return float("nan")
    k = 3.0 - 2.0 * math.sqrt(2.0)
    beta = np.log(H[:-1] / L[:-1]) ** 2 + np.log(H[1:] / L[1:]) ** 2
    h2 = np.maximum(H[:-1], H[1:])
    l2 = np.minimum(L[:-1], L[1:])
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s_rel = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))   # relative full spread
    s_rel = np.clip(np.where(np.isfinite(s_rel), s_rel, np.nan), 0.0, None)
    full_pts = float(np.nanmedian(s_rel * lvl[1:]))
    return full_pts / 2.0


# --------------------------------------------------------------------------- #
# Hedge over the cycle on the TX-future path → (hedge_pnl_pts, turnover_units).
# Cost is recovered as turnover * half_spread_pts (linear; exact for the sweep).
# --------------------------------------------------------------------------- #
def hedge_cycle_tx(path_dates: np.ndarray, path_f: np.ndarray, expiry: pd.Timestamp, k: float,
                   sig_c: float, sig_p: float, vol_floor: float) -> tuple[float, float]:
    hedge_pnl = 0.0
    turnover = 0.0
    h_prev = 0.0
    for i in range(len(path_f) - 1):
        tau = max((expiry - pd.Timestamp(path_dates[i])).days / 365.0, vol_floor)
        h = straddle_delta_legs(float(path_f[i]), k, tau, sig_c, sig_p)
        turnover += abs(h - h_prev)
        hedge_pnl += h * (float(path_f[i + 1]) - float(path_f[i]))
        h_prev = h
    turnover += abs(0.0 - h_prev)        # unwind at expiry
    return hedge_pnl, turnover


def build_cycles(chains: pd.DataFrame, settle: pd.DataFrame, tx_daily: pd.DataFrame,
                 spot: pd.DataFrame, costs: dict, ann: int, mode: str) -> pd.DataFrame:
    """One row per monthly cycle. Option leg == the scout (entry premium, cash-settle intrinsic,
    entry-IV back-out); HEDGE runs on the TX-future path with the chosen delta `mode`."""
    sd = dict(zip(settle["contract_month"], settle["settle_date"]))
    sp = dict(zip(settle["contract_month"], settle["settle_price"]))
    spot_close = dict(zip(spot["date"], spot["close"]))
    fdates = tx_daily["date"].to_numpy()
    fclose = tx_daily["close"].to_numpy(float)
    hs, fee, tax = (costs["half_spread_pts_per_leg"], costs["fee_pts_per_leg"],
                    costs["tax_rate_on_premium"])
    hedge_hs, vol_floor = costs["hedge_half_spread_pts"], costs["vol_floor"]
    min_path = int(costs.get("min_path_points", 5))   # weekly cycles are short (~5 trading days)

    out = []
    for cm, g in chains.groupby("contract_month"):
        if cm not in sd or cm not in sp:
            continue
        d0 = pd.Timestamp(g["entry_date"].iloc[0])
        d1 = pd.Timestamp(sd[cm])
        s0 = float(g["entry_spot"].iloc[0])
        st = float(sp[cm])
        if not (d0 < d1) or not (s0 > 0) or not (st > 0):
            continue
        atm = pick_atm(g, s0)
        if atm is None:
            continue
        k, c, p = atm
        premium = c + p
        intrinsic = abs(st - k)
        t_years = max((d1 - d0).days, 1) / 365.0
        iv = premium / (_BS_ATM * s0 * math.sqrt(t_years)) if s0 > 0 and t_years > 0 else float("nan")
        if not (iv > 0):
            continue
        rv = realized_vol(spot, d0, d1, ann)

        # delta-model selection (per-leg IVs sig_c, sig_p)
        if mode == "entry_iv":
            sig_c = sig_p = iv
        elif mode == "perturb_lo":
            sig_c = sig_p = 0.8 * iv
        elif mode == "perturb_hi":
            sig_c = sig_p = 1.2 * iv
        elif mode == "smile":
            ivc = implied_vol_leg(c, s0, k, t_years, is_call=True)
            ivp = implied_vol_leg(p, s0, k, t_years, is_call=False)
            sig_c = ivc if np.isfinite(ivc) else iv
            sig_p = ivp if np.isfinite(ivp) else iv
        elif mode == "rv_oracle":              # look-ahead diagnostic only
            sig_c = sig_p = rv if np.isfinite(rv) and rv > 0 else iv
        else:
            raise ValueError(f"unknown delta mode {mode!r}")

        # TX-future path over the cycle (real, tradeable hedge instrument)
        m = (fdates >= np.datetime64(d0)) & (fdates <= np.datetime64(d1))
        path_dates, path_f = fdates[m], fclose[m]
        if len(path_f) < min_path:
            continue
        hedge_pnl, turnover = hedge_cycle_tx(path_dates, path_f, d1, k, sig_c, sig_p, vol_floor)
        hedge_cost = turnover * hedge_hs

        option_cost = 2.0 * hs + 2.0 * fee + tax * (premium + intrinsic)
        gross_unhedged = premium - intrinsic
        gross_hedged = gross_unhedged + hedge_pnl
        net = gross_hedged - option_cost - hedge_cost
        s_t_spot = float(spot_close.get(d1, st))
        eq_ret = math.log(s_t_spot / s0) if s_t_spot > 0 and s0 > 0 else float("nan")
        out.append({
            "contract_month": cm, "entry_date": d0, "expiry_date": d1, "dte": (d1 - d0).days,
            "spot0": s0, "strike": k, "settle": st, "premium": premium, "intrinsic": intrinsic,
            "gross_unhedged_pts": gross_unhedged, "hedge_pnl_pts": hedge_pnl,
            "hedge_turnover": turnover, "gross_hedged_pts": gross_hedged,
            "option_cost_pts": option_cost, "hedge_cost_pts": hedge_cost, "net_pts": net,
            "gross_ret": gross_hedged / s0, "net_ret": net / s0,
            "iv": iv, "rv": rv, "vrp": iv - rv, "eq_ret": eq_ret})
    df = pd.DataFrame(out)
    return df.sort_values("entry_date").reset_index(drop=True) if not df.empty else df


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio — thin wrapper over the audited canonical implementation
# (finrl_pro_ds.crypto.eval.statistics). All Sharpe inputs are PER-CYCLE.
# --------------------------------------------------------------------------- #
def _per_cycle_sr(r: np.ndarray) -> float:
    """Per-cycle Sharpe = mean/std(ddof=1) (periods_per_year=1 => non-annualized)."""
    return float(_per_period_sharpe([float(x) for x in r if np.isfinite(x)], periods_per_year=1))


def compute_dsr(net_rets: np.ndarray, trial_sr_per_cycle: list[float], n_trials: int,
                cpy: float) -> dict:
    """DSR via the canonical control: deflate the per-cycle SR to the expected-max-SR of the
    trial family, skew/kurtosis-adjusted. Returns a flat dict for the report (NaN if undefined)."""
    r = [float(x) for x in net_rets if np.isfinite(x)]
    sr_cyc = _per_cycle_sr(net_rets)
    g1, g2 = skewness(r), excess_kurtosis(r)
    var_sr = float(np.var(np.asarray(trial_sr_per_cycle), ddof=1)) if len(trial_sr_per_cycle) > 1 else 0.0
    res = deflated_sharpe_ratio(sr_cyc, list(trial_sr_per_cycle), n_obs=len(r), skew=g1,
                                excess_kurt=g2, n_trials=n_trials, periods_per_year=int(cpy))
    return {"sr_per_cycle": sr_cyc, "skew": g1, "kurtosis": g2 + 3.0, "n": len(r),
            "sr0_benchmark": (res["sr_star"] if res else float("nan")),
            "var_sr_trials": var_sr, "dsr": (res["dsr"] if res else float("nan"))}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def hedge_cost_sweep_pts(cyc: pd.DataFrame, cpy: float,
                         grid=(0.0, 0.5, 1.0, 2.0, 4.0)) -> list[dict]:
    """Net Sharpe vs the TX-future half-spread (index pts/rebalance). cost = turnover * hs (linear)."""
    gross_h = cyc["gross_hedged_pts"].to_numpy(float)
    opt = cyc["option_cost_pts"].to_numpy(float)
    turn = cyc["hedge_turnover"].to_numpy(float)
    s0 = cyc["spot0"].to_numpy(float)
    rows = []
    for hs in grid:
        net_pts = gross_h - opt - turn * hs
        rows.append({"hedge_half_spread_pts": hs,
                     "net_sharpe_ann": round(sharpe(net_pts / s0, cpy), 4)})
    return rows


def evaluate(cycles_by_mode: dict, gate: dict, defl: dict, nm: dict, scout: dict,
             require_all: list[str], cpy: float, cs_halfspread: float, out_dir: Path) -> dict:
    cyc = cycles_by_mode["entry_iv"]      # PRIMARY variant
    n = len(cyc)
    if n < 6:
        log.error("Only %d cycles — too few to evaluate.", n)
        return {"_go": False, "n_cycles": n}
    net = cyc["net_ret"].to_numpy(float)
    gross = cyc["gross_ret"].to_numpy(float)
    eq = cyc["eq_ret"].to_numpy(float)
    net_sharpe = sharpe(net, cpy)
    gross_sharpe = sharpe(gross, cpy)
    split = int(n * scout["oos_split_frac"])
    oos_net_sharpe = sharpe(net[split:], cpy)
    is_net_sharpe = sharpe(net[:split], cpy)
    mask = np.isfinite(net) & np.isfinite(eq)
    beta = float(np.polyfit(eq[mask], net[mask], 1)[0]) if mask.sum() >= 3 else float("nan")
    corr = float(np.corrcoef(eq[mask], net[mask])[0, 1]) if mask.sum() >= 3 else float("nan")
    vrp = cyc["vrp"].to_numpy(float)
    vrp_pos_frac = float(np.mean(vrp[np.isfinite(vrp)] > 0)) if np.isfinite(vrp).any() else float("nan")
    boot_p = bootstrap_p_mean_le_zero(net, nm["n_boot"])
    worst = float(np.nanmin(net))
    cvar5 = float(np.mean(np.sort(net[np.isfinite(net)])[:max(1, int(0.05 * n))]))

    # delta-model robustness: net Sharpe per mode; gate on the CAUSAL ones
    delta_robust = []
    causal_sharpes = []
    for name, is_causal in DELTA_MODES:
        cm = cycles_by_mode[name]
        s = sharpe(cm["net_ret"].to_numpy(float), cpy)
        delta_robust.append({"mode": name, "causal": is_causal, "net_sharpe_ann": round(s, 4),
                             "mean_hedge_pnl_pts": round(float(cm["hedge_pnl_pts"].mean()), 2)})
        if is_causal:
            causal_sharpes.append(s)
    delta_models_robust = all(np.isfinite(s) and s >= gate["delta_robust_min_sharpe"]
                              for s in causal_sharpes)

    # DSR: trial-SR dispersion estimated from the family of per-cycle SRs we actually computed
    # (causal delta modes + the cost-sweep variants) — an empirical proxy for the search dispersion.
    trial_sr_per_cycle = [_per_cycle_sr(cycles_by_mode[name]["net_ret"].to_numpy(float))
                          for name, is_causal in DELTA_MODES if is_causal]
    gross_h = cyc["gross_hedged_pts"].to_numpy(float)
    opt = cyc["option_cost_pts"].to_numpy(float)
    turn = cyc["hedge_turnover"].to_numpy(float)
    s0 = cyc["spot0"].to_numpy(float)
    for hs in (0.0, 0.5, 1.0, 2.0, 4.0):
        trial_sr_per_cycle.append(_per_cycle_sr((gross_h - opt - turn * hs) / s0))
    dsr_grid = [{"n_trials": nt, **{k: round(v, 4) for k, v in
                                    compute_dsr(net, trial_sr_per_cycle, nt, cpy).items()}}
                for nt in defl["n_trials_report"]]
    dsr_gate = compute_dsr(net, trial_sr_per_cycle, defl["n_trials_gate"], cpy)
    var_sr = dsr_gate["var_sr_trials"]

    checks = {
        "net_sharpe_above_floor": np.isfinite(net_sharpe) and net_sharpe >= gate["net_sharpe_floor"],
        "oos_net_sharpe_positive": np.isfinite(oos_net_sharpe) and oos_net_sharpe > gate["oos_net_sharpe_min"],
        "beats_bootstrap_null": np.isfinite(boot_p) and boot_p < gate["bootstrap_p_max"],
        "equity_beta_below_ceiling": np.isfinite(beta) and abs(beta) <= gate["max_equity_beta_abs"],
        "vrp_premium_exists": np.isfinite(vrp_pos_frac) and vrp_pos_frac >= gate["vrp_positive_frac_min"],
        "delta_models_robust": delta_models_robust,
        "dsr_above_floor": np.isfinite(dsr_gate["dsr"]) and dsr_gate["dsr"] >= gate["dsr_min"],
    }
    missing = [k for k in require_all if k not in checks]
    if missing:
        raise KeyError(f"verdict.require_all names unknown checks: {missing}")
    go = all(checks[k] for k in require_all)
    verdict = ("GO (delta-hedged TXO VRP survives a real tradeable TX-future hedge, real tick cost, "
               "delta-model perturbation, AND DSR deflation; proceed to token-gated daily-chain "
               "real-delta + weeklies, then Tier-2 audit before capital)" if go else
               "NO-GO (edge was an artifact of spot/2bps/constant-IV-delta modeling, or fails DSR)")

    report = {
        "instrument": "TXO monthly ATM short straddle, DAILY delta-hedged on the REAL TX FUTURE",
        "n_cycles": n, "date_min": str(cyc["entry_date"].min()), "date_max": str(cyc["expiry_date"].max()),
        "hedge_instrument": "TX future (cont-86 1-min -> daily reg-session close)",
        "cs_halfspread_pts_diagnostic": round(cs_halfspread, 3),
        "gross_sharpe_ann": round(gross_sharpe, 4), "net_sharpe_ann": round(net_sharpe, 4),
        "is_net_sharpe_ann": round(is_net_sharpe, 4), "oos_net_sharpe_ann": round(oos_net_sharpe, 4),
        "mean_net_ret_per_cycle": round(float(np.nanmean(net)), 6),
        "mean_gross_unhedged_pts": round(float(cyc["gross_unhedged_pts"].mean()), 2),
        "mean_hedge_pnl_pts": round(float(cyc["hedge_pnl_pts"].mean()), 2),
        "mean_gross_hedged_pts": round(float(cyc["gross_hedged_pts"].mean()), 2),
        "mean_option_cost_pts": round(float(cyc["option_cost_pts"].mean()), 2),
        "mean_hedge_cost_pts": round(float(cyc["hedge_cost_pts"].mean()), 2),
        "mean_iv": round(float(np.nanmean(cyc["iv"])), 4), "mean_rv": round(float(np.nanmean(cyc["rv"])), 4),
        "mean_vrp": round(float(np.nanmean(vrp)), 4), "vrp_positive_frac": round(vrp_pos_frac, 3),
        "equity_beta": round(beta, 4), "equity_corr": round(corr, 4),
        "bootstrap_p_mean_le_zero": round(boot_p, 4),
        "worst_cycle_net_ret": round(worst, 5), "cvar5_net_ret": round(cvar5, 5),
        "delta_model_robustness": delta_robust,
        "hedge_cost_sweep_pts": hedge_cost_sweep_pts(cyc, cpy),
        "dsr_var_sr_trials": round(var_sr, 8),
        "dsr_gate": {"n_trials": defl["n_trials_gate"], **{k: round(v, 4) for k, v in dsr_gate.items()}},
        "dsr_grid": dsr_grid,
        "gate": gate, "checks": checks, "verdict": verdict, "_go": go,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vrp_validation_report.json").write_text(json.dumps(report, indent=2))
    cyc.to_parquet(out_dir / "vrp_validation_cycles.parquet", index=False)

    print("\n=== TXO MONTHLY VRP - FULLER VALIDATION (real TX-future hedge) ===")
    print(f"cycles: {n}  ({report['date_min'][:10]} .. {report['date_max'][:10]})")
    print(f"Sharpe (ann)   : gross {gross_sharpe:+.3f}   NET {net_sharpe:+.3f}   "
          f"(IS {is_net_sharpe:+.3f} / OOS {oos_net_sharpe:+.3f})")
    print(f"un-hedged gross {report['mean_gross_unhedged_pts']:+.0f}pt  + hedge "
          f"{report['mean_hedge_pnl_pts']:+.0f}pt  = hedged {report['mean_gross_hedged_pts']:+.0f}pt")
    print(f"costs: option {report['mean_option_cost_pts']:.1f}pt  TXfut-hedge "
          f"{report['mean_hedge_cost_pts']:.1f}pt  -> mean net ret {report['mean_net_ret_per_cycle']:+.4%}")
    print(f"IV {report['mean_iv']:.3f}  RV {report['mean_rv']:.3f}  VRP {report['mean_vrp']:+.3f}  "
          f"(IV>RV {vrp_pos_frac:.0%})   equity beta {beta:+.3f} (corr {corr:+.3f})")
    print(f"tail: worst {worst:+.2%}  CVaR5 {cvar5:+.2%}   bootstrap p {boot_p:.4f}")
    print("delta-model robustness (ann net Sharpe):",
          "  ".join(f"{d['mode']}{'' if d['causal'] else '*'}={d['net_sharpe_ann']:+.2f}"
                    for d in delta_robust), " (*=non-causal oracle, not gated)")
    print("TXfut hedge-cost sweep (half-spread pt -> net Sharpe):",
          "  ".join(f"{r['hedge_half_spread_pts']:.1f}={r['net_sharpe_ann']:+.2f}"
                    for r in report["hedge_cost_sweep_pts"]))
    print(f"DSR @N={defl['n_trials_gate']}: {dsr_gate['dsr']:.3f}  "
          f"(sr/cyc {dsr_gate['sr_per_cycle']:+.3f}, sr0 {dsr_gate['sr0_benchmark']:+.3f}, "
          f"skew {dsr_gate['skew']:+.2f}, kurt {dsr_gate['kurtosis']:.2f})")
    print("DSR grid:", "  ".join(f"N{d['n_trials']}={d['dsr']:.3f}" for d in dsr_grid))
    print("checks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print(f"VERDICT: {verdict}")
    print(f"written: {out_dir/'vrp_validation_report.json'}")
    return report


def _build_all_modes(chains, settle, tx_daily, spot, costs, ann) -> dict:
    return {name: build_cycles(chains, settle, tx_daily, spot, costs, ann, name)
            for name, _ in DELTA_MODES}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Fuller validation of the delta-hedged Taiwan TXO VRP")
    ap.add_argument("--data", default="data/taiwan_options")
    ap.add_argument("--tx", default="data/taiwan_intraday/TX_1min.parquet")
    ap.add_argument("--gates", default="configs/taiwan_txo_vrp_validation.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_txo_vrp_validation")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    g = yaml.safe_load(((ROOT / args.gates) if not Path(args.gates).is_absolute()
                        else Path(args.gates)).read_text(encoding="utf-8"))
    scout, costs, gate = g["scout"], g["costs"], g["gate"]
    defl, nm, hcfg = g["deflation"], g["null_model"], g["hedge"]
    ann = scout["rv_window_trading_days"]
    cpy = float(scout.get("cycles_per_year", 12.0))    # 12 monthly, ~52 weekly (annualization)
    require_all = g["verdict"]["require_all"]

    if args.selftest:
        return _selftest(costs, gate, defl, nm, scout, hcfg, ann, cpy, require_all)

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    tx_path = (ROOT / args.tx) if not Path(args.tx).is_absolute() else Path(args.tx)
    chains = pd.read_parquet(data / "TXO_vrp_entry_chains.parquet")
    settle = pd.read_parquet(data / "TXO_vrp_settlement.parquet")
    spot = pd.read_parquet(data / "TAIEX_spot.parquet")
    tx_daily = load_tx_daily(tx_path, hcfg["session_start"], hcfg["session_end"])
    cs_hs = corwin_schultz_halfspread_pts(tx_daily)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    cycles_by_mode = _build_all_modes(chains, settle, tx_daily, spot, costs, ann)
    # Guard: a wrong cycles_per_year mis-annualizes Sharpe by sqrt(cpy/true). Cross-check the
    # configured cpy against the median calendar DTE actually in the data; halt on a gross mismatch.
    base = cycles_by_mode["entry_iv"]
    if len(base):
        med_dte = float(np.median(base["dte"].to_numpy(float)))
        implied_cpy = 365.25 / med_dte if med_dte > 0 else cpy
        if not (0.6 <= cpy / implied_cpy <= 1.6):
            raise ValueError(
                f"cycles_per_year={cpy} contradicts the data (median DTE {med_dte:.0f} cal-days "
                f"=> ~{implied_cpy:.0f}/yr). Fix scout.cycles_per_year in the gates yaml.")
        log.info("cadence check: median DTE %.0f cal-days => ~%.0f cycles/yr (configured cpy=%.0f)",
                 med_dte, implied_cpy, cpy)
    report = evaluate(cycles_by_mode, gate, defl, nm, scout, require_all, cpy, cs_hs, out_dir)
    return 0 if report["_go"] else 1


def _selftest(costs, gate, defl, nm, scout, hcfg, ann, cpy, require_all) -> int:
    """Synthetic GBM: TX-future path == spot path (zero basis), realized < implied => the
    delta-hedged short straddle nets positive; proves BS legs, IV inversion, hedge loop, DSR wiring."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rng = np.random.default_rng(7)
    px = 15000 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(dates)))
    spot = pd.DataFrame({"date": dates, "close": px})
    tx_daily = pd.DataFrame({"date": dates, "close": px, "high": px * 1.001, "low": px * 0.999})
    months = pd.date_range("2020-02-01", "2024-12-01", freq="MS")
    rows_chain, rows_settle = [], []
    for m in months:
        entry = dates[dates >= m][0]
        exp = dates[dates >= (m + pd.Timedelta(days=28))][0]
        s0 = float(spot.loc[spot["date"] == entry, "close"].iloc[0])
        st = float(spot.loc[spot["date"] == exp, "close"].iloc[0])
        cm = m.strftime("%Y%m")
        k = round(s0 / 100) * 100
        prem = 0.030 * s0
        rows_chain += [
            {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
             "call_put": "call", "close": prem / 2, "volume": 1, "oi": 1},
            {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
             "call_put": "put", "close": prem / 2, "volume": 1, "oi": 1}]
        rows_settle.append({"contract_month": cm, "settle_date": exp, "settle_price": st})
    chains, settle = pd.DataFrame(rows_chain), pd.DataFrame(rows_settle)
    cs_hs = corwin_schultz_halfspread_pts(tx_daily)
    cycles_by_mode = _build_all_modes(chains, settle, tx_daily, spot, costs, ann)
    out = Path(os.environ.get("TMPDIR", ".")) / "_txo_vrp_validation_selftest"
    rep = evaluate(cycles_by_mode, gate, defl, nm, scout, require_all, cpy, cs_hs, out)
    ok = rep["n_cycles"] >= 12 and np.isfinite(rep["net_sharpe_ann"]) and \
        np.isfinite(rep["dsr_gate"]["dsr"])
    log.info("SELFTEST cycles=%d net_sharpe=%s dsr=%s -> %s", rep["n_cycles"],
             rep["net_sharpe_ann"], rep["dsr_gate"]["dsr"], "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
