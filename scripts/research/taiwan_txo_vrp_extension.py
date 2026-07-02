"""TXO VRP monthly BACKWARD-EXTENSION confirmatory test (single-shot, pre-registered).

Two-stage frame (configs/taiwan_txo_vrp_extension.gates.yaml, hash-locked BEFORE the extension
data was fetched):
  Stage 1 (done, cont-87/88): design selected from a ~50-trial family on 2019-2025 (n=83) —
  net Sharpe +0.63, FAILED the capital bar on power (bootstrap p=0.0501, DSR@N=50=0.767).
  Stage 2 (THIS script): the EXACT frozen design evaluated ONCE on monthly cycles 2002-2018
  that the search never touched. No selection on this sample => N_trials=1 => the confirmatory
  statistic is the skew/kurt-adjusted PSR at 0.95 (canonical implementation), plus a circular
  block-bootstrap p(SR<=0) and the same economic gates. The COMBINED 2002-2025 sample still
  carries the stage-1 search => DSR deflated at the ORIGINAL N=50.

Engine notes:
  - Option leg identical to the audited stage-1 machinery (imported, not re-implemented).
  - Hedge path: per-contract TX-future daily REGULAR-session close (13:45) — prereg deviation
    D1, measured conservative (-0.09 Sharpe on 2019-2025). A reconciliation tripwire re-runs
    the stage-1 period on THIS data source and must reproduce the known 13:45-mark result
    before the extension sample is evaluated.
  - Period-correct co-primary: pre-2013 Taiwan futures/options transaction taxes were up to
    ~12x today's — the co-primary must ALSO clear the Sharpe floor (prereg D4).

Usage:
    python scripts/research/taiwan_txo_vrp_extension.py --selftest
    python scripts/research/taiwan_txo_vrp_extension.py \
        --data data/taiwan_options_ext --prior-data data/taiwan_options \
        --prior-cycles results/taiwan_txo_vrp_validation/vrp_validation_cycles.parquet \
        --gates configs/taiwan_txo_vrp_extension.gates.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Audited stage-1 machinery (DRY — identical formulas; equivalence tripwired in tests).
from scripts.research.taiwan_txo_vrp_scout import (  # noqa: E402
    _BS_ATM, pick_atm, realized_vol, sharpe,
)
from scripts.research.taiwan_txo_vrp_validation import (  # noqa: E402
    _per_cycle_sr, compute_dsr, implied_vol_leg, straddle_delta_legs,
)
from finrl_pro_ds.crypto.eval.statistics import (  # noqa: E402
    block_bootstrap_sharpe_ci, excess_kurtosis, probabilistic_sharpe_ratio, skewness,
)

log = logging.getLogger("txo_vrp_extension")

DELTA_MODES = (("entry_iv", True), ("perturb_lo", True), ("perturb_hi", True),
               ("smile", True), ("rv_oracle", False))


# --------------------------------------------------------------------------- #
# Schedules (period-correct costs) — values live in the gates yaml, never here.
# --------------------------------------------------------------------------- #
def sched_rate(schedule: list[dict], when: pd.Timestamp, key: str) -> float:
    """First schedule row whose 'through' date >= when. Schedule must be sorted ascending."""
    for row in schedule:
        if when <= pd.Timestamp(str(row["through"])):
            return float(row[key])
    return float(schedule[-1][key])


# --------------------------------------------------------------------------- #
# Hedge loop over the per-contract TX path. Mirrors the audited stage-1
# hedge_cycle_tx EXACTLY for pnl/turnover (equivalence is unit-tested), and
# additionally accumulates a per-date-rate cost and returns the final-day delta
# (for the settlement-gap stress).
# --------------------------------------------------------------------------- #
def hedge_cycle_tx_ext(path_dates: np.ndarray, path_f: np.ndarray, expiry: pd.Timestamp,
                       k: float, sig_c: float, sig_p: float, vol_floor: float,
                       cost_per_unit: np.ndarray) -> tuple[float, float, float, float]:
    """-> (hedge_pnl_pts, turnover_units, cost_pts_at_per_date_rate, h_into_final_day)."""
    hedge_pnl = 0.0
    turnover = 0.0
    cost = 0.0
    h_prev = 0.0
    for i in range(len(path_f) - 1):
        tau = max((expiry - pd.Timestamp(path_dates[i])).days / 365.0, vol_floor)
        h = straddle_delta_legs(float(path_f[i]), k, tau, sig_c, sig_p)
        dturn = abs(h - h_prev)
        turnover += dturn
        cost += dturn * float(cost_per_unit[i])
        hedge_pnl += h * (float(path_f[i + 1]) - float(path_f[i]))
        h_prev = h
    turnover += abs(0.0 - h_prev)                      # unwind at expiry
    cost += abs(0.0 - h_prev) * float(cost_per_unit[-1])
    return hedge_pnl, turnover, cost, h_prev


def _atm_leg_volumes(chain: pd.DataFrame, k: float) -> tuple[float, float]:
    c = chain[(chain["call_put"] == "call") & (chain["strike"] == k)]["volume"]
    p = chain[(chain["call_put"] == "put") & (chain["strike"] == k)]["volume"]
    return (float(c.iloc[0]) if len(c) else 0.0, float(p.iloc[0]) if len(p) else 0.0)


def build_cycles_ext(chains: pd.DataFrame, settle: pd.DataFrame, tx_daily: pd.DataFrame,
                     spot: pd.DataFrame, costs: dict, pc: dict | None, ann: int, mode: str,
                     apply_quality_floors: bool = True) -> pd.DataFrame:
    """One row per monthly cycle. Option leg == stage 1 (entry premium, cash-settle intrinsic,
    entry-IV back-out); hedge on the cycle's OWN TX contract daily-close path (13:45 marks).

    ``pc`` (period-correct block) adds the co-primary cost columns; the FROZEN primary cost
    columns are always produced and identical in construction to stage 1."""
    sd = dict(zip(settle["contract_month"], settle["settle_date"]))
    sp = dict(zip(settle["contract_month"], settle["settle_price"]))
    spot_close = dict(zip(spot["date"], spot["close"]))
    hs, fee, tax = (costs["half_spread_pts_per_leg"], costs["fee_pts_per_leg"],
                    costs["tax_rate_on_premium"])
    hedge_hs, vol_floor = costs["hedge_half_spread_pts"], costs["vol_floor"]
    min_path = int(costs.get("min_path_points", 5))
    min_vol = float(costs.get("min_atm_leg_volume", 0)) if apply_quality_floors else 0.0
    dte_min = int(costs.get("dte_min", 0)) if apply_quality_floors else 0
    dte_max = int(costs.get("dte_max", 10_000)) if apply_quality_floors else 10_000
    tx_by_cm = {cm: g.sort_values("date") for cm, g in tx_daily.groupby("contract_month")}

    out = []
    for cm, g in chains.groupby("contract_month"):
        if cm not in sd or cm not in sp:
            continue
        d0 = pd.Timestamp(g["entry_date"].iloc[0])
        d1 = pd.Timestamp(sd[cm])
        s0 = float(g["entry_spot"].iloc[0])
        st = float(sp[cm])
        dte = (d1 - d0).days
        if not (d0 < d1) or not (s0 > 0) or not (st > 0):     # causality + sanity
            continue
        if not (dte_min <= dte <= dte_max):
            continue
        atm = pick_atm(g, s0)                                  # frozen selection rule
        if atm is None:
            continue
        k, c, p = atm
        vol_c, vol_p = _atm_leg_volumes(g, k)
        if min(vol_c, vol_p) < min_vol:                        # pick-then-filter (prereg D2)
            continue
        premium = c + p
        intrinsic = abs(st - k)
        t_years = max(dte, 1) / 365.0
        iv = premium / (_BS_ATM * s0 * math.sqrt(t_years)) if s0 > 0 and t_years > 0 else float("nan")
        if not (iv > 0):
            continue
        rv = realized_vol(spot, d0, d1, ann)

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
        elif mode == "rv_oracle":                              # look-ahead diagnostic only
            sig_c = sig_p = rv if np.isfinite(rv) and rv > 0 else iv
        else:
            raise ValueError(f"unknown delta mode {mode!r}")

        txg = tx_by_cm.get(str(cm))
        if txg is None:
            continue
        m = (txg["date"] >= d0) & (txg["date"] <= d1)
        path_dates = txg.loc[m, "date"].to_numpy()
        path_f = txg.loc[m, "close"].to_numpy(float)
        # OLD settlement regime (pre-2008): the final settlement day is the business day
        # AFTER the TX contract's last trading day, so the path has no mark on d1. The TX
        # future itself cash-settles to the SAME settlement print, so the faithful hedger
        # holds to settlement: append (d1, settlement_print) as the final mark. Modern-regime
        # cycles (settle date == last trading day) are unaffected.
        if len(path_f) and pd.Timestamp(path_dates[-1]) < d1:
            path_dates = np.append(path_dates, np.datetime64(d1))
            path_f = np.append(path_f, st)
        if len(path_f) < min_path:
            continue

        # per-date hedge cost per unit turnover: frozen primary = flat half-spread only;
        # period-correct = half-spread + fee + levied tax(rate_t x F_t in pts).
        if pc is not None:
            dts = pd.DatetimeIndex(path_dates)
            rates = np.array([sched_rate(pc["hedge_tax_schedule"], d, "rate") for d in dts])
            pc_unit = hedge_hs + float(pc["hedge_fee_pts_per_side"]) + rates * path_f
        else:
            pc_unit = np.full(len(path_f), hedge_hs)
        hedge_pnl, turnover, pc_hedge_cost, h_last = hedge_cycle_tx_ext(
            path_dates, path_f, d1, k, sig_c, sig_p, vol_floor, pc_unit)
        hedge_cost = turnover * hedge_hs                       # frozen primary

        option_cost = 2.0 * hs + 2.0 * fee + tax * (premium + intrinsic)
        if pc is not None:
            fee_pc = sched_rate(pc["option_fee_pts_per_leg_schedule"], d0, "fee")
            tax_pc = sched_rate(pc["option_tax_schedule"], d0, "rate")
            option_cost_pc = 2.0 * hs + 2.0 * fee_pc + tax_pc * (premium + intrinsic)
        else:
            pc_hedge_cost = hedge_cost
            option_cost_pc = option_cost
        gross_unhedged = premium - intrinsic
        gross_hedged = gross_unhedged + hedge_pnl
        net = gross_hedged - option_cost - hedge_cost
        net_pc = gross_hedged - option_cost_pc - pc_hedge_cost
        s_t_spot = float(spot_close.get(d1, st))
        eq_ret = math.log(s_t_spot / s0) if s_t_spot > 0 and s0 > 0 else float("nan")
        out.append({
            "contract_month": cm, "entry_date": d0, "expiry_date": d1, "dte": dte,
            "spot0": s0, "strike": k, "settle": st, "premium": premium, "intrinsic": intrinsic,
            "gross_unhedged_pts": gross_unhedged, "hedge_pnl_pts": hedge_pnl,
            "hedge_turnover": turnover, "gross_hedged_pts": gross_hedged,
            "option_cost_pts": option_cost, "hedge_cost_pts": hedge_cost, "net_pts": net,
            "gross_ret": gross_hedged / s0, "net_ret": net / s0,
            "net_ret_period_correct": net_pc / s0,
            "option_cost_pc_pts": option_cost_pc, "hedge_cost_pc_pts": pc_hedge_cost,
            "iv": iv, "rv": rv, "vrp": iv - rv, "eq_ret": eq_ret,
            "atm_call_vol": vol_c, "atm_put_vol": vol_p,
            "h_final": h_last, "f_final": float(path_f[-1]),
        })
    df = pd.DataFrame(out)
    return df.sort_values("entry_date").reset_index(drop=True) if not df.empty else df


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def newey_west_t(r: np.ndarray, lags: int) -> float:
    """NW t-stat of the mean with Bartlett weights (per-cycle returns)."""
    x = r[np.isfinite(r)]
    n = len(x)
    if n < 4:
        return float("nan")
    xd = x - x.mean()
    g0 = float(xd @ xd) / n
    s = g0
    for lag in range(1, min(lags, n - 1) + 1):
        cov = float(xd[lag:] @ xd[:-lag]) / n
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * cov
    se = math.sqrt(max(s, 1e-18) / n)
    return float(x.mean() / se)


def block_boot_p_sr_le_zero(net: np.ndarray, block: int, n_boot: int, cpy: float) -> float:
    res = block_bootstrap_sharpe_ci([float(x) for x in net if np.isfinite(x)],
                                    block=block, n_boot=n_boot, periods_per_year=int(cpy))
    return float(res["p_sharpe_lt_0"]) if res else float("nan")


def psr_per_cycle(net: np.ndarray) -> float:
    """Skew/kurt-adjusted PSR vs SR*=0 on PER-CYCLE returns (periods_per_year=1 keeps the
    BLdP variance bracket in per-period units — the codebase convention)."""
    r = [float(x) for x in net if np.isfinite(x)]
    return float(probabilistic_sharpe_ratio(r, sr_benchmark=0.0, periods_per_year=1))


def gap_stress(cyc: pd.DataFrame, gap: float) -> dict:
    """+/-`gap` synthetic settlement gap on the FINAL day, hedge frozen at h_final.
    Stressed net = net + h_final*f_final*g - (|settle*(1+g) - k| - |settle - k|). Report-only."""
    worst = {"gap_frac": gap, "worst_net_ret": float("nan"), "worst_contract": None,
             "worst_direction": None}
    best_seen = math.inf
    for _, row in cyc.iterrows():
        for g in (gap, -gap):
            settle_g = row["settle"] * (1.0 + g)
            d_intr = abs(settle_g - row["strike"]) - row["intrinsic"]
            net_g = (row["net_pts"] + row["h_final"] * row["f_final"] * g - d_intr) / row["spot0"]
            if net_g < best_seen:
                best_seen = net_g
                worst.update({"worst_net_ret": round(float(net_g), 5),
                              "worst_contract": str(row["contract_month"]),
                              "worst_direction": ("+" if g > 0 else "-") + f"{abs(g):.0%}"})
    return worst


def _series_stats(net: np.ndarray, eq: np.ndarray, cpy: float, nm: dict) -> dict:
    mask = np.isfinite(net)
    n = int(mask.sum())
    x = net[mask]
    beta = corr = float("nan")
    m2 = np.isfinite(net) & np.isfinite(eq)
    if m2.sum() >= 3:
        beta = float(np.polyfit(eq[m2], net[m2], 1)[0])
        corr = float(np.corrcoef(eq[m2], net[m2])[0, 1])
    half = n // 2
    return {
        "n": n,
        "net_sharpe_ann": round(sharpe(x, cpy), 4),
        "sr_per_cycle": round(_per_cycle_sr(x), 4),
        "first_half_sharpe_ann": round(sharpe(x[:half], cpy), 4),
        "second_half_sharpe_ann": round(sharpe(x[half:], cpy), 4),
        "mean_net_ret": round(float(np.mean(x)), 6),
        "skew": round(skewness([float(v) for v in x]), 4),
        "excess_kurtosis": round(excess_kurtosis([float(v) for v in x]), 4),
        "psr": round(psr_per_cycle(x), 4),
        "block_bootstrap_p_sr_le_0": round(
            block_boot_p_sr_le_zero(x, nm["block_cycles"], nm["n_boot"], cpy), 4),
        "newey_west_t": round(newey_west_t(x, nm["nw_lags"]), 3),
        "equity_beta": round(beta, 4), "equity_corr": round(corr, 4),
        "worst_cycle_net_ret": round(float(np.min(x)), 5),
        "cvar5_net_ret": round(float(np.mean(np.sort(x)[:max(1, int(0.05 * n))])), 5),
    }


# --------------------------------------------------------------------------- #
# Evaluation (single shot)
# --------------------------------------------------------------------------- #
def evaluate(ext_by_mode: dict, prior_cycles: pd.DataFrame, prior_by_mode: dict, g: dict,
             recon_sharpe: float, out_dir: Path) -> dict:
    gate, cg, defl, nm = g["gate"], g["combined_gate"], g["deflation"], g["null_model"]
    cpy = float(g["scout"]["cycles_per_year"])
    ext = ext_by_mode["entry_iv"]
    n_ext = len(ext)
    report: dict = {"prereg_gates_file": "configs/taiwan_txo_vrp_extension.gates.yaml",
                    "reconciliation_net_sharpe_1345": round(recon_sharpe, 4)}

    if n_ext < int(g["prereg"]["min_cycles_decision_grade"]):
        report.update({"n_extension_cycles": n_ext, "verdict": "DATA-INSUFFICIENT",
                       "_go": False})
        _write_report(report, ext, out_dir)
        return report

    net = ext["net_ret"].to_numpy(float)
    eq = ext["eq_ret"].to_numpy(float)
    st = _series_stats(net, eq, cpy, nm)
    vrp = ext["vrp"].to_numpy(float)
    vrp_pos = float(np.mean(vrp[np.isfinite(vrp)] > 0))
    pc_sharpe = sharpe(ext["net_ret_period_correct"].to_numpy(float), cpy)

    causal = {}
    for mode, is_causal in DELTA_MODES:
        s = sharpe(ext_by_mode[mode]["net_ret"].to_numpy(float), cpy)
        causal[mode] = {"causal": is_causal, "net_sharpe_ann": round(s, 4)}
    robust_ok = all(np.isfinite(v["net_sharpe_ann"]) and v["net_sharpe_ann"] >= gate["delta_robust_min_sharpe"]
                    for m, v in causal.items() if v["causal"])

    checks_ext = {
        "net_sharpe_above_floor": st["net_sharpe_ann"] >= gate["net_sharpe_floor"],
        "second_half_positive": st["second_half_sharpe_ann"] > gate["second_half_net_sharpe_min"],
        "beats_block_bootstrap": st["block_bootstrap_p_sr_le_0"] < gate["block_bootstrap_p_max"],
        "psr_above_bar": st["psr"] >= gate["psr_min"],
        "equity_beta_below_ceiling": abs(st["equity_beta"]) <= gate["max_equity_beta_abs"],
        "vrp_premium_exists": vrp_pos >= gate["vrp_positive_frac_min"],
        "primary_mode_above_floor": causal["entry_iv"]["net_sharpe_ann"] >= gate["primary_delta_mode_floor"],
        "delta_models_robust": robust_ok,
        "worst_cycle_ok": st["worst_cycle_net_ret"] >= gate["worst_cycle_net_ret_min"],
        "cvar5_ok": st["cvar5_net_ret"] >= gate["cvar5_net_ret_min"],
        "period_correct_above_floor": (np.isfinite(pc_sharpe)
                                       and pc_sharpe >= gate["period_correct_net_sharpe_floor"]),
    }
    ext_pass = all(checks_ext[k] for k in g["verdict"]["require_all_extension"])

    # ---------------- combined 2002-2025 (stored stage-1 primary + extension primary) -------
    overlap = set(ext["contract_month"].astype(str)) & set(prior_cycles["contract_month"].astype(str))
    if overlap:
        raise RuntimeError(f"extension/prior contract overlap: {sorted(overlap)[:5]}")
    shared = ["contract_month", "entry_date", "expiry_date", "spot0", "net_ret", "gross_ret",
              "eq_ret", "gross_hedged_pts", "option_cost_pts", "hedge_turnover", "vrp", "iv", "rv"]
    comb = (pd.concat([prior_cycles[shared], ext[shared]], ignore_index=True)
            .sort_values("entry_date").reset_index(drop=True))
    cnet = comb["net_ret"].to_numpy(float)
    cst = _series_stats(cnet, comb["eq_ret"].to_numpy(float), cpy, nm)

    # DSR trial family mirrors stage 1: causal delta modes + the hedge-cost grid, computed on
    # the pooled sample (prior-period non-primary modes REBUILT on 13:45 marks — affects only
    # Var(trial SR); the stage-1 frozen var is reported as sensitivity).
    trial_sr = []
    for mode, is_causal in DELTA_MODES:
        if not is_causal:
            continue
        pooled = pd.concat([prior_by_mode[mode]["net_ret"], ext_by_mode[mode]["net_ret"]],
                           ignore_index=True).to_numpy(float)
        trial_sr.append(_per_cycle_sr(pooled))
    gross_h = comb["gross_hedged_pts"].to_numpy(float)
    opt_c = comb["option_cost_pts"].to_numpy(float)
    turn = comb["hedge_turnover"].to_numpy(float)
    s0 = comb["spot0"].to_numpy(float)
    for hs_i in (0.0, 0.5, 1.0, 2.0, 4.0):
        trial_sr.append(_per_cycle_sr((gross_h - opt_c - turn * hs_i) / s0))
    dsr_gate = compute_dsr(cnet, trial_sr, defl["n_trials_gate"], cpy)
    dsr_grid = [{"n_trials": nt, "dsr": round(compute_dsr(cnet, trial_sr, nt, cpy)["dsr"], 4)}
                for nt in defl["n_trials_report"]]
    # sensitivity: stage-1 frozen across-trial variance (results/.../vrp_validation_report.json)
    frozen_var = 0.00167382
    sr_c = _per_cycle_sr(cnet)
    z_frozen = ((sr_c - _sr_star(frozen_var, defl["n_trials_gate"]))
                * math.sqrt(max(cst["n"] - 1, 1))
                / math.sqrt(max(1.0 - cst["skew"] * sr_c
                                + (cst["excess_kurtosis"] + 2.0) / 4.0 * sr_c ** 2, 1e-12)))
    dsr_frozen_var = 0.5 * (1.0 + math.erf(z_frozen / math.sqrt(2.0)))

    checks_comb = {
        "combined_sharpe_above_floor": cst["net_sharpe_ann"] >= cg["net_sharpe_floor"],
        "combined_beats_bootstrap": cst["block_bootstrap_p_sr_le_0"] < cg["block_bootstrap_p_max"],
        "combined_dsr_above_conditional_floor": dsr_gate["dsr"] >= cg["dsr_n50_conditional_floor"],
        "combined_dsr_above_full_floor": dsr_gate["dsr"] >= cg["dsr_n50_full_floor"],
    }
    comb_cond = all(checks_comb[k] for k in g["verdict"]["require_all_combined_conditional"])

    if ext_pass and comb_cond and checks_comb["combined_dsr_above_full_floor"]:
        verdict = "GO"
    elif ext_pass and comb_cond:
        verdict = "CONDITIONAL-GO (PROMISING, needs Tier-2; two-stage frame adjudication)"
    else:
        verdict = "NO-GO-FINAL (book closes permanently)"

    # exploratory (declared in prereg 3.6, not gated)
    ext_y = ext.assign(year=ext["entry_date"].dt.year)
    per_year = {int(y): round(sharpe(gg["net_ret"].to_numpy(float), cpy), 3)
                for y, gg in ext_y.groupby("year")}
    worst10 = (ext.nsmallest(10, "net_ret")
               [["contract_month", "entry_date", "net_ret", "iv", "rv"]].copy())
    worst10["entry_date"] = worst10["entry_date"].astype(str)
    worst10["net_ret"] = worst10["net_ret"].round(4)
    pc_early_wc = g["period_correct"].get("early_tax_worstcase_rate")

    report.update({
        "n_extension_cycles": n_ext,
        "extension_date_min": str(ext["entry_date"].min()), "extension_date_max": str(ext["expiry_date"].max()),
        "extension": st,
        "extension_gross_sharpe_ann": round(sharpe(ext["gross_ret"].to_numpy(float), cpy), 4),
        "vrp_positive_frac": round(vrp_pos, 3),
        "mean_iv": round(float(np.nanmean(ext["iv"])), 4),
        "mean_rv": round(float(np.nanmean(ext["rv"])), 4),
        "period_correct_net_sharpe_ann": round(float(pc_sharpe), 4),
        "delta_model_robustness": causal,
        "hedge_cost_sweep_pts": [
            {"hedge_half_spread_pts": hs_i,
             "net_sharpe_ann": round(sharpe(
                 (ext["gross_hedged_pts"] - ext["option_cost_pts"]
                  - ext["hedge_turnover"] * hs_i).to_numpy(float) / ext["spot0"].to_numpy(float),
                 cpy), 4)}
            for hs_i in (0.0, 0.5, 1.0, 2.0, 4.0)],
        "gap_stress": gap_stress(ext, float(g["stress"]["settlement_gap_frac"])),
        "per_year_net_sharpe": per_year,
        "worst10_cycles": worst10.to_dict("records"),
        "early_tax_worstcase_rate_diag": pc_early_wc,
        "combined": cst,
        "combined_dsr_gate": {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in dsr_gate.items()},
        "combined_dsr_grid": dsr_grid,
        "combined_dsr_frozen_var_sensitivity": round(dsr_frozen_var, 4),
        "checks_extension": checks_ext, "checks_combined": checks_comb,
        "extension_pass": bool(ext_pass), "verdict": verdict,
        "_go": bool(ext_pass and comb_cond),
    })
    _write_report(report, ext, out_dir)
    return report


def _sr_star(var_sr: float, n_trials: int) -> float:
    from statistics import NormalDist
    nd = NormalDist()
    ge = 0.5772156649015329
    return math.sqrt(var_sr) * ((1.0 - ge) * nd.inv_cdf(1.0 - 1.0 / n_trials)
                                + ge * nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e)))


def _write_report(report: dict, ext: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vrp_extension_report.json").write_text(json.dumps(report, indent=2, default=str))
    if not ext.empty:
        ext.to_parquet(out_dir / "vrp_extension_cycles.parquet", index=False)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("n_extension_cycles", "extension", "period_correct_net_sharpe_ann",
                               "combined", "combined_dsr_gate", "checks_extension",
                               "checks_combined", "verdict", "reconciliation_net_sharpe_1345")},
                     indent=2, default=str))
    log.info("written: %s", out_dir / "vrp_extension_report.json")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _build_all_modes(chains, settle, tx_daily, spot, costs, pc, ann, quality=True) -> dict:
    return {name: build_cycles_ext(chains, settle, tx_daily, spot, costs, pc, ann, name,
                                   apply_quality_floors=quality)
            for name, _ in DELTA_MODES}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="TXO VRP backward-extension confirmatory test")
    ap.add_argument("--data", default="data/taiwan_options_ext")
    ap.add_argument("--prior-data", default="data/taiwan_options")
    ap.add_argument("--prior-cycles",
                    default="results/taiwan_txo_vrp_validation/vrp_validation_cycles.parquet")
    ap.add_argument("--gates", default="configs/taiwan_txo_vrp_extension.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_txo_vrp_extension")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    g = yaml.safe_load(((ROOT / args.gates) if not Path(args.gates).is_absolute()
                        else Path(args.gates)).read_text(encoding="utf-8"))
    ann = g["scout"]["rv_window_trading_days"]
    cpy = float(g["scout"]["cycles_per_year"])
    if args.selftest:
        return _selftest(g, ann, cpy)

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    pdata = (ROOT / args.prior_data) if not Path(args.prior_data).is_absolute() else Path(args.prior_data)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    chains = pd.read_parquet(data / "TXO_vrp_entry_chains.parquet")
    settle = pd.read_parquet(data / "TXO_vrp_settlement.parquet")
    spot = pd.read_parquet(data / "TAIEX_spot.parquet")
    tx_daily = pd.read_parquet(data / "TX_daily.parquet")

    # -------- reconciliation tripwire FIRST (stage-1 period, THIS data source) ------------
    p_chains = pd.read_parquet(pdata / "TXO_vrp_entry_chains.parquet")
    p_settle = pd.read_parquet(pdata / "TXO_vrp_settlement.parquet")
    p_spot = pd.read_parquet(pdata / "TAIEX_spot.parquet")
    recon_costs = dict(g["costs"])
    recon_costs["min_path_points"] = 5          # stage-1 floors (no volume/DTE floors)
    recon = build_cycles_ext(p_chains, p_settle, tx_daily, p_spot, recon_costs, None, ann,
                             "entry_iv", apply_quality_floors=False)
    recon_sharpe = sharpe(recon["net_ret"].to_numpy(float), cpy) if len(recon) else float("nan")
    exp, tol = g["reconciliation"]["expected_net_sharpe_1345_mark"], g["reconciliation"]["tolerance"]
    log.info("reconciliation (2019-2025 on TaiwanFuturesDaily 13:45 closes): net Sharpe %.4f "
             "(expected %.2f +/- %.2f, n=%d)", recon_sharpe, exp, tol, len(recon))
    if not (np.isfinite(recon_sharpe) and abs(recon_sharpe - exp) <= tol):
        log.error("RECONCILIATION FAILED — engine/data mismatch. HALT (no verdict).")
        return 3

    # -------- extension (single shot) ------------------------------------------------------
    ext_by_mode = _build_all_modes(chains, settle, tx_daily, spot, g["costs"],
                                   g["period_correct"], ann, quality=True)
    base = ext_by_mode["entry_iv"]
    if len(base):
        med_dte = float(np.median(base["dte"].to_numpy(float)))
        implied_cpy = 365.25 / med_dte if med_dte > 0 else cpy
        if not (0.6 <= cpy / implied_cpy <= 1.6):
            raise ValueError(f"cycles_per_year={cpy} contradicts data (median DTE {med_dte:.0f} "
                             f"=> ~{implied_cpy:.0f}/yr)")
    prior_cycles = pd.read_parquet((ROOT / args.prior_cycles)
                                   if not Path(args.prior_cycles).is_absolute()
                                   else Path(args.prior_cycles))
    prior_by_mode = _build_all_modes(p_chains, p_settle, tx_daily, p_spot, recon_costs, None,
                                     ann, quality=False)
    report = evaluate(ext_by_mode, prior_cycles, prior_by_mode, g, recon_sharpe, out_dir)
    return 0 if report.get("_go") else 1


# --------------------------------------------------------------------------- #
# Self-test: synthetic GBM with a positive variance premium proves the plumbing
# (per-contract paths, period-correct costs, PSR/bootstrap/NW, verdict wiring).
# --------------------------------------------------------------------------- #
def _selftest(g: dict, ann: int, cpy: float) -> int:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2003-01-02", "2017-12-29")
    px = 8000 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(dates)))
    spot = pd.DataFrame({"date": dates, "close": px})
    months = pd.date_range("2003-02-01", "2017-12-01", freq="MS")
    rows_c, rows_s, rows_tx = [], [], []
    prev_exp = dates[10]
    for m in months:
        entry = dates[dates > prev_exp][0]
        exp = dates[dates >= (m + pd.Timedelta(days=25))][0]
        if entry >= exp:
            prev_exp = exp
            continue
        s0 = float(spot.loc[spot["date"] == entry, "close"].iloc[0])
        st = float(spot.loc[spot["date"] == exp, "close"].iloc[0])
        cm = m.strftime("%Y%m")
        k = round(s0 / 100) * 100
        prem = 0.032 * s0
        rows_c += [{"contract_month": cm, "entry_date": entry, "entry_spot": s0,
                    "strike": float(k), "call_put": cp, "close": prem / 2, "volume": 500,
                    "oi": 1000} for cp in ("call", "put")]
        rows_s.append({"contract_month": cm, "settle_date": exp, "settle_price": st})
        seg = spot[(spot["date"] >= entry) & (spot["date"] <= exp)]
        rows_tx += [{"date": d, "contract_month": cm, "close": c, "high": c * 1.001,
                     "low": c * 0.999, "settlement_price": c}
                    for d, c in zip(seg["date"], seg["close"])]
        prev_exp = exp
    chains, settle = pd.DataFrame(rows_c), pd.DataFrame(rows_s)
    tx_daily = pd.DataFrame(rows_tx)
    ext_by_mode = _build_all_modes(chains, settle, tx_daily, spot, g["costs"],
                                   g["period_correct"], ann, quality=True)
    ext = ext_by_mode["entry_iv"]
    # a fake tiny "prior" sample from the tail so the combined path is exercised
    prior = ext.tail(20).copy()
    prior["contract_month"] = prior["contract_month"].astype(str) + "P"
    prior_by_mode = {m: ext_by_mode[m].tail(20).assign(
        contract_month=lambda d: d["contract_month"].astype(str) + "P") for m, _ in DELTA_MODES}
    ext_by_mode = {m: d.iloc[:-20] for m, d in ext_by_mode.items()}
    out = Path(os.environ.get("TMPDIR", os.environ.get("TEMP", "."))) / "_txo_ext_selftest"
    rep = evaluate(ext_by_mode, prior, prior_by_mode, g, recon_sharpe=0.54, out_dir=out)
    ok = (rep["n_extension_cycles"] >= 100 and np.isfinite(rep["extension"]["net_sharpe_ann"])
          and np.isfinite(rep["combined_dsr_gate"]["dsr"])
          and np.isfinite(rep["period_correct_net_sharpe_ann"])
          and rep["extension"]["net_sharpe_ann"] > rep["period_correct_net_sharpe_ann"] - 5.0)
    log.info("SELFTEST n=%d net=%s pc=%s dsr=%s -> %s", rep["n_extension_cycles"],
             rep["extension"]["net_sharpe_ann"], rep["period_correct_net_sharpe_ann"],
             rep["combined_dsr_gate"]["dsr"], "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
