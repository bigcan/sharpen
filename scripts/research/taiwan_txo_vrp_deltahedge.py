"""DELTA-HEDGED Taiwan TXO VRP scout — is the positive variance premium harvestable alpha?

The un-hedged monthly short straddle was NO-GO (taiwan_txo_vrp_scout.py: net Sharpe -0.54,
frictionless -0.53, killed by the crash tail). But TXO's variance premium is genuinely POSITIVE
(IV 0.183 vs RV 0.161, IV>RV 75%). This script tests whether that premium survives once the
directional/crash exposure is hedged away — the proper way to harvest IV-RV.

Method (CPU, seconds; reuses the TAIEX path already fetched, NO new FinMind calls):
  - Short the front-monthly ATM straddle at entry (ACTUAL premium from the chain).
  - Back out the entry implied vol (Brenner-Subrahmanyam ATM, shared with the un-hedged scout).
  - DAILY delta-hedge with the index: each day compute the short-straddle's BS delta at the
    constant entry IV (r=q=0, tau decreasing), hold that many index units, rebalance to the next
    close. Option leg cash-settles to the official final settlement (intrinsic).
  - Per-cycle PnL (index pts) = (premium - intrinsic) + sum_t H_t (S_{t+1}-S_t).
  - NET subtracts the option entry cost (= un-hedged scout) AND the daily hedge-rebalance cost on
    the index notional (the NEW load-bearing assumption — swept in the report).
  - Same gate battery as the un-hedged scout (tighter equity-beta ceiling: a working hedge must
    neutralize direction).

BS delta of a short straddle hedge: the straddle delta is call_delta + put_delta = 2*N(d1) - 1;
holding H_t = 2*N(d1)-1 units of the index neutralizes the short position's delta (when spot
rises and the call side dominates, the long hedge offsets the short-call loss).

Usage:
    python scripts/research/taiwan_txo_vrp_deltahedge.py --selftest
    python scripts/research/taiwan_txo_vrp_deltahedge.py --data data/taiwan_options \
        --gates configs/taiwan_txo_vrp_deltahedge.gates.yaml
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
# Reuse the audited, math-verified helpers from the un-hedged scout (DRY; same formulas).
from scripts.research.taiwan_txo_vrp_scout import (  # noqa: E402
    _BS_ATM, bootstrap_p_mean_le_zero, pick_atm, realized_vol, sharpe,
)

log = logging.getLogger("txo_vrp_dh")
_N = NormalDist().cdf


def bs_straddle(s: float, k: float, tau: float, sigma: float) -> tuple[float, float]:
    """Black-Scholes ATM-ish straddle (call+put) price and straddle delta, r=q=0.
    Returns (price_pts, delta = 2*N(d1)-1). tau in years, sigma annualized."""
    if tau <= 0 or sigma <= 0 or s <= 0:
        # at/after expiry: intrinsic value, delta is the long-call indicator minus short-put
        price = abs(s - k)
        delta = 1.0 if s > k else (-1.0 if s < k else 0.0)
        return price, delta
    vsqrt = sigma * math.sqrt(tau)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * tau) / vsqrt
    d2 = d1 - vsqrt
    call = s * _N(d1) - k * _N(d2)
    put = k * _N(-d2) - s * _N(-d1)
    return call + put, (2.0 * _N(d1) - 1.0)


def hedge_cycle(path_dates: np.ndarray, path_s: np.ndarray, expiry: pd.Timestamp, k: float,
                sigma: float, hedge_bps: float, vol_floor: float) -> tuple[float, float]:
    """Daily delta-hedge of the SHORT straddle over the cycle path → (hedge_pnl_pts, hedge_cost_pts).
    H_t = straddle delta (units of index held); rebalanced each close; unwound at expiry."""
    hedge_pnl = 0.0
    hedge_cost = 0.0
    h_prev = 0.0
    bps = hedge_bps / 1e4
    for i in range(len(path_s) - 1):
        tau = max((expiry - pd.Timestamp(path_dates[i])).days / 365.0, vol_floor)
        _, h = bs_straddle(float(path_s[i]), k, tau, sigma)
        hedge_cost += abs(h - h_prev) * float(path_s[i]) * bps      # rebalance trade cost
        hedge_pnl += h * (float(path_s[i + 1]) - float(path_s[i]))  # hold to next close
        h_prev = h
    hedge_cost += abs(0.0 - h_prev) * float(path_s[-1]) * bps       # unwind at expiry
    return hedge_pnl, hedge_cost


def build_cycles(chains: pd.DataFrame, settle: pd.DataFrame, spot: pd.DataFrame,
                 costs: dict, ann: int) -> pd.DataFrame:
    sd = dict(zip(settle["contract_month"], settle["settle_date"]))
    sp = dict(zip(settle["contract_month"], settle["settle_price"]))
    spot = spot.sort_values("date").reset_index(drop=True)
    sdates = spot["date"].to_numpy()
    sclose = spot["close"].to_numpy(float)
    hs, fee, tax = (costs["half_spread_pts_per_leg"], costs["fee_pts_per_leg"],
                    costs["tax_rate_on_premium"])
    hedge_bps, vol_floor = costs["hedge_cost_bps"], costs["vol_floor"]

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
        sigma = premium / (_BS_ATM * s0 * math.sqrt(t_years)) if s0 > 0 and t_years > 0 else float("nan")
        if not (sigma > 0):
            continue

        # daily path from entry (inclusive) to expiry (inclusive) for the hedge
        m = (sdates >= np.datetime64(d0)) & (sdates <= np.datetime64(d1))
        path_dates, path_s = sdates[m], sclose[m]
        if len(path_s) < 5:                       # need a real path to hedge
            continue
        hedge_pnl, hedge_cost = hedge_cycle(path_dates, path_s, d1, k, sigma, hedge_bps, vol_floor)

        option_cost = 2.0 * hs + 2.0 * fee + tax * (premium + intrinsic)
        gross_unhedged = premium - intrinsic
        gross_hedged = gross_unhedged + hedge_pnl
        net = gross_hedged - option_cost - hedge_cost
        rv = realized_vol(spot, d0, d1, ann)
        s_t_spot = float(path_s[-1])
        eq_ret = math.log(s_t_spot / s0) if s_t_spot > 0 and s0 > 0 else float("nan")
        out.append({
            "contract_month": cm, "entry_date": d0, "expiry_date": d1, "dte": (d1 - d0).days,
            "spot0": s0, "strike": k, "settle": st, "premium": premium, "intrinsic": intrinsic,
            "gross_unhedged_pts": gross_unhedged, "hedge_pnl_pts": hedge_pnl,
            "gross_hedged_pts": gross_hedged, "option_cost_pts": option_cost,
            "hedge_cost_pts": hedge_cost, "net_pts": net,
            "gross_ret": gross_hedged / s0, "net_ret": net / s0,
            "iv": sigma, "rv": rv, "vrp": sigma - rv, "eq_ret": eq_ret})
    return pd.DataFrame(out).sort_values("entry_date").reset_index(drop=True)


def hedge_cost_sensitivity(cyc: pd.DataFrame, costs: dict, cpy: float,
                           bps_grid=(0.0, 1.0, 2.0, 4.0, 8.0)) -> list[dict]:
    """Net Sharpe vs the daily hedge-rebalance cost (bps) — the load-bearing hedged-variant cost.
    hedge_cost scales linearly in bps, so recompute from the stored hedge_cost at the base bps."""
    base_bps = costs["hedge_cost_bps"]
    gross_h = cyc["gross_hedged_pts"].to_numpy(float)
    opt_cost = cyc["option_cost_pts"].to_numpy(float)
    hc_base = cyc["hedge_cost_pts"].to_numpy(float)
    s0 = cyc["spot0"].to_numpy(float)
    rows = []
    for bps in bps_grid:
        scale = (bps / base_bps) if base_bps > 0 else 0.0
        net_pts = gross_h - opt_cost - hc_base * scale
        rows.append({"hedge_cost_bps": bps,
                     "net_sharpe_ann": round(sharpe(net_pts / s0, cpy), 4),
                     "mean_net_ret": round(float(np.nanmean(net_pts / s0)), 6)})
    return rows


def evaluate(cyc: pd.DataFrame, gate: dict, nm: dict, costs: dict, scout: dict,
             require_all: list[str], cpy: float, out_dir: Path) -> dict:
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
    sweep = hedge_cost_sensitivity(cyc, costs, cpy)

    checks = {
        "net_sharpe_above_floor": np.isfinite(net_sharpe) and net_sharpe >= gate["net_sharpe_floor"],
        "oos_net_sharpe_positive": np.isfinite(oos_net_sharpe) and oos_net_sharpe > gate["oos_net_sharpe_min"],
        "beats_bootstrap_null": np.isfinite(boot_p) and boot_p < gate["bootstrap_p_max"],
        "equity_beta_below_ceiling": np.isfinite(beta) and abs(beta) <= gate["max_equity_beta_abs"],
        "vrp_premium_exists": np.isfinite(vrp_pos_frac) and vrp_pos_frac >= gate["vrp_positive_frac_min"],
    }
    missing = [k for k in require_all if k not in checks]
    if missing:
        raise KeyError(f"verdict.require_all names unknown checks: {missing}")
    go = all(checks[k] for k in require_all)
    verdict = ("GO (delta-hedged TXO VRP harvestable net of hedge cost; proceed to fuller "
               "validation + Tier-2 tail audit before capital)" if go else
               "NO-GO (positive VRP is crash compensation, not harvestable alpha)")

    report = {
        "instrument": "TXO monthly ATM short straddle, DAILY delta-hedged", "n_cycles": n,
        "date_min": str(cyc["entry_date"].min()), "date_max": str(cyc["expiry_date"].max()),
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
        "hedge_cost_sensitivity": sweep,
        "gate": gate, "checks": checks, "verdict": verdict, "_go": go,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vrp_deltahedge_report.json").write_text(json.dumps(report, indent=2))
    cyc.to_parquet(out_dir / "vrp_deltahedge_cycles.parquet", index=False)

    print("\n=== TXO MONTHLY VRP - DELTA-HEDGED SCOUT ===")
    print(f"cycles: {n}  ({report['date_min'][:10]} .. {report['date_max'][:10]})")
    print(f"Sharpe (ann)   : gross {gross_sharpe:+.3f}   NET {net_sharpe:+.3f}   "
          f"(IS {is_net_sharpe:+.3f} / OOS {oos_net_sharpe:+.3f})")
    print(f"un-hedged gross {report['mean_gross_unhedged_pts']:+.0f}pt  + hedge "
          f"{report['mean_hedge_pnl_pts']:+.0f}pt  = hedged {report['mean_gross_hedged_pts']:+.0f}pt")
    print(f"costs: option {report['mean_option_cost_pts']:.1f}pt  hedge {report['mean_hedge_cost_pts']:.1f}pt"
          f"  -> mean net ret {report['mean_net_ret_per_cycle']:+.4%}")
    print(f"IV {report['mean_iv']:.3f}  RV {report['mean_rv']:.3f}  VRP {report['mean_vrp']:+.3f}  "
          f"(IV>RV {vrp_pos_frac:.0%})")
    print(f"equity beta {beta:+.3f} (corr {corr:+.3f})   bootstrap p(mean<=0) {boot_p:.4f}")
    print(f"tail: worst cycle {worst:+.2%}  CVaR5 {cvar5:+.2%}")
    print("hedge-cost sweep (bps -> net Sharpe):",
          "  ".join(f"{r['hedge_cost_bps']:.0f}={r['net_sharpe_ann']:+.2f}" for r in sweep))
    print("checks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print(f"VERDICT: {verdict}")
    print(f"written: {out_dir/'vrp_deltahedge_report.json'}")
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Delta-hedged Taiwan TXO monthly VRP scout")
    ap.add_argument("--data", default="data/taiwan_options")
    ap.add_argument("--gates", default="configs/taiwan_txo_vrp_deltahedge.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_txo_vrp_deltahedge")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    g = yaml.safe_load(((ROOT / args.gates) if not Path(args.gates).is_absolute()
                        else Path(args.gates)).read_text(encoding="utf-8"))
    scout, costs, gate, nm = g["scout"], g["costs"], g["gate"], g["null_model"]
    ann = scout["rv_window_trading_days"]
    cpy = 12.0
    require_all = g["verdict"]["require_all"]

    if args.selftest:
        return _selftest(costs, gate, nm, scout, ann, cpy, require_all)

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    chains = pd.read_parquet(data / "TXO_vrp_entry_chains.parquet")
    settle = pd.read_parquet(data / "TXO_vrp_settlement.parquet")
    spot = pd.read_parquet(data / "TAIEX_spot.parquet")
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    cyc = build_cycles(chains, settle, spot, costs, ann)
    report = evaluate(cyc, gate, nm, costs, scout, require_all, cpy, out_dir)
    return 0 if report["_go"] else 1


def _selftest(costs: dict, gate: dict, nm: dict, scout: dict, ann: int, cpy: float,
              require_all: list[str]) -> int:
    """Synthetic GBM proves BS pricing/delta + hedge loop + gate wiring end-to-end.
    A delta-hedged short straddle on a path with realized < implied should net positive."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rng = np.random.default_rng(1)
    px = 15000 * np.cumprod(1 + rng.normal(0.0002, 0.008, len(dates)))  # ~12.7% realized vol
    spot = pd.DataFrame({"date": dates, "close": px})
    months = pd.date_range("2020-02-01", "2024-12-01", freq="MS")
    rows_chain, rows_settle = [], []
    for m in months:
        entry = dates[dates >= m][0]
        exp = dates[dates >= (m + pd.Timedelta(days=28))][0]
        s0 = float(spot.loc[spot["date"] == entry, "close"].iloc[0])
        st = float(spot.loc[spot["date"] == exp, "close"].iloc[0])
        cm = m.strftime("%Y%m")
        k = round(s0 / 100) * 100
        prem = 0.030 * s0                      # IV ~ above realized -> positive VRP to harvest
        rows_chain += [
            {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
             "call_put": "call", "close": prem / 2, "volume": 1, "oi": 1},
            {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
             "call_put": "put", "close": prem / 2, "volume": 1, "oi": 1}]
        rows_settle.append({"contract_month": cm, "settle_date": exp, "settle_price": st})
    cyc = build_cycles(pd.DataFrame(rows_chain), pd.DataFrame(rows_settle), spot, costs, ann)
    out = Path(os.environ.get("TMPDIR", ".")) / "_txo_vrp_dh_selftest"
    rep = evaluate(cyc, gate, nm, costs, scout, require_all, cpy, out)
    ok = rep["n_cycles"] >= 12 and np.isfinite(rep["net_sharpe_ann"])
    log.info("SELFTEST cycles=%d net_sharpe=%s -> %s", rep["n_cycles"], rep["net_sharpe_ann"],
             "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
