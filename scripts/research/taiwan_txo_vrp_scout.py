"""Taiwan TXO (台指選擇權) variance-risk-premium (VRP) signal scout — cost-survivable short-vol?

The cheap CPU gate (mirrors the TX intraday canary) before any deeper VRP validation. The
standing project verdict is VRP = NO-GO on COST REALISM (Deribit USDC-contamination; clean BTC
straddle -0.25). This re-tests the thesis on Taiwan index options, where TXO is cash-settled to
an official final settlement price — so a held-to-expiry straddle pays NO exit spread, the most
cost-favorable honest VRP test available.

Strategy under test (pre-registered in configs/taiwan_txo_vrp_scout.gates.yaml):
  - Enter the front-MONTHLY ATM straddle on the first trading day after the prior monthly expiry
    (~30 DTE). SHORT 1 call + 1 put at strike K = nearest listed strike to spot S0.
  - Hold to cash settlement; payoff = official final settlement level S_T.
  - Un-hedged (the classic raw-VRP test; delta-hedge is a later refinement).
  - Per-cycle PnL (index pts) = premium_collected - |S_T - K|.
  - NET PnL subtracts a CONSERVATIVE modeled cost (entry half-spread x2 legs + fees + 期交稅);
    NO exit spread (cash settlement). Return = NET PnL / S0 (return on notional).
  - Annualized Sharpe by ~12 monthly cycles/yr.

Diagnostics: ATM implied vol (Brenner-Subrahmanyam ATM approximation) vs realized vol over the
holding window → the VRP itself; equity-beta of straddle returns vs TAIEX (is the "premium" just
disguised equity beta — the wheel/0DTE NO-GO failure mode); IS/OOS split; IID bootstrap p-value;
short-vol tail (worst cycle, CVaR5).

Inputs are produced by scripts/data/fetch_taiwan_options_finmind.py.

Usage:
    python scripts/research/taiwan_txo_vrp_scout.py --selftest          # synthetic, no data
    python scripts/research/taiwan_txo_vrp_scout.py \
        --data data/taiwan_options --gates configs/taiwan_txo_vrp_scout.gates.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("txo_vrp_scout")
RNG = np.random.RandomState(20260629)

# Brenner-Subrahmanyam ATM straddle <-> vol constant: straddle ~= c * S * sigma * sqrt(T),
# c = 2 * phi(0) = 2/sqrt(2*pi) = 0.79788..., valid for at-the-(forward-)money, rates~0.
_BS_ATM = 2.0 / math.sqrt(2.0 * math.pi)


def pick_atm(chain: pd.DataFrame, spot: float) -> tuple[float, float, float] | None:
    """Strike K nearest spot with BOTH a call and a put present → (K, call_close, put_close)."""
    calls = chain[chain["call_put"] == "call"].set_index("strike")["close"]
    puts = chain[chain["call_put"] == "put"].set_index("strike")["close"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return None
    k = float(min(common, key=lambda s: abs(s - spot)))
    return k, float(calls[k]), float(puts[k])


def realized_vol(spot: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, ann: int) -> float:
    """Annualized close-to-close realized vol of TAIEX over (d0, d1] (the holding window)."""
    seg = spot[(spot["date"] > d0) & (spot["date"] <= d1)]["close"].to_numpy(float)
    if len(seg) < 3:
        return float("nan")
    r = np.diff(np.log(seg))
    return float(np.std(r, ddof=1) * math.sqrt(ann))


def build_cycles(chains: pd.DataFrame, settle: pd.DataFrame, spot: pd.DataFrame,
                 costs: dict, ann: int) -> pd.DataFrame:
    """One row per monthly cycle with gross/net PnL, returns, IV, RV, and the equity move."""
    sd = dict(zip(settle["contract_month"], settle["settle_date"]))
    sp = dict(zip(settle["contract_month"], settle["settle_price"]))
    spot_close = dict(zip(spot["date"], spot["close"]))
    hs, fee, tax = (costs["half_spread_pts_per_leg"], costs["fee_pts_per_leg"],
                    costs["tax_rate_on_premium"])

    out = []
    for cm, g in chains.groupby("contract_month"):
        if cm not in sd or cm not in sp:
            continue
        d0 = pd.Timestamp(g["entry_date"].iloc[0])
        d1 = pd.Timestamp(sd[cm])
        s0 = float(g["entry_spot"].iloc[0])
        st = float(sp[cm])
        if not (d0 < d1) or not (s0 > 0) or not (st > 0):  # causality + sanity
            continue
        atm = pick_atm(g, s0)
        if atm is None:
            continue
        k, c, p = atm
        premium = c + p
        intrinsic = abs(st - k)                                   # cash-settlement payoff
        gross = premium - intrinsic                               # short straddle, index pts
        # Conservative cost: entry half-spread on 2 short legs + per-leg fee*2 + tax on premium
        # collected (entry) and on settled intrinsic. NO exit spread (cash settlement).
        cost = 2.0 * hs + 2.0 * fee + tax * (premium + intrinsic)
        net = gross - cost
        t_years = max((d1 - d0).days, 1) / 365.0
        iv = premium / (_BS_ATM * s0 * math.sqrt(t_years)) if s0 > 0 and t_years > 0 else float("nan")
        rv = realized_vol(spot, d0, d1, ann)
        s_t_spot = float(spot_close.get(d1, st))                  # TAIEX close near expiry
        eq_ret = math.log(s_t_spot / s0) if s_t_spot > 0 and s0 > 0 else float("nan")
        out.append({
            "contract_month": cm, "entry_date": d0, "expiry_date": d1, "dte": (d1 - d0).days,
            "spot0": s0, "strike": k, "settle": st, "premium": premium, "intrinsic": intrinsic,
            "gross_pts": gross, "cost_pts": cost, "net_pts": net,
            "gross_ret": gross / s0, "net_ret": net / s0,
            "iv": iv, "rv": rv, "vrp": iv - rv, "eq_ret": eq_ret})
    return pd.DataFrame(out).sort_values("entry_date").reset_index(drop=True)


def sharpe(r: np.ndarray, cycles_per_year: float) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) < 1e-12:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(cycles_per_year))


def cost_sensitivity(cyc: pd.DataFrame, costs: dict, cpy: float,
                     spreads=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)) -> list[dict]:
    """Net Sharpe vs the MODELED entry half-spread — the load-bearing assumption (no real bid/ask).
    Recomputed from the cycle frame (gross_pts, premium, intrinsic, spot0); no refetch. The 0.0pt
    row is the frictionless ceiling; the verdict must not hinge on a single optimistic spread."""
    fee, tax = costs["fee_pts_per_leg"], costs["tax_rate_on_premium"]
    gross = cyc["gross_pts"].to_numpy(float)
    prem = cyc["premium"].to_numpy(float)
    intr = cyc["intrinsic"].to_numpy(float)
    s0 = cyc["spot0"].to_numpy(float)
    rows = []
    for hs in spreads:
        net_pts = gross - (2.0 * hs + 2.0 * fee + tax * (prem + intr))
        rows.append({"half_spread_pts": hs,
                     "net_sharpe_ann": round(sharpe(net_pts / s0, cpy), 4),
                     "mean_net_ret": round(float(np.nanmean(net_pts / s0)), 6)})
    return rows


def bootstrap_p_mean_le_zero(r: np.ndarray, n_boot: int) -> float:
    """IID bootstrap: p that the mean net cycle return is <= 0 (one-sided)."""
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float("nan")
    means = np.array([np.mean(r[RNG.randint(0, n, n)]) for _ in range(n_boot)])
    return float((means <= 0).mean())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan TXO monthly VRP signal scout")
    ap.add_argument("--data", default="data/taiwan_options")
    ap.add_argument("--gates", default="configs/taiwan_txo_vrp_scout.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_txo_vrp")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    g = yaml.safe_load(((ROOT / args.gates) if not Path(args.gates).is_absolute()
                        else Path(args.gates)).read_text(encoding="utf-8"))
    scout, costs, gate, nm = g["scout"], g["costs"], g["gate"], g["null_model"]
    ann = scout["rv_window_trading_days"]
    cycles_per_year = 12.0  # front-monthly TXO

    require_all = g["verdict"]["require_all"]
    if args.selftest:
        return _selftest(costs, gate, nm, ann, cycles_per_year, require_all)

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    chains = pd.read_parquet(data / "TXO_vrp_entry_chains.parquet")
    settle = pd.read_parquet(data / "TXO_vrp_settlement.parquet")
    spot = pd.read_parquet(data / "TAIEX_spot.parquet")
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cyc = build_cycles(chains, settle, spot, costs, ann)
    report = evaluate(cyc, gate, nm, costs, scout, require_all, cycles_per_year, out_dir)
    return 0 if report["_go"] else 1


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

    # IS/OOS split by time
    split = int(n * scout["oos_split_frac"])
    oos_net_sharpe = sharpe(net[split:], cpy)
    is_net_sharpe = sharpe(net[:split], cpy)

    # equity beta: net straddle return regressed on the contemporaneous TAIEX log return
    mask = np.isfinite(net) & np.isfinite(eq)
    beta = float(np.polyfit(eq[mask], net[mask], 1)[0]) if mask.sum() >= 3 else float("nan")
    corr = float(np.corrcoef(eq[mask], net[mask])[0, 1]) if mask.sum() >= 3 else float("nan")

    vrp = cyc["vrp"].to_numpy(float)
    vrp_pos_frac = float(np.mean(vrp[np.isfinite(vrp)] > 0)) if np.isfinite(vrp).any() else float("nan")

    boot_p = bootstrap_p_mean_le_zero(net, nm["n_boot"])
    worst = float(np.nanmin(net))
    cvar5 = float(np.mean(np.sort(net[np.isfinite(net)])[:max(1, int(0.05 * n))]))
    cost_sweep = cost_sensitivity(cyc, costs, cpy)

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
    verdict = ("GO (Taiwan VRP survives conservative cost; proceed to fuller validation + Tier-2 "
               "tail audit before capital)" if go else
               "NO-GO (cost-killed / disguised equity beta - the standing VRP class)")

    report = {
        "instrument": "TXO monthly ATM short straddle", "n_cycles": n,
        "date_min": str(cyc["entry_date"].min()), "date_max": str(cyc["expiry_date"].max()),
        "gross_sharpe_ann": round(gross_sharpe, 4), "net_sharpe_ann": round(net_sharpe, 4),
        "is_net_sharpe_ann": round(is_net_sharpe, 4), "oos_net_sharpe_ann": round(oos_net_sharpe, 4),
        "mean_net_ret_per_cycle": round(float(np.nanmean(net)), 6),
        "mean_gross_pts": round(float(cyc["gross_pts"].mean()), 2),
        "mean_cost_pts": round(float(cyc["cost_pts"].mean()), 2),
        "mean_premium_pts": round(float(cyc["premium"].mean()), 2),
        "mean_iv": round(float(np.nanmean(cyc["iv"])), 4), "mean_rv": round(float(np.nanmean(cyc["rv"])), 4),
        "mean_vrp": round(float(np.nanmean(vrp)), 4), "vrp_positive_frac": round(vrp_pos_frac, 3),
        "equity_beta": round(beta, 4), "equity_corr": round(corr, 4),
        "bootstrap_p_mean_le_zero": round(boot_p, 4),
        "worst_cycle_net_ret": round(worst, 5), "cvar5_net_ret": round(cvar5, 5),
        "cost_sensitivity": cost_sweep,
        "gate": gate, "checks": checks, "verdict": verdict, "_go": go,
    }
    (out_dir / "vrp_scout_report.json").write_text(json.dumps(report, indent=2))
    cyc.to_parquet(out_dir / "vrp_cycles.parquet", index=False)

    print("\n=== TXO MONTHLY VRP SIGNAL SCOUT ===")
    print(f"cycles: {n}  ({report['date_min'][:10]} .. {report['date_max'][:10]})")
    print(f"Sharpe (ann)   : gross {gross_sharpe:+.3f}   NET {net_sharpe:+.3f}   "
          f"(IS {is_net_sharpe:+.3f} / OOS {oos_net_sharpe:+.3f})")
    print(f"premium {report['mean_premium_pts']:.0f}pt  gross {report['mean_gross_pts']:+.0f}pt  "
          f"cost {report['mean_cost_pts']:.1f}pt  -> mean net ret {report['mean_net_ret_per_cycle']:+.4%}")
    print(f"IV {report['mean_iv']:.3f}  RV {report['mean_rv']:.3f}  VRP {report['mean_vrp']:+.3f}  "
          f"(IV>RV {vrp_pos_frac:.0%})")
    print(f"equity beta {beta:+.3f} (corr {corr:+.3f})   bootstrap p(mean<=0) {boot_p:.4f}")
    print(f"tail: worst cycle {worst:+.2%}  CVaR5 {cvar5:+.2%}")
    print("cost sweep (half-spread pt -> net Sharpe):",
          "  ".join(f"{r['half_spread_pts']:.1f}={r['net_sharpe_ann']:+.2f}" for r in cost_sweep))
    print("checks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print(f"VERDICT: {verdict}")
    print(f"written: {out_dir/'vrp_scout_report.json'}")
    return report


def _selftest(costs: dict, gate: dict, nm: dict, ann: int, cpy: float, require_all: list[str]) -> int:
    """Synthetic cycles prove the PnL/Sharpe/beta/cost wiring end-to-end (no data, no token).
    Construct a known short-vol premium: settlement lands inside the straddle most months."""
    dates = pd.bdate_range("2020-01-02", "2024-12-31")
    rng = np.random.default_rng(0)
    px = 15000 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
    spot = pd.DataFrame({"date": dates, "close": px})
    months = pd.date_range("2020-02-01", "2024-12-01", freq="MS")
    rows_chain, rows_settle = [], []
    for m in months:
        exp = dates[dates >= (m + pd.Timedelta(days=18))][0]
        entry = dates[dates >= m][0]
        s0 = float(spot.loc[spot["date"] == entry, "close"].iloc[0])
        st = float(spot.loc[spot["date"] == exp, "close"].iloc[0])
        cm = m.strftime("%Y%m")
        k = round(s0 / 100) * 100
        # premium priced ABOVE the realized move on average -> a positive VRP
        prem = 0.022 * s0
        rows_chain += [
            {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
             "call_put": "call", "close": prem / 2, "volume": 1, "oi": 1},
            {"contract_month": cm, "entry_date": entry, "entry_spot": s0, "strike": float(k),
             "call_put": "put", "close": prem / 2, "volume": 1, "oi": 1}]
        rows_settle.append({"contract_month": cm, "settle_date": exp, "settle_price": st})
    chains = pd.DataFrame(rows_chain)
    settle = pd.DataFrame(rows_settle)
    cyc = build_cycles(chains, settle, spot, costs, ann)
    out = Path(os.environ.get("TMPDIR", ".")) / "_txo_vrp_scout_selftest"
    out.mkdir(parents=True, exist_ok=True)
    rep = evaluate(cyc, gate, nm, costs, {"oos_split_frac": 0.5}, require_all, cpy, out)
    ok = rep["n_cycles"] >= 12 and np.isfinite(rep["net_sharpe_ann"])
    log.info("SELFTEST cycles=%d net_sharpe=%s -> %s", rep["n_cycles"], rep["net_sharpe_ann"],
             "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
