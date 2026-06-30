"""Step 2a — short-gamma TXO conditioning of TX intraday momentum (PROXY SCREEN).

Step 1 (`taiwan_intraday_momentum_falsification.py`) found a real but THIN intraday-momentum
edge on TX (cash-close spec: gross SR 1.17, NW-t 2.57, every yr 2019-26 +, but net ~0.4 bps/day
@ ~1bp cost). Baltussen-2021's mechanism says the effect should CONCENTRATE on days when dealers
are SHORT gamma (high hedging demand). If so, trading only short-gamma days lifts per-trade edge
above the ~1bp TX cost floor. This screens that thesis with the TXO data WE ALREADY HAVE.

PROXY CAVEATS (this is a SCREEN, not a verdict — it decides whether a full daily fetch is worth it):
  * We only have ENTRY-DAY snapshots (358 weekly dates), single expiry each — NOT a daily, all-expiry
    GEX. Net-gamma is forward-filled (causal: strictly-prior reading) to a daily long/short label.
  * Naive SqueezeMetrics dealer sign: dealers LONG call-gamma, SHORT put-gamma. Real positioning is
    noisier. A clean signal here justifies Step 2b (daily multi-expiry TaiwanOptionDaily re-fetch).
  * BS r=q=0 (Black-76, matches taiwan_txo_vrp_validation.py); IV inverted per leg from close.

VERDICT logic: short-gamma-day timing must (a) beat long-gamma-day timing, (b) show positive
r_rest x short-gamma interaction, (c) the short-gamma subset clear net cost (~1bp). Else proxy
inconclusive / NO-GO and we do NOT spend on the daily fetch.

Usage:
    python scripts/research/taiwan_intraday_momentum_gamma_conditioned.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("tx_gamma_cond")
RNG = np.random.RandomState(20260630)
COST_BPS = 1.0  # realistic TX round-trip (tax 0.4 + ~1 tick 0.6); see Step-1 grounding


# --- BS gamma + per-leg IV (r=q=0, Black-76; same convention as taiwan_txo_vrp_validation.py) ---
def _bs_price(s: float, k: float, tau: float, sig: float, is_call: bool) -> float:
    if tau <= 0 or sig <= 0 or s <= 0:
        return max(s - k, 0.0) if is_call else max(k - s, 0.0)
    vs = sig * math.sqrt(tau)
    d1 = (math.log(s / k) + 0.5 * sig * sig * tau) / vs
    d2 = d1 - vs
    return (s * norm.cdf(d1) - k * norm.cdf(d2)) if is_call else (k * norm.cdf(-d2) - s * norm.cdf(-d1))


def implied_vol_leg(price: float, s: float, k: float, tau: float, is_call: bool) -> float:
    intrinsic = max(s - k, 0.0) if is_call else max(k - s, 0.0)
    if not (price > intrinsic + 1e-9) or tau <= 0 or s <= 0:
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _bs_price(s, k, tau, mid, is_call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def bs_gamma(s: float, k: float, tau: float, sig: float) -> float:
    if tau <= 0 or sig <= 0 or s <= 0:
        return 0.0
    vs = sig * math.sqrt(tau)
    d1 = (math.log(s / k) + 0.5 * sig * sig * tau) / vs
    return float(norm.pdf(d1) / (s * vs))


def ann_sharpe(x: np.ndarray) -> float:
    sd = float(np.std(x))
    return float(np.mean(x) / sd * np.sqrt(252)) if sd > 0 and len(x) > 1 else 0.0


# --- Step-1 intraday-momentum daily build (cash-close 13:30 winning spec) ---
def build_intraday(tx_path: Path, open_hm="08:45", last_start="13:00", session_end="13:30") -> pd.DataFrame:
    df = pd.read_parquet(tx_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hm"] = df["timestamp"].dt.strftime("%H:%M")
    df["date"] = df["timestamp"].dt.normalize()
    df = df[(df["hm"] >= open_hm) & (df["hm"] <= session_end)].sort_values("timestamp")
    rows = []
    for d, day in df.groupby("date", sort=True):
        po = day.loc[day["hm"] == open_hm, "open"]
        pt = day.loc[day["hm"] == last_start, "open"]
        if not len(po) or not len(pt) or day["hm"].iloc[-1] < last_start:
            continue
        po, pt, pc = float(po.iloc[0]), float(pt.iloc[0]), float(day["close"].iloc[-1])
        if po <= 0 or pt <= 0:
            continue
        rows.append({"date": d, "r_rest": np.log(pt / po), "r_last": np.log(pc / pt)})
    return pd.DataFrame(rows)


# --- net dealer GEX per weekly snapshot date ---
def build_gex(chains_path: Path, settle_path: Path) -> pd.DataFrame:
    ch = pd.read_parquet(chains_path)
    st = pd.read_parquet(settle_path)[["contract_month", "settle_date"]]
    ch = ch.merge(st, on="contract_month", how="left")
    ch["settle_date"] = pd.to_datetime(ch["settle_date"])
    ch["entry_date"] = pd.to_datetime(ch["entry_date"])
    ch["tau"] = (ch["settle_date"] - ch["entry_date"]).dt.days / 365.0
    ch = ch[(ch["tau"] > 0) & (ch["oi"] > 0) & (ch["close"] > 0)].copy()
    recs = []
    for ed, grp in ch.groupby("entry_date", sort=True):
        gex = 0.0
        for _, r in grp.iterrows():
            is_call = r["call_put"] == "call"
            iv = implied_vol_leg(r["close"], r["entry_spot"], r["strike"], r["tau"], is_call)
            if not np.isfinite(iv):
                continue
            g = bs_gamma(r["entry_spot"], r["strike"], r["tau"], iv)
            sign = 1.0 if is_call else -1.0           # dealer long call-gamma, short put-gamma
            gex += sign * r["oi"] * g * r["entry_spot"] ** 2 * 0.01
        recs.append({"date": pd.Timestamp(ed).normalize(), "net_gex": gex})
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)


def causal_label(intraday: pd.DataFrame, gex: pd.DataFrame, max_stale_days: int = 10) -> pd.DataFrame:
    """Each trading day gets the most recent STRICTLY-PRIOR gex reading (causal), capped staleness."""
    g = gex.sort_values("date")
    merged = pd.merge_asof(intraday.sort_values("date"), g, on="date",
                           direction="backward", allow_exact_matches=False)
    # staleness cap: drop rows whose backing reading is older than max_stale_days trading days is
    # hard without a calendar join; use calendar-day proxy (<= ~16 calendar days ≈ 10 trading days).
    last_read = pd.merge_asof(intraday.sort_values("date"), g.assign(read_date=g["date"]),
                              on="date", direction="backward", allow_exact_matches=False)["read_date"]
    stale = (merged["date"] - last_read).dt.days
    merged = merged[(merged["net_gex"].notna()) & (stale <= max_stale_days * 1.6)].copy()
    return merged


def subset_stats(d: pd.DataFrame, label: str) -> dict:
    sgn = np.sign(d["r_rest"].values)
    gross = sgn * d["r_last"].values
    net = gross - COST_BPS / 1e4
    return {"label": label, "n_days": int(len(d)),
            "gross_sharpe": round(ann_sharpe(gross), 3),
            "net_sharpe_1bp": round(ann_sharpe(net), 3),
            "mean_gross_bps": round(float(gross.mean() * 1e4), 3),
            "mean_abs_last_bps": round(float(np.abs(d["r_last"].values).mean() * 1e4), 2),
            "hit_rate": round(float((gross > 0).mean()), 4)}


def newey_west_interaction(d: pd.DataFrame, lags: int = 5) -> dict:
    """r_last ~ a + b1 r_rest + b2 (r_rest * short_dummy); b2>0 => momentum stronger when short-gamma."""
    rr = d["r_rest"].values
    sd = (d["net_gex"].values < 0).astype(float)        # short-gamma dummy
    y = d["r_last"].values
    X = np.column_stack([np.ones(len(rr)), rr, rr * sd])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {"b_rest": round(float(beta[1]), 4), "t_rest": round(float(beta[1] / se[1]), 3),
            "b_interaction_short": round(float(beta[2]), 4),
            "t_interaction_short": round(float(beta[2] / se[2]), 3),
            "frac_short_days": round(float(sd.mean()), 3)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Step 2a: short-gamma TXO conditioning (proxy screen)")
    ap.add_argument("--tx", default="data/taiwan_intraday/TX_1min.parquet")
    ap.add_argument("--chains", default="data/taiwan_options_weekly/TXO_vrp_entry_chains.parquet")
    ap.add_argument("--settle", default="data/taiwan_options_weekly/TXO_vrp_settlement.parquet")
    ap.add_argument("--out", default="results/taiwan_intraday_momentum_gamma")
    args = ap.parse_args()
    rp = lambda p: (ROOT / p) if not Path(p).is_absolute() else Path(p)

    intraday = build_intraday(rp(args.tx))
    gex = build_gex(rp(args.chains), rp(args.settle))
    log.info("intraday days=%d  gex readings=%d (%s..%s)", len(intraday), len(gex),
             gex["date"].min().date(), gex["date"].max().date())
    d = causal_label(intraday, gex)
    log.info("conditioned days=%d", len(d))

    short = d[d["net_gex"] < 0]
    long_ = d[d["net_gex"] >= 0]
    # tercile split by net_gex (most-negative = most short-gamma)
    q1, q2 = d["net_gex"].quantile([1 / 3, 2 / 3])
    most_short = d[d["net_gex"] <= q1]
    most_long = d[d["net_gex"] >= q2]

    stats = {
        "all_conditioned": subset_stats(d, "all conditioned days"),
        "short_gamma_neg": subset_stats(short, "short-gamma (net_gex<0)"),
        "long_gamma_pos": subset_stats(long_, "long-gamma (net_gex>=0)"),
        "most_short_tercile": subset_stats(most_short, "most-short tercile"),
        "most_long_tercile": subset_stats(most_long, "most-long tercile"),
    }
    interaction = newey_west_interaction(d)

    checks = {
        "short_beats_long_gross": stats["short_gamma_neg"]["gross_sharpe"] > stats["long_gamma_pos"]["gross_sharpe"],
        "positive_short_interaction": interaction["t_interaction_short"] > 1.5,
        "short_subset_clears_cost": stats["short_gamma_neg"]["net_sharpe_1bp"] >= 0.5,
    }
    go = all(checks.values())
    verdict = ("GO -> justify Step 2b daily multi-expiry GEX fetch" if go else
               "PROXY INCONCLUSIVE / NO-GO -> conditioning does not lift the thin TX edge")

    report = {"proxy_caveats": "single-expiry weekly snapshot, naive dealer sign, fwd-filled; SCREEN only",
              "n_intraday_days": int(len(intraday)), "n_gex_readings": int(len(gex)),
              "n_conditioned_days": int(len(d)), "cost_bps_roundtrip": COST_BPS,
              "subsets": stats, "interaction_regression": interaction,
              "checks": checks, "verdict": verdict}
    out = rp(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "gamma_conditioned_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== STEP 2a: SHORT-GAMMA CONDITIONING (PROXY SCREEN) ===")
    print(f"gex readings={len(gex)}  conditioned days={len(d)}  (cost {COST_BPS}bps RT)")
    for k in ("all_conditioned", "short_gamma_neg", "long_gamma_pos", "most_short_tercile", "most_long_tercile"):
        s = stats[k]
        print(f"  {s['label']:<26} n={s['n_days']:>4}  grossSR {s['gross_sharpe']:+.2f}  "
              f"netSR@1bp {s['net_sharpe_1bp']:+.2f}  mean {s['mean_gross_bps']:+.2f}bps  "
              f"|last| {s['mean_abs_last_bps']:.1f}bps  hit {s['hit_rate']:.3f}")
    print(f"\ninteraction r_last ~ r_rest + r_rest*short:  b_rest {interaction['b_rest']:+.3f} (t {interaction['t_rest']:+.2f}), "
          f"b_short {interaction['b_interaction_short']:+.3f} (t {interaction['t_interaction_short']:+.2f}), "
          f"frac_short {interaction['frac_short_days']:.2f}")
    print("checks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print(f"VERDICT: {verdict}")
    print(f"written: {out/'gamma_conditioned_report.json'}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
