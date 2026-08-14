"""BALLAST G1b — coverage-matched survivorship inflation test.

Question: the P2 core beat SPY by +0.104 Sharpe on 2007-2014, on a panel whose coverage over that
window is only ~66% of the actual index. Is that margin real, or is it manufactured by the missing
34%? This runs entirely inside the TRAINING window, so it costs nothing from the single-shot OOS
budget.

**Method — an index-exit ladder, built from real prices.**

The obvious construction (progressively require longer *survival*) is not available: only 18 of the
panel's 774 names have a price series that ends during 2007-2025. yfinance retains essentially
nothing that died, which is exactly why the 428 missing names are missing. There is no internal
survival gradient to exploit.

What IS available is the *index-exit* cohort: 402 names traded in 2007-2014, and 86 of them are no
longer S&P 500 members by 2026 — yet we can still price them. So the ladder requires membership at
successively later vintages ``V``:

    universe(V) = names that were STILL S&P 500 members on V,   V in {2015, 2018, 2022, 2026}

Higher ``V`` = a stricter survival requirement = a more survivorship-biased panel, and the whole
ladder is measured on real returns. Regressing the core's Sharpe on the resulting coverage gives
d(Sharpe)/d(coverage) — the inflation slope.

**What this bounds, and in which direction.** Index deletion is a MILDER event than the bankruptcies
and distressed acquisitions that make up the 428 fully-missing names: deletions include mergers,
which are neutral to positive. So the slope measured here is a **LOWER BOUND** on the true
inflation. If even the mild cohort moves the Sharpe materially, the severe cohort moves it more.
The extrapolation to full coverage is reported with that caveat and with the extrapolation ratio
made explicit — it is an estimate, not a measurement.

    python scripts/research/ballast_g1b_inflation.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data import sp500_pit_panel as pit  # noqa: E402
from finrl_pro_ds.portfolio.long_only import (  # noqa: E402
    PortfolioConfig, active_stats, backtest, performance,
)
from finrl_pro_ds.signals.library import ballast as bs  # noqa: E402
from scripts.research.ballast_linear_core import benchmark_returns  # noqa: E402

log = logging.getLogger("ballast_g1b")
OUT = ROOT / "results" / "ballast_v1"
PANEL = ROOT / "data" / "raw" / "equity_panel" / "_ballast_pit_1996.pkl"

WARMUP_START = np.datetime64("2005-01-01")   # 2y warmup for the 252d windows
EVAL_START = np.datetime64("2007-01-01")
EVAL_END = np.datetime64("2014-12-31")
VINTAGES = ["2015-01-01", "2018-01-01", "2022-01-01", "2026-06-01"]


def slice_panel(panel, lo_date, hi_date):
    lo = int(np.searchsorted(panel.dates, lo_date))
    hi = int(np.searchsorted(panel.dates, hi_date, side="right"))
    s = slice(lo, hi)
    return dataclasses.replace(
        panel, dates=panel.dates[s], open=panel.open[s], high=panel.high[s], low=panel.low[s],
        close=panel.close[s], volume=panel.volume[s], active=panel.active[s],
        adv_usd=panel.adv_usd[s])


def members_at(vintage: str) -> frozenset[str]:
    snap_d, snap_m, _ = pit.load_membership(pit.MEMBERS_CSV, start="1996-01-02")
    snap = np.asarray(snap_d, dtype="datetime64[ns]")
    k = int(np.searchsorted(snap, np.datetime64(vintage), side="right")) - 1
    return snap_m[max(0, k)]


def coverage(panel, true_counts, lo, hi) -> float:
    """Mean (active names) / (actual index members) over the evaluation window."""
    n_act = panel.active[lo:hi].sum(1)
    tc = true_counts[lo:hi]
    m = tc > 0
    return float((n_act[m] / tc[m]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=75)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    with open(PANEL, "rb") as fh:
        full = pickle.load(fh)
    panel = slice_panel(full, WARMUP_START, EVAL_END)
    log.info("sliced panel T=%d N=%d", panel.T, panel.N)

    # True index size per date, from the membership CSV (independent of prices).
    snap_d, snap_m, _ = pit.load_membership(pit.MEMBERS_CSV, start="1996-01-02")
    snap = np.asarray(snap_d, dtype="datetime64[ns]")
    ki = np.searchsorted(snap, panel.dates, side="right") - 1
    true_counts = np.array([len(snap_m[i]) if i >= 0 else 0 for i in ki])

    lo = int(np.searchsorted(panel.dates, EVAL_START))
    hi = panel.T
    bench = benchmark_returns(panel.dates)[lo:hi]
    bench_perf = performance(bench)
    cfg = PortfolioConfig(k=args.k, cost_bps=args.cost_bps)

    tickers = np.array(panel.tickers)
    rows = []

    def evaluate(label: str, keep_mask: np.ndarray) -> dict:
        """Restrict the universe to ``keep_mask`` (over names), recompute sleeves, backtest."""
        p = dataclasses.replace(panel, active=panel.active & keep_mask[None, :])
        cov = coverage(p, true_counts, lo, hi)
        sleeves = bs.compute_sleeves(p)                    # market-only: fundamentals are ~absent
        out = {"universe": label, "n_names": int(keep_mask.sum()), "coverage": round(cov, 4),
               "mean_active_per_day": round(float(p.active[lo:hi].sum(1).mean()), 1)}
        for name in ["core"] + list(bs.MARKET_SLEEVES):
            names = list(bs.MARKET_SLEEVES) if name == "core" else [name]
            blend = np.nanmean(np.stack([sleeves[n] for n in names]), axis=0)
            res = backtest(p, blend, cfg, start=EVAL_START, end=EVAL_END)
            perf, act = performance(res.returns), active_stats(res.returns, bench)
            out[f"{name}_sharpe"] = round(perf["sharpe"], 4)
            out[f"{name}_ir"] = round(act["information_ratio"], 4)
            out[f"{name}_maxdd"] = round(perf["max_drawdown"], 4)
        rows.append(out)
        log.info("%s: coverage %.3f  core sharpe %.3f  core IR %.3f",
                 label, cov, out["core_sharpe"], out["core_ir"])
        return out

    # Rung 0: the panel as-is (the most complete universe free data provides).
    evaluate("baseline_all_priceable", np.ones(panel.N, dtype=bool))
    # Rungs 1..n: require membership at successively later vintages.
    for v in VINTAGES:
        mem = members_at(v)
        evaluate(f"member_at_{v[:4]}", np.array([t in mem for t in tickers], dtype=bool))

    df = pd.DataFrame(rows).sort_values("coverage").reset_index(drop=True)

    # ---- inflation slope ---------------------------------------------------------------- #
    est = {}
    base = df[df["universe"] == "baseline_all_priceable"].iloc[0]
    for metric in ["core_sharpe", "core_ir"] + [f"{s}_ir" for s in bs.MARKET_SLEEVES]:
        slope, intercept = np.polyfit(df["coverage"], df[metric], 1)
        r = float(np.corrcoef(df["coverage"], df[metric])[0, 1])
        at_full = intercept + slope * 1.0
        est[metric] = {
            "slope_per_unit_coverage": round(float(slope), 4),
            "r": round(r, 3),
            "observed_at_baseline": round(float(base[metric]), 4),
            "baseline_coverage": round(float(base["coverage"]), 4),
            "extrapolated_at_full_coverage": round(float(at_full), 4),
            "implied_inflation": round(float(base[metric] - at_full), 4),
        }

    infl = est["core_sharpe"]["implied_inflation"]
    measured_span = float(df["coverage"].max() - df["coverage"].min())
    extrap_span = 1.0 - float(base["coverage"])
    gates = __import__("yaml").safe_load(
        (ROOT / "configs" / "ballast_v1.gates.yaml").read_text())["gates"]
    thresh = float(gates["g1b_coverage_matched_inflation"]["max_sharpe_inflation"])
    p2_uplift = 0.104          # core_market_only 2007-2014, from results/ballast_v1/p2_linear_core.json

    payload = {
        "method": "index-exit ladder (real prices); survival ladder unavailable - only 18/774 "
                  "names have a series ending 2007-2025",
        "direction": "LOWER BOUND - index deletion is milder than the bankruptcies that make up "
                     "the 428 fully-missing names",
        "window": "2007-01-01..2014-12-31 (training only; OOS untouched)",
        "ladder": df.to_dict(orient="records"),
        "estimates": est,
        "measured_coverage_span": round(measured_span, 4),
        "extrapolation_span": round(extrap_span, 4),
        "extrapolation_ratio": round(extrap_span / measured_span, 2) if measured_span > 0 else None,
        "gate": {"max_sharpe_inflation": thresh,
                 "implied_inflation": infl,
                 "verdict": "PASS" if infl < thresh else "FAIL"},
        "p2_margin_at_risk": {"p2_core_sharpe_uplift_vs_spy": p2_uplift,
                              "inflation_estimate": infl,
                              "margin_survives": bool(p2_uplift > infl)},
        "benchmark_spy_sharpe": round(bench_perf["sharpe"], 4),
    }
    (OUT / "g1b_inflation.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- report ------------------------------------------------------------------------- #
    print("\n=== BALLAST G1b — coverage-matched survivorship inflation (2007-2014) ===")
    print("  method: index-exit ladder on REAL prices (survival ladder unavailable: only")
    print("          18/774 names have a series ending 2007-2025)")
    print("  direction: LOWER BOUND (deletion is milder than bankruptcy)\n")
    cols = ["universe", "n_names", "coverage", "mean_active_per_day", "core_sharpe", "core_ir",
            "core_maxdd"]
    print(df[cols].to_string(index=False))
    print("\n  per-sleeve IR across the ladder:")
    print(df[["coverage"] + [f"{s}_ir" for s in bs.MARKET_SLEEVES]].to_string(index=False))

    print("\n  INFLATION SLOPE (metric regressed on coverage):")
    for m, e in est.items():
        print(f"    {m:16s} slope {e['slope_per_unit_coverage']:+7.3f} (r={e['r']:+.2f})  "
              f"observed {e['observed_at_baseline']:+.3f} @cov {e['baseline_coverage']:.3f}  "
              f"-> extrapolated {e['extrapolated_at_full_coverage']:+.3f} @cov 1.00  "
              f"= inflation {e['implied_inflation']:+.3f}")

    print(f"\n  measured coverage span {measured_span:.3f}; extrapolating a further "
          f"{extrap_span:.3f} (ratio {payload['extrapolation_ratio']}x) — an ESTIMATE, not a "
          f"measurement")
    print(f"\n  G1b GATE (max_sharpe_inflation={thresh}): implied inflation {infl:+.3f} "
          f"=> {payload['gate']['verdict']}")
    print(f"  P2 margin at risk: uplift vs SPY was {p2_uplift:+.3f}; inflation estimate "
          f"{infl:+.3f} => margin "
          f"{'SURVIVES' if payload['p2_margin_at_risk']['margin_survives'] else 'DOES NOT SURVIVE'}")
    print(f"\n  wrote {OUT / 'g1b_inflation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
