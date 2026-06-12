"""Fable decisive experiment: does the TSMOM rule generalize to an UNTOUCHED universe?

The curated-18 result reproduces exactly (fable_tsmom_repro.py), but the
universe was hand-picked — the one remaining cheap overfit channel. This test
runs the SAME frozen rule (sign-momentum 3/6/12m skip-5, vol-scale 10% cap 2,
month-end, T+1, costs) on 32 liquid US-listed ETFs the project never used,
selected by a mechanical rule (liquidity, inception <= 2012, no
leveraged/inverse, class coverage), written down BEFORE running.

PRE-REGISTERED GATES (set before first run):
  G1: pooled untouched-universe net Sharpe @2bps >= 0.30
  G2: >= 60% of instruments have positive single-asset net Sharpe @2bps
  G3: frictionless minus net(2bps) Sharpe gap <= 0.10
PASS = curated-universe selection bias bounded; the trend premium is structural.
FAIL on G1 => the 0.60 is substantially a universe artifact: downgrade verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fable_oracle import backtest_weights, block_bootstrap_sharpe_ci
from fable_tsmom_repro import tsmom_weights

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fable_verdict"
CACHE = OUT / "untouched_universe_prices.parquet"

UNTOUCHED = {
    "equity_sector": ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"],
    "equity_intl": ["EWJ", "EWG", "EWU", "EWA", "EWC", "EWZ", "EWY", "FXI", "EWT", "EWH"],
    "bonds_credit": ["SHY", "AGG", "TIP", "HYG", "EMB", "BWX", "MUB"],
    "commodity": ["PPLT", "PALL", "UNG", "CPER"],
    "reit": ["VNQ", "RWX"],
}
ALL = [t for v in UNTOUCHED.values() for t in v]


def get_prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    px = yf.download(ALL, start="2006-01-01", end="2026-06-10",
                     progress=False, auto_adjust=True)["Close"]
    px = px.dropna(how="all").sort_index()
    px.to_parquet(CACHE)
    return px


def main():
    px = get_prices()
    px = px[[t for t in ALL if t in px.columns]]
    print(f"untouched universe: {px.shape[1]} tickers x {px.shape[0]} days "
          f"({px.index.min().date()} -> {px.index.max().date()})")

    w = tsmom_weights(px)
    res = {}
    for cost, name in [(0.0, "frictionless"), (2.0, "net_2bps"), (10.0, "harsh_10bps")]:
        r = backtest_weights(px, w, cost_bps=cost)
        res[name] = r.metrics | {"turnover_ann": round(r.turnover_ann, 1),
                                 "avg_gross_lev": round(r.avg_gross_lev, 2)}
        print(f"pooled {name:13s} SR={r.metrics['sharpe']:+.3f} PF={r.metrics['pf']:.3f} "
              f"t={r.metrics['t_stat']:+.2f} turn={r.turnover_ann:.0f}")
    net2 = backtest_weights(px, w, cost_bps=2.0).net

    # per-instrument single-asset books
    per_asset = {}
    for t in px.columns:
        rt = backtest_weights(px[[t]], w[[t]], cost_bps=2.0)
        per_asset[t] = rt.metrics["sharpe"]
    n_pos = sum(1 for v in per_asset.values() if v > 0)
    pct_pos = n_pos / len(per_asset)

    # per-class pooled
    per_class = {}
    for cls, ts in UNTOUCHED.items():
        ts = [t for t in ts if t in px.columns]
        rc = backtest_weights(px[ts], w[ts], cost_bps=2.0)
        per_class[cls] = rc.metrics["sharpe"]

    subs = {}
    for a, b in [("2006", "2009"), ("2010", "2015"), ("2016", "2020"), ("2021", "2026")]:
        seg = net2.loc[a:b].dropna()
        subs[f"{a}-{b}"] = round(float(seg.mean() / seg.std() * (252 ** 0.5)), 3) if len(seg) > 50 else None

    ci = block_bootstrap_sharpe_ci(net2, n_boot=3000, block=21)

    g1 = res["net_2bps"]["sharpe"] >= 0.30
    g3_gap = round(res["frictionless"]["sharpe"] - res["net_2bps"]["sharpe"], 3)
    gates = {
        "G1_pooled_net_sharpe_ge_0.30": {"value": res["net_2bps"]["sharpe"], "pass": bool(g1)},
        "G2_pct_assets_positive_ge_60": {"value": round(pct_pos * 100, 1), "n_pos": n_pos,
                                          "n_total": len(per_asset), "pass": bool(pct_pos >= 0.60)},
        "G3_cost_gap_le_0.10": {"value": g3_gap, "pass": bool(g3_gap <= 0.10)},
    }
    overall = all(g["pass"] for g in gates.values())

    out = {"universe": UNTOUCHED, "pooled": res, "per_class_sharpe_2bps": per_class,
           "per_asset_sharpe_2bps": per_asset, "subperiods": subs,
           "bootstrap_ci_2bps": ci, "gates": gates,
           "overall": "PASS" if overall else "FAIL"}
    (OUT / "untouched_universe_result.json").write_text(json.dumps(out, indent=2))

    print("\nper-class:", {k: round(v, 2) for k, v in per_class.items()})
    print("subperiods:", subs)
    print("bootstrap:", ci)
    print(f"per-asset positive: {n_pos}/{len(per_asset)} ({pct_pos*100:.0f}%)")
    worst = sorted(per_asset.items(), key=lambda kv: kv[1])[:5]
    best = sorted(per_asset.items(), key=lambda kv: kv[1])[-5:]
    print("worst 5:", [(t, round(v, 2)) for t, v in worst])
    print("best  5:", [(t, round(v, 2)) for t, v in best])
    print("\nGATES:")
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print(f"OVERALL: {out['overall']}")


if __name__ == "__main__":
    main()
