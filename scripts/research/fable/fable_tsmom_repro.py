"""Fable independent re-derivation of the TSMOM 'survivor' claim.

Claim under test (results/xsec_momentum/results.json, 2026-06-03):
  pooled TSMOM, 18 ETF proxies, monthly rebalance, net Sharpe ~0.60 @ 2bps,
  4/4 asset classes positive, corr->SPY ~0.08, frictionless gap ~0.02.

This script re-implements the SPEC from scratch (vectorized, different code
path) and prices it through the unit-tested Fable oracle. It then stresses
what the legacy test did not: execution lag +1 day, short borrow, financing
on leverage above NAV, subperiods, parameter sweep, block-bootstrap CI.

Run from scripts/research/fable/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fable_oracle import backtest_weights, block_bootstrap_sharpe_ci

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "fable_verdict"
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "rates": ["TLT", "IEF", "LQD"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA"],
}
LOOKBACKS = [63, 126, 252]
SKIP = 5
VOL_WIN = 63
TARGET_VOL = 0.10
LEV_CAP = 2.0


def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(idx.to_period("M")).max().values)


def tsmom_weights(close: pd.DataFrame, lookbacks=LOOKBACKS, skip=SKIP,
                  vol_win=VOL_WIN, target_vol=TARGET_VOL) -> pd.DataFrame:
    """Vectorized independent implementation of the spec:
    score(t) = mean_L sign(P[t-skip] / P[t-skip-L] - 1)
    weight(t) = score * clip(target_vol / realized_vol[t-1], max=LEV_CAP),
    evaluated at month-end decision dates only."""
    ref = close.shift(skip)
    scores = [np.sign(ref / close.shift(skip + L) - 1.0) for L in lookbacks]
    sig = pd.concat(scores, keys=range(len(scores))).groupby(level=1).mean()
    sig = sig.reindex(close.index)

    rv = close.pct_change().rolling(vol_win).std().shift(1) * np.sqrt(252)
    scale = (target_vol / rv).clip(upper=LEV_CAP)
    w = (sig * scale).clip(-LEV_CAP, LEV_CAP)

    me = month_end_dates(close.index)
    warm = close.index[max(lookbacks) + skip + vol_win]
    return w.loc[me[me >= warm]].fillna(0.0)


def run(label: str, close: pd.DataFrame, w: pd.DataFrame, bench: pd.Series,
        cost_bps: float, **kw) -> dict:
    res = backtest_weights(close, w, cost_bps=cost_bps, **kw)
    m = res.metrics
    m["turnover_ann"] = round(res.turnover_ann, 1)
    m["avg_gross_lev"] = round(res.avg_gross_lev, 2)
    m["corr_SPY"] = round(float(res.net.corr(bench.reindex(res.net.index))), 3)
    m["label"] = label
    m["_net"] = res.net
    return m


def fmt(m: dict) -> str:
    return (f"{m['label']:42s} SR={m['sharpe']:+.3f} PF={m['pf']:.3f} "
            f"DD={m['max_dd']*100:6.1f}% vol={m['ann_vol']*100:5.1f}% "
            f"turn={m['turnover_ann']:6.1f} lev={m['avg_gross_lev']:4.2f} "
            f"corr={m['corr_SPY']:+.2f} t={m['t_stat']:+.2f}")


def main():
    close = pd.read_parquet(ROOT / "results" / "xsec_momentum" / "prices_daily.parquet").sort_index()
    bench = close["SPY"].pct_change()
    legacy = json.loads((ROOT / "results" / "xsec_momentum" / "results.json").read_text())
    out: dict = {"spec": {"lookbacks": LOOKBACKS, "skip": SKIP, "vol_win": VOL_WIN}}

    w = tsmom_weights(close)
    rows = []

    print("=" * 110)
    print("A. REPRODUCTION (pooled TSMOM monthly, my oracle vs legacy results.json)")
    print("=" * 110)
    for cost, lname in [(0.0, "frictionless"), (2.0, "standard_2bps"), (10.0, "harsh_10bps")]:
        m = run(f"pooled cost={cost}bps", close, w, bench, cost)
        leg = legacy["books"]["TSMOM_pooled_monthly"]["by_cost"][lname]
        rows.append(m)
        print(fmt(m) + f"   | legacy SR={leg['sharpe']:+.2f} PF={leg['pf']:.3f} "
                       f"corr={leg['corr_SPY']:+.2f}")
    out["reproduction"] = [{k: v for k, v in m.items() if k != "_net"} for m in rows]

    net2 = rows[1]["_net"]  # standard 2bps net, reused below

    print("\nB. PER-ASSET-CLASS net Sharpe @2bps (legacy: eq .37 rates .39 cmd .44 fx .27)")
    cls_out = {}
    for cls, tickers in UNIVERSE.items():
        wc = w[tickers]
        m = run(f"class={cls}", close[tickers], wc, bench, 2.0)
        cls_out[cls] = m["sharpe"]
        print(fmt(m))
    out["per_class_sharpe_2bps"] = cls_out

    print("\nC. STRESSES the legacy test did not run")
    stress_rows = []
    for label, kw in [
        ("exec_lag=2 (trade NEXT day's close)", dict(cost_bps=2.0, exec_lag=2)),
        ("borrow 50bps/yr on shorts", dict(cost_bps=2.0, borrow_bps_yr=50.0)),
        ("borrow 100bps + financing 50bps", dict(cost_bps=2.0, borrow_bps_yr=100.0, financing_bps_yr=50.0)),
        ("harsh: 10bps + borrow100 + fin50 + lag2", dict(cost_bps=10.0, borrow_bps_yr=100.0, financing_bps_yr=50.0, exec_lag=2)),
    ]:
        m = run(label, close, w, bench, **kw)
        stress_rows.append({k: v for k, v in m.items() if k != "_net"})
        print(fmt(m))
    out["stresses"] = stress_rows

    print("\nD. SUBPERIODS @2bps (legacy: 06-09 .57 | 10-15 .76 | 16-20 .23 | 21-26 .82)")
    subs = {}
    for a, b in [("2006", "2009"), ("2010", "2015"), ("2016", "2020"), ("2021", "2026")]:
        seg = net2.loc[a:b]
        subs[f"{a}-{b}"] = round(float(seg.mean() / seg.std() * np.sqrt(252)), 3)
    out["subperiods"] = subs
    print("   " + "  ".join(f"{k}: {v:+.2f}" for k, v in subs.items()))

    print("\nE. PARAMETER SWEEP net Sharpe @2bps (single lookbacks x vol windows)")
    sweep = {}
    for lbs in ([63], [126], [252], [63, 126, 252]):
        for vw in (21, 63, 126):
            ws = tsmom_weights(close, lookbacks=lbs, vol_win=vw)
            r = backtest_weights(close, ws, cost_bps=2.0)
            sweep[f"L={lbs}/vw={vw}"] = r.metrics["sharpe"]
    out["param_sweep"] = sweep
    for k, v in sweep.items():
        print(f"   {k:22s} {v:+.3f}")

    print("\nF. BLOCK-BOOTSTRAP 95% CI for pooled net Sharpe @2bps")
    ci = block_bootstrap_sharpe_ci(net2, n_boot=3000, block=21)
    out["bootstrap_ci_2bps"] = ci
    print(f"   {ci}")

    print("\nG. CAUSALITY A/B: same-day execution (lag0, leaky) vs honest lag1")
    leak = run("LEAKY exec_lag=0", close, w, bench, 2.0, exec_lag=0)
    out["leak_ab"] = {"lag0_sharpe": leak["sharpe"], "lag1_sharpe": rows[1]["sharpe"],
                      "gap": round(leak["sharpe"] - rows[1]["sharpe"], 3)}
    print(fmt(leak) + f"   gap={out['leak_ab']['gap']}")

    (OUT / "tsmom_reproduction.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nwritten -> {OUT / 'tsmom_reproduction.json'}")


if __name__ == "__main__":
    main()
