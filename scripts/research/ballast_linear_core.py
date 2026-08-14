"""BALLAST P2 runner — the frozen linear core, evaluated against SPY on TRAIN+VALIDATION only.

This is the project's cheap kill point (design §8). If the deterministic six-sleeve core cannot beat
SPY risk-adjusted here, the RL layer (P4+) is not worth building.

**The OOS window 2015-2025 is NOT touched by this script.** It is single-shot and belongs to P6.
The `--window` argument refuses to accept it.

    python scripts/research/ballast_linear_core.py [--start 2007-01-01] [--end 2014-12-31]
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finrl_pro_ds.data import cross_asset_loader as cal  # noqa: E402
from finrl_pro_ds.data import fundamentals as fnd  # noqa: E402
from finrl_pro_ds.portfolio.long_only import (  # noqa: E402
    PortfolioConfig, active_stats, backtest, performance,
)
from finrl_pro_ds.signals.library import ballast as bs  # noqa: E402

log = logging.getLogger("ballast_p2")
OUT = ROOT / "results" / "ballast_v1"
PANEL = ROOT / "data" / "raw" / "equity_panel" / "_ballast_pit_1996.pkl"
FUND = ROOT / "data" / "raw" / "fundamentals" / "_ballast_fundamentals.pkl"
UNADJ = ROOT / "data" / "raw" / "equity_panel" / "_ballast_unadjusted_close.pkl"

OOS_START = np.datetime64("2015-01-01")     # reserved; P2 must never read past this


def fetch_unadjusted_close(tickers, start, end=None, *, batch=100):
    """UNADJUSTED close, needed only for market cap.

    EDGAR share counts are as-reported; multiplying them by a split-adjusted price corrupts every
    value ratio at each split (``fundamentals.market_cap`` refuses to guess). Cached separately so
    the main panel keeps its adjusted series for returns.
    """
    if UNADJ.exists():
        with open(UNADJ, "rb") as fh:
            return pickle.load(fh)
    frames = []
    tickers = list(tickers)
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        log.info("unadjusted fetch %d-%d/%d", i + 1, i + len(chunk), len(tickers))
        try:
            frames.append(cal.fetch_ohlcv_wide(chunk, start, end, auto_adjust=False)["close"])
        except Exception as exc:  # noqa: BLE001
            log.warning("chunk failed: %r", exc)
    df = pd.concat(frames, axis=1).sort_index()
    with open(UNADJ, "wb") as fh:
        pickle.dump(df, fh)
    return df


def benchmark_returns(dates, symbol="SPY"):
    """Daily total-return series for a benchmark ETF, aligned to the panel calendar."""
    start = str(dates[0])[:10]
    end = str(dates[-1] + np.timedelta64(1, "D"))[:10]
    wide = cal.fetch_ohlcv_wide([symbol], start, end, auto_adjust=True)["close"]
    s = wide[symbol].reindex(pd.DatetimeIndex(dates)).ffill()
    return s.pct_change().fillna(0.0).to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2014-12-31")
    ap.add_argument("--k", type=int, default=75)
    ap.add_argument("--rebalance-days", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--recompute", action="store_true", help="rebuild the sleeve cache")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    end = np.datetime64(args.end)
    if end >= OOS_START:
        raise SystemExit(
            f"REFUSED: --end {args.end} reaches into the reserved OOS window "
            f"({str(OOS_START)[:10]}+). The OOS run is single-shot and belongs to P6.")

    with open(PANEL, "rb") as fh:
        panel = pickle.load(fh)
    log.info("panel T=%d N=%d", panel.T, panel.N)

    fp = None
    if FUND.exists():
        with open(FUND, "rb") as fh:
            fp = pickle.load(fh)
        log.info("fundamentals: %d/%d names have facts", fp.meta["n_with_facts"],
                 fp.meta["n_tickers"])

    mcap = None
    if fp is not None:
        unadj = fetch_unadjusted_close(panel.tickers, str(panel.dates[0])[:10])
        unadj = unadj.reindex(index=pd.DatetimeIndex(panel.dates),
                              columns=list(panel.tickers)).to_numpy(dtype=np.float64)
        mcap = fnd.market_cap(fp, unadj)

    sleeve_cache = ROOT / "data" / "raw" / "equity_panel" / "_ballast_sleeves.pkl"
    if sleeve_cache.exists() and not args.recompute:
        with open(sleeve_cache, "rb") as fh:
            sleeves = pickle.load(fh)
    else:
        sleeves = bs.compute_sleeves(panel, fp, mcap)
        with open(sleeve_cache, "wb") as fh:
            pickle.dump(sleeves, fh)
    log.info("sleeves: %s", sorted(sleeves))

    cfg = PortfolioConfig(k=args.k, rebalance_days=args.rebalance_days, cost_bps=args.cost_bps)
    start = np.datetime64(args.start)
    lo = int(np.searchsorted(panel.dates, start))
    hi = int(np.searchsorted(panel.dates, end, side="right"))

    bench = benchmark_returns(panel.dates)[lo:hi]
    bench_perf = performance(bench)

    results = {}
    bench_full = benchmark_returns(panel.dates)

    def run(name: str, names: list[str], w_start=None) -> dict:
        """Backtest a sleeve blend.

        ``w_start`` overrides the evaluation start. This matters: a sleeve whose inputs do not
        exist yet holds NOTHING, and a book that is flat through 2008 posts a *flattering* drawdown
        it never earned. The fundamental sleeves are therefore scored on the window where they can
        actually trade, against a benchmark measured over the SAME window.
        """
        avail = [n for n in names if n in sleeves]
        if not avail:
            return {}
        s0 = w_start if w_start is not None else start
        lo_i = int(np.searchsorted(panel.dates, s0))
        blend = np.nanmean(np.stack([sleeves[n] for n in avail]), axis=0)
        res = backtest(panel, blend, cfg, start=s0, end=end)
        b = bench_full[lo_i:hi]
        perf = performance(res.returns)
        act = active_stats(res.returns, b)
        invested = res.n_holdings > 0
        row = {
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in perf.items()},
            **{k: round(v, 4) for k, v in act.items()},
            "window": f"{str(s0)[:10]}..{args.end}",
            "avg_holdings": round(float(res.n_holdings.mean()), 1),
            # Fraction of days the book actually held anything. Anything < ~0.99 means the metrics
            # above are partly a cash record, and the drawdown in particular is not comparable.
            "exposure_frac": round(float(invested.mean()), 3),
            "avg_holdings_when_invested": round(
                float(res.n_holdings[invested].mean()) if invested.any() else 0.0, 1),
            # Two-sided sum|dw| (a full replacement = 2.0); the conventional one-way figure is half.
            "turnover_2sided_per_year": round(
                float(res.turnover.sum() / (len(res.returns) / 252)), 3),
            "turnover_1way_per_year": round(
                float(0.5 * res.turnover.sum() / (len(res.returns) / 252)), 3),
            "cost_drag_per_year": round(float(res.costs.sum() / (len(res.returns) / 252)), 5),
            "benchmark_sharpe_same_window": round(performance(b)["sharpe"], 4),
            "benchmark_maxdd_same_window": round(performance(b)["max_drawdown"], 4),
            "sleeves": avail,
        }
        results[name] = row
        return row

    # Fundamental sleeves cannot trade before XBRL coverage is broad (measured: 0% pre-2009,
    # 45.7% in 2010, 64.6% in 2011), so they are scored from 2011 on a matched benchmark.
    fund_start = np.datetime64("2011-01-01")

    for s in sorted(sleeves):
        run(f"sleeve_{s}", [s], fund_start if s in bs.FUNDAMENTAL_SLEEVES else None)
    run("core_market_only", list(bs.MARKET_SLEEVES))
    run("core_full_2011", list(bs.SLEEVE_NAMES), fund_start)
    run("core_market_only_2011", list(bs.MARKET_SLEEVES), fund_start)   # like-for-like control

    results["BENCHMARK_SPY"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in bench_perf.items()}
    results["BENCHMARK_SPY_2011"] = {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in performance(
            bench_full[int(np.searchsorted(panel.dates, fund_start)):hi]).items()}

    payload = {
        "window": {"start": args.start, "end": args.end,
                   "note": "TRAIN+VALIDATION only; OOS 2015-2025 is reserved and untouched"},
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
        "panel_survivorship": panel.meta.get("survivorship", {}).get("unpriceable_frac"),
        "verdict_cap": panel.meta.get("verdict_cap"),
        "results": results,
    }
    (OUT / "p2_linear_core.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- report ----------------------------------------------------------------------- #
    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "information_ratio",
            "exposure_frac", "turnover_1way_per_year", "avg_holdings"]
    print(f"\n=== BALLAST P2 — frozen linear core, {args.start} .. {args.end} ===")
    print(f"  (survivorship: {panel.meta['survivorship']['unpriceable_frac']:.1%} of ever-members "
          f"unpriceable; verdict capped at {panel.meta['verdict_cap']})\n")
    hdr = f"{'strategy':22s}" + "".join(f"{c[:9]:>11s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for name, row in results.items():
        cells = "".join(
            f"{row.get(c, float('nan')):>11.3f}" if isinstance(row.get(c), (int, float))
            else f"{'':>11s}" for c in cols)
        print(f"{name:22s}{cells}")

    # G2 thresholds come from the gates file, never from this script.
    gates_path = ROOT / "configs" / "ballast_v1.gates.yaml"
    import yaml
    g2 = yaml.safe_load(gates_path.read_text())["gates"]["g2_beats_benchmark"]
    min_uplift = float(g2["min_sharpe_uplift"])

    print(f"\n  P2 GATE (g2_beats_benchmark, min_sharpe_uplift={min_uplift} from "
          f"{gates_path.name}) — BOTH legs required:")
    for label, key, bkey in (("core_market_only", "core_market_only", "BENCHMARK_SPY"),
                             ("core_full_2011", "core_full_2011", "BENCHMARK_SPY_2011"),
                             ("core_market_only_2011", "core_market_only_2011",
                              "BENCHMARK_SPY_2011")):
        if key not in results:
            continue
        c, b = results[key], results[bkey]
        up = c["sharpe"] - b["sharpe"]
        ok_s = up >= min_uplift
        ok_d = c["max_drawdown"] > b["max_drawdown"]
        print(f"    {label:24s} sharpe {c['sharpe']:+.3f} vs {b['sharpe']:+.3f} "
              f"(uplift {up:+.3f}) {'PASS' if ok_s else 'FAIL'}   "
              f"maxDD {c['max_drawdown']:+.3f} vs {b['max_drawdown']:+.3f} "
              f"{'PASS' if ok_d else 'FAIL'}   => "
              f"{'PASS' if (ok_s and ok_d) else 'FAIL'}")
    print(f"\n  wrote {OUT / 'p2_linear_core.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
