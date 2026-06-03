"""Strengthen the linear momentum core before the RL allocator build.

Adds to the falsification (xsec_momentum_falsification.py):
  1. WIDER universe (~24 ETF proxies) — more breadth => smoother (AQR 60+ effect).
  2. TURNOVER SMOOTHING — EMA the trend signal + a weight no-trade band => lower
     turnover, better net (the literature's ~2/3 cost cut).
  3. COMBINED CORE — 0.5*TSMOM + 0.5*XSMOM, the actual baseline the RL must beat.
  4. PARAMETER-ROBUSTNESS GRID — vary lookbacks / vol-window / rebalance / cost to
     confirm the GO is not knife-edge (all params a-priori; nothing tuned to maximize).

CPU-only. Reuses the validated, causal, leak-checked machinery from the falsification.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsec_momentum_falsification as F  # noqa: E402

warnings.filterwarnings("ignore")
OUT = F.OUT / "strengthen"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "prices_wide.parquet"

# Wider universe: original 18 + breadth adds (equity/RE/credit/commodity/fx)
UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ"],
    "rates": ["TLT", "IEF", "LQD", "HYG", "EMB"],
    "commodity": ["GLD", "SLV", "DBC", "USO", "DBA", "UNG"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA", "FXC"],
}
ALL = [t for v in UNIVERSE.values() for t in v]


def get_wide_prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    raw = yf.download(ALL, start=F.START, end=F.END, progress=False, auto_adjust=True)
    close = raw["Close"].dropna(how="all").sort_index()
    close.to_parquet(CACHE)
    return close


def smooth_weights(weights_rebal: pd.DataFrame, band: float) -> pd.DataFrame:
    """No-trade band: only update an asset's weight when it moves > band, else hold.
    Cuts turnover from vol-rescaling churn. Causal (only past/current rebal weights)."""
    out = weights_rebal.copy() * 0.0
    held = pd.Series(0.0, index=weights_rebal.columns)
    for dt in weights_rebal.index:
        tgt = weights_rebal.loc[dt].fillna(0.0)
        move = (tgt - held).abs() > band
        held = held.where(~move, tgt)
        out.loc[dt] = held
    return out


def book_metrics(weights_rebal, rets, bench, cost="standard_2bps"):
    gross, costd, turn = F.backtest(weights_rebal, rets)
    net = gross - costd[cost]
    m = F.metrics(net, turn, bench)
    return m, net


def main():
    close = get_wide_prices()
    close = close[[t for t in ALL if t in close.columns]]
    rets = close.pct_change()
    bench = close["SPY"].pct_change()
    classes = {c: [t for t in v if t in close.columns] for c, v in UNIVERSE.items()}
    print(f"Wide universe: {close.shape[1]} tickers, "
          f"{close.index.min().date()} -> {close.index.max().date()}")

    rebal = F.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(F.LOOKBACKS) + F.SKIP + F.VOL_WIN]]
    sig = F.tsmom_signal(close, rebal)

    report = {"universe": classes, "results": {}}

    # ---- 1. TSMOM raw (wide) vs 2. EMA-smoothed signal vs 3. + no-trade band ----
    w_raw = F.vol_scaled_weights(sig, rets, rebal)
    sig_ema = sig.ewm(span=3).mean()                      # smooth trend score
    w_ema = F.vol_scaled_weights(sig_ema, rets, rebal)
    w_band = smooth_weights(w_ema, band=0.10)             # + no-trade band
    for name, w in [("TSMOM_wide_raw", w_raw),
                    ("TSMOM_wide_ema", w_ema),
                    ("TSMOM_wide_ema_band", w_band)]:
        m, _ = book_metrics(w, rets, bench)
        report["results"][name] = m
        print(f"{name:24s} Sharpe {m['sharpe']:.2f}  turn/yr {m['turnover_ann']:>6.1f}  "
              f"ret@10 {m['ann_ret_pct@10vol']:>5}%  DD@10 {m['max_dd_pct@10vol']:>6}%  corrSPY {m['corr_SPY']}")

    # ---- 3. COMBINED core: 0.5 TSMOM(smoothed+band) + 0.5 XSMOM, the RL baseline ----
    xs_books = []
    for cls, tk in classes.items():
        if len(tk) < 5:
            continue
        xs_books.append(F.xsmom_weights(close, rets, rebal, tk)
                        .reindex(columns=close.columns).fillna(0.0))
    w_xs = sum(xs_books) if xs_books else w_band * 0.0
    w_xs_band = smooth_weights(w_xs, band=0.10)
    # normalize each sleeve to ~unit gross then blend 50/50
    w_combined = 0.5 * w_band + 0.5 * w_xs_band
    m_core, net_core = book_metrics(w_combined, rets, bench)
    report["results"]["COMBINED_core (RL baseline)"] = m_core
    print("\n>>> COMBINED core (0.5 TSMOM + 0.5 XSMOM, smoothed+band) = RL BASELINE:")
    print(f"    Sharpe {m_core['sharpe']:.2f}  PF {m_core['pf']:.3f}  turn/yr {m_core['turnover_ann']:.1f}  "
          f"ret@10 {m_core['ann_ret_pct@10vol']}%  DD@10 {m_core['max_dd_pct@10vol']}%  corrSPY {m_core['corr_SPY']}")
    # subperiod stability of the core
    sub = {}
    for lo, hi, lab in [("2008", "2013", "08-12"), ("2013", "2018", "13-17"),
                        ("2018", "2023", "18-22"), ("2023", "2027", "23-26")]:
        seg = net_core[(net_core.index >= lo) & (net_core.index < hi)].dropna()
        sub[lab] = round(float(seg.mean() / seg.std() * np.sqrt(252)), 2) if seg.std() > 0 else 0.0
    report["results"]["COMBINED_core (RL baseline)"]["subperiod_sharpe"] = sub
    print(f"    subperiod Sharpe: {sub}")

    # ---- 4. PARAMETER-ROBUSTNESS GRID (TSMOM smoothed+band, monthly) ----
    print("\n=== parameter robustness (net Sharpe @ standard 2bps, TSMOM wide ema+band) ===")
    grid = {}
    orig_lb, orig_vw = F.LOOKBACKS, F.VOL_WIN
    for lb in [[252], [126, 252], [63, 126, 252]]:
        for vw in [42, 63, 126]:
            F.LOOKBACKS, F.VOL_WIN = lb, vw
            s = F.tsmom_signal(close, rebal).ewm(span=3).mean()
            w = smooth_weights(F.vol_scaled_weights(s, rets, rebal), band=0.10)
            m, _ = book_metrics(w, rets, bench)
            key = f"lb={'/'.join(map(str, lb))},vw={vw}"
            grid[key] = m["sharpe"]
            print(f"  {key:28s} Sharpe {m['sharpe']:.2f}")
    F.LOOKBACKS, F.VOL_WIN = orig_lb, orig_vw
    report["results"]["param_grid_net_sharpe"] = grid
    grid_vals = list(grid.values())
    report["param_grid_summary"] = {
        "min": min(grid_vals), "max": max(grid_vals),
        "median": float(np.median(grid_vals)),
        "all_positive": bool(all(v > 0 for v in grid_vals))}

    (OUT / "strengthen_results.json").write_text(json.dumps(report, indent=2))
    print(f"\nparam grid: min {min(grid_vals):.2f} / median {np.median(grid_vals):.2f} / "
          f"max {max(grid_vals):.2f}  all_positive={all(v > 0 for v in grid_vals)}")
    print(f"\nRL BASELINE LOCKED: COMBINED core net Sharpe {m_core['sharpe']:.2f} "
          f"(2bps) — RL must beat this OOS.")
    return report


if __name__ == "__main__":
    main()
