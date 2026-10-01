"""Re-derive FinRL-X Adaptive Rotation headline metrics from its exported weights.

FinRL-X (AI4Finance ``FinRL-Trading`` @ 4409abe9) prints its README table from
``_generate_performance_report`` in ``src/strategies/run_adaptive_rotation_strategy.py``:
the exported weekly target weights are compounded close-to-close between consecutive
rebalance rows, cash earns zero, no transaction cost is charged, and
Sharpe = CAGR / (std of weekly returns * sqrt(52)). Daily stop-loss / fast risk-off
exits never reach that curve (they are logged, not appended to the weights).

This script, reading only FinRL-X's outputs (weights CSV + the ``{SYM}_daily.csv`` price
files the backtest consumed):
  1. recomputes FinRL-X's equity curve and metrics independently and, with
     ``--their-log``, asserts parity with the table FinRL-X printed (to half a printed
     unit; a missing or incomplete table fails);
  2. re-prices the same weights at ``--cost-bps`` per side (the paper states 10 bps),
     charging turnover against drifted, not target, prior weights, plus a ``--cost-grid``
     and the break-even cost against QQQ;
  3. adds two numbers FinRL-X does not report: the arithmetic Sharpe and the max
     drawdown on daily marks (FinRL-X marks drawdown weekly).

Turnover is two-sided (sum of |dw| over buys and sells), so cost = turnover * bps per side.

Usage:
    python scripts/research/finrl_x_rotation_repro.py \
        --weights <FinRL-Trading>/src/strategies/output/weights/adaptive_rotation/ars_portfolio_weights_2018-01-07_to_2025-10-24.csv \
        --data-dir <FinRL-Trading>/data/fmp_daily --their-log run.log --out results/finrl_x/repro/unadj.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("finrl_x_rotation_repro")

META_COLS = ("date", "cash", "regime")
BENCHMARKS = ("SPY", "QQQ")
# FinRL-X prints multiples/ratios to 0.01 and percentages to 0.01% -- parity allows half a printed unit
RATIO_LABELS = ("Cumulative Return", "Sharpe Ratio", "Calmar Ratio")
STRATEGY_ONLY_LABELS = ("Calmar Ratio", "Win Rate")
BREAKEVEN_MAX_BPS = 100.0


def load_close(data_dir: Path, sym: str) -> pd.Series | None:
    path = data_dir / f"{sym}_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def price_at(series: pd.Series, when: pd.Timestamp) -> float:
    """Last close on or before ``when`` -- FinRL-X's ``series.loc[:d].dropna().iloc[-1]``."""
    return float(series.loc[:when].dropna().iloc[-1])


def metrics(equity: pd.Series) -> dict:
    """FinRL-X's metric definitions, verbatim, plus the arithmetic Sharpe (rf = 0 in both)."""
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    wk = equity.pct_change().dropna()
    ann_vol = wk.std() * np.sqrt(52)
    max_dd = float((equity / equity.cummax() - 1).min())
    return {
        "cumulative_x": float(equity.iloc[-1] / equity.iloc[0]),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe_finrlx": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,  # CAGR / vol
        "sharpe_arith": float(wk.mean() * 52 / ann_vol) if ann_vol > 0 else 0.0,
        "max_dd_weekly": max_dd,
        "calmar": float(ann_ret / abs(max_dd)) if max_dd != 0 else 0.0,
        "win_rate_weekly": float((wk > 0).mean()),
        "years": float(years),
        "n_weeks": int(len(wk)),
    }


def _periods(weights: pd.DataFrame, closes: dict[str, pd.Series]):
    """One pass over the rebalance rows: turnover vs drifted weights, period return, in-period daily path.

    Weights are fractions of NAV, so neither turnover nor period returns depend on the cost level: the equity at
    any cost is prod((1 - turnover_i * c) * (1 + ret_i)), see ``_equity``. The last row is an endpoint (FinRL-X
    never applies its weights).
    """
    assets = [c for c in weights.columns if c not in META_COLS and c in closes]
    dates = pd.DatetimeIndex(pd.to_datetime(weights["date"]))
    w_target = weights[assets].fillna(0.0).to_numpy(dtype=float)
    turnover, ret, paths = [], [], []
    held = np.zeros(len(assets))  # drifted weights (fraction of NAV) before this rebalance
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        w = w_target[i]
        turnover.append(float(np.abs(w - held).sum()))
        rel = np.array([
            price_at(closes[a], d1) / price_at(closes[a], d0) if w[j] > 0 else 1.0
            for j, a in enumerate(assets)
        ])
        period_ret = float((w * (rel - 1)).sum())
        ret.append(period_ret)
        # buy-and-hold of the period's weights, marked on each day in (d0, d1]; an all-cash period is flat
        held_now = [a for j, a in enumerate(assets) if w[j] > 0]
        if held_now:
            path = pd.DataFrame({a: closes[a].loc[d0:d1] / price_at(closes[a], d0) for a in held_now}).ffill()
            path = path.loc[path.index > d0]
            wj = np.array([w[assets.index(a)] for a in path.columns])
            paths.append(pd.Series(1 + ((path.to_numpy() - 1) * wj).sum(axis=1), index=path.index))
        else:
            paths.append(pd.Series(dtype=float))
        cash_after = 1 - w.sum()
        held = w * rel / (cash_after + (w * rel).sum()) if period_ret > -1 else np.zeros_like(w)
    return dates, np.array(turnover), np.array(ret), paths


def _equity(dates: pd.DatetimeIndex, turnover: np.ndarray, ret: np.ndarray, cost_bps: float) -> pd.Series:
    """Weekly-marked NAV: the cost is paid at each rebalance, before that period's return."""
    growth = (1 - turnover * cost_bps / 1e4) * (1 + ret)
    return pd.Series(np.concatenate([[1.0], np.cumprod(growth)]), index=dates)


def backtest(weights: pd.DataFrame, closes: dict[str, pd.Series], cost_bps: float) -> dict:
    """Compound target weights between rebalance rows; charge cost_bps per side on (two-sided) turnover."""
    dates, turnover, ret, paths = _periods(weights, closes)
    equity = _equity(dates, turnover, ret, cost_bps)
    out = metrics(equity)
    start_nav = equity.to_numpy()[:-1] * (1 - turnover * cost_bps / 1e4)  # NAV just after each rebalance
    daily_nav = pd.concat([pd.Series([1.0], index=dates[:1])]
                          + [n0 * p for n0, p in zip(start_nav, paths) if len(p)])
    daily_nav = daily_nav[~daily_nav.index.duplicated(keep="last")]
    out["max_dd_daily"] = float((daily_nav / daily_nav.cummax() - 1).min())
    out["avg_turnover_per_rebalance"] = float(turnover.mean())
    out["annual_turnover"] = float(turnover.sum() / out["years"])
    return out


def breakeven_bps(dates, turnover, ret, target: float, key: str, hi: float = BREAKEVEN_MAX_BPS) -> float:
    """Per-side cost (bps) at which ``metrics[key]`` falls to ``target``.

    0.0 if the strategy does not beat the target even at zero cost; inf if it still beats it at ``hi``.
    """
    def gap(bps: float) -> float:
        return metrics(_equity(dates, turnover, ret, bps))[key] - target

    if gap(0.0) <= 0:
        return 0.0
    if gap(hi) > 0:
        return math.inf
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def benchmark(close: pd.Series, dates: pd.DatetimeIndex) -> dict:
    """FinRL-X's benchmark curve: close reindexed onto rebalance dates, ffilled."""
    start_price = close.loc[: dates[0]].iloc[-1]
    return metrics(close.reindex(dates, method="ffill") / start_price)


def parse_their_table(log_text: str) -> dict[str, list[str]]:
    """Rows of FinRL-X's printed 'Performance Analysis' table: label -> [strategy, SPY, QQQ]."""
    table = log_text.split("Performance Analysis")[-1]
    rows = {}
    for label in ("Cumulative Return", "Annualized Return", "Annualized Volatility",
                  "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Win Rate"):
        m = re.search(rf"^{label}\s+(.+)$", table, flags=re.M)
        if m:
            rows[label] = m.group(1).split()
    return rows


def to_num(tok: str) -> float:
    tok = tok.strip()
    if tok.endswith("%"):
        return float(tok[:-1]) / 100
    return float(tok.rstrip("x"))


def check_parity(ours: dict, bench: dict, their: dict) -> list[str]:
    """Problems between our metrics and FinRL-X's printed table; a missing or short row is a problem (fail closed)."""
    keys = {"Cumulative Return": "cumulative_x", "Annualized Return": "ann_return",
            "Annualized Volatility": "ann_vol", "Sharpe Ratio": "sharpe_finrlx",
            "Max Drawdown": "max_dd_weekly", "Calmar Ratio": "calmar",
            "Win Rate": "win_rate_weekly"}
    cols = [("Strategy", ours)] + [(b, bench[b]) for b in BENCHMARKS if b in bench]
    problems = []
    for label, key in keys.items():
        expected = cols[:1] if label in STRATEGY_ONLY_LABELS else cols
        got = their.get(label, [])
        if len(got) < len(expected):
            problems.append(f"{label}: FinRL-X table has {len(got)} of {len(expected)} values")
            continue
        half_unit = 0.005 if label in RATIO_LABELS else 5e-5
        for k, (name, m) in enumerate(expected):
            theirs, mine = to_num(got[k]), m[key]
            if abs(theirs - mine) > half_unit + 1e-9:
                problems.append(f"{label} [{name}]: theirs {theirs:.4f} vs ours {mine:.4f}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cost-bps", type=float, default=10.0, help="per side (paper: 10)")
    ap.add_argument("--cost-grid", default="0,2,3,5,10",
                    help="per-side bps levels to report (our US-equity funnel prices at 2; paper: 10)")
    ap.add_argument("--their-log", type=Path, help="FinRL-X stdout, for the parity check")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    weights = pd.read_csv(args.weights, parse_dates=["date"]).sort_values("date")
    assets = [c for c in weights.columns if c not in META_COLS]
    closes = {s: load_close(args.data_dir, s) for s in set(assets) | set(BENCHMARKS)}
    missing = [s for s in assets if closes.get(s) is None]
    if missing:
        logger.warning("no price file for %s -- FinRL-X silently drops these too", missing)
    closes = {s: v for s, v in closes.items() if v is not None}

    dates, turnover, ret, _ = _periods(weights, closes)
    bench = {b: benchmark(closes[b], dates) for b in BENCHMARKS if b in closes}
    grid = [float(b) for b in args.cost_grid.split(",")]
    result = {
        "weights_file": str(args.weights),
        "data_dir": str(args.data_dir),
        "window": [str(dates[0].date()), str(dates[-1].date())],
        "n_rebalances": int(len(dates)),
        "turnover_convention": "two-sided sum |dw|; cost = turnover * bps per side",
        "strategy_0bps": backtest(weights, closes, 0.0),
        f"strategy_{args.cost_bps:g}bps": backtest(weights, closes, args.cost_bps),
        "cost_grid": {f"{b:g}": metrics(_equity(dates, turnover, ret, b)) for b in grid},
        "benchmarks": bench,
    }
    if "QQQ" in bench:
        be = {key: breakeven_bps(dates, turnover, ret, bench["QQQ"][key], key) for key in ("ann_return", "sharpe_arith")}
        result["breakeven_vs_QQQ_bps"] = {k: (v if math.isfinite(v) else f">{BREAKEVEN_MAX_BPS:g}") for k, v in be.items()}
    if args.their_log:
        their = parse_their_table(args.their_log.read_text(encoding="utf-8", errors="replace"))
        result["their_table"] = their
        result["parity_problems"] = check_parity(result["strategy_0bps"], bench, their)
        status = "PASS" if not result["parity_problems"] else "FAIL"
        logger.info("parity vs FinRL-X printed table: %s %s", status, result["parity_problems"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for key in ("strategy_0bps", f"strategy_{args.cost_bps:g}bps"):
        m = result[key]
        logger.info("%-16s cum %.2fx  CAGR %.2f%%  vol %.2f%%  SR(finrlx) %.2f  SR(arith) %.2f  "
                    "MDD wk %.2f%% / daily %.2f%%  turnover/yr %.1fx (two-sided)",
                    key, m["cumulative_x"], 100 * m["ann_return"], 100 * m["ann_vol"],
                    m["sharpe_finrlx"], m["sharpe_arith"], 100 * m["max_dd_weekly"],
                    100 * m["max_dd_daily"], m["annual_turnover"])
    for b, m in bench.items():
        logger.info("%-16s cum %.2fx  CAGR %.2f%%  vol %.2f%%  SR(finrlx) %.2f  SR(arith) %.2f  MDD %.2f%%",
                    b, m["cumulative_x"], 100 * m["ann_return"], 100 * m["ann_vol"],
                    m["sharpe_finrlx"], m["sharpe_arith"], 100 * m["max_dd_weekly"])
    logger.info("cost grid (bps/side: CAGR, SR arith): %s", "  ".join(
        f"{b}: {100 * m['ann_return']:.2f}%, {m['sharpe_arith']:.2f}" for b, m in result["cost_grid"].items()))
    if "breakeven_vs_QQQ_bps" in result:
        logger.info("break-even vs QQQ (bps/side): %s", result["breakeven_vs_QQQ_bps"])
    logger.info("wrote %s", args.out)
    return 0 if not result.get("parity_problems") else 1


if __name__ == "__main__":
    raise SystemExit(main())
