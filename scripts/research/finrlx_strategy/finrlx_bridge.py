"""FinRL-X as the platform: our book in FinRL-X's weight contract, run through FinRL-X's own backtester.

FinRL-X's ``BacktestEngine`` is long-only and fully invested (every weight row is renormalised to sum to 1;
it rebalances to target every day; it has no execution lag and no cash rate). So the parity check runs the
LONG-ONLY variant of the book, with an explicit CASH column (a price series compounding the T-bill rate
minus the cash spread), weights forward-filled daily and lagged one bar by us, and the same per-side cost on
every column. Our engine in the same mode must reproduce FinRL-X's NAV; after that, our engine's richer
account model (shorts, borrow, drift between rebalances, no free daily rebalancing) is an extension of a
parity-checked base, not an unverified rewrite.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import pnl
from .paths import FINRLX_CLONE, FINRLX_VENV_PY, RESULTS

log = logging.getLogger("finrlx.bridge")
RUNNER = Path(__file__).with_name("finrlx_bt_runner.py")
CASH = "CASH"


def contract_weights(targets: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Decision-date targets -> FinRL-X's wide daily weight frame (ffill), with a CASH column so rows sum to 1.

    Row t holds the target DECIDED at t. Long-only: any negative weight is an error here.
    """
    if (targets < -1e-12).any(axis=None):
        raise ValueError("FinRL-X's engine is long-only: negative weights cannot be expressed")
    w = targets.reindex(calendar).ffill().fillna(0.0)
    w[CASH] = (1.0 - w.sum(axis=1)).clip(lower=0.0)
    return w


def cash_price(rf: pd.Series, spread_bps_yr: float) -> pd.Series:
    return (1.0 + rf.fillna(0.0) - spread_bps_yr / 1e4 / 252).cumprod().rename(CASH)


def run_finrlx(prices: pd.DataFrame, weights_exec: pd.DataFrame, tc: float, name: str, io_dir: Path) -> pd.Series:
    """Run FinRL-X's BacktestEngine in its own venv. ``weights_exec`` row t = weights that trade at close t."""
    io_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(io_dir / "prices.csv")
    weights_exec.to_csv(io_dir / "weights.csv")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)   # never leak our packages into FinRL-X's interpreter
    r = subprocess.run([str(FINRLX_VENV_PY), str(RUNNER), str(io_dir), repr(tc), name], cwd=str(FINRLX_CLONE),
                       capture_output=True, text=True, env=env, timeout=1800)
    (io_dir / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"FinRL-X runner failed ({r.returncode}); see {io_dir / 'stdout.txt'}")
    nav = pd.read_csv(io_dir / "nav.csv", index_col=0, parse_dates=True)["nav"]
    return nav / nav.iloc[0]


def replica(prices: pd.DataFrame, weights_decided: pd.DataFrame, tc_bps: float) -> pd.Series:
    """Our engine in FinRL-X mode: daily rebalance to the ffilled target, lag 1, same cost on every column."""
    rets = prices.pct_change(fill_method=None)
    cm = pnl.CostModel(default_cost_bps=tc_bps, default_borrow_bps_yr=0.0, cash_spread_bps_yr=0.0)
    res = pnl.run(weights_decided, rets, pd.Series(0.0, index=prices.index), cm)
    return res["nav"] / res["nav"].iloc[0]


def parity(prices: pd.DataFrame, targets_long_only: pd.DataFrame, rf: pd.Series, *, tc_bps: float,
           spread_bps_yr: float, start: pd.Timestamp, name: str = "finrlx_parity") -> dict:
    """Run both engines on identical inputs and compare NAVs from ``start``."""
    cal = prices.index[prices.index >= start]
    px = prices.loc[cal].copy()
    px[CASH] = cash_price(rf.reindex(cal), spread_bps_yr)
    w_dec = contract_weights(targets_long_only.reindex(columns=[c for c in px.columns if c != CASH]).fillna(0.0), cal)
    w_exec = w_dec.shift(1).fillna(0.0)          # decided at t-1, trades at close t (our lag, applied by us)
    w_exec.iloc[0, w_exec.columns.get_loc(CASH)] = 1.0
    io = RESULTS / "finrlx_bridge" / name
    ours = replica(px, w_dec, tc_bps)
    theirs = run_finrlx(px, w_exec, tc_bps / 1e4, name, io)
    j = ours.index.intersection(theirs.index)
    # FinRL-X's bt skips the first row (RunAfterDate is strict): align on the common span, both rebased
    o, t = ours.loc[j] / ours.loc[j].iloc[0], theirs.loc[j] / theirs.loc[j].iloc[0]
    rel = (o / t - 1.0).abs()
    dr = (o.pct_change() - t.pct_change()).dropna()
    out = {"name": name, "tc_bps": tc_bps, "n_days": int(len(j)), "start": str(j[0].date()), "end": str(j[-1].date()),
           "final_ours": float(o.iloc[-1]), "final_finrlx": float(t.iloc[-1]),
           "max_abs_rel_nav_diff": float(rel.max()), "mean_abs_daily_ret_diff_bps": float(dr.abs().mean() * 1e4),
           "ann_return_gap_bps": float((np.log(o.iloc[-1]) - np.log(t.iloc[-1])) / (len(j) / 252) * 1e4)}
    (io / "parity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("parity %s", out)
    return out
