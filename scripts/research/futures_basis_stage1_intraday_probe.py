"""Stage-1a: intraday 13:30-alignment confirmation for the TAIEX futures-basis signal (S553-cont-128).

Implements docs/research/futures_basis_arb_preregistration_2026-07-12.md §8 (Stage-1a), gates frozen
before this ran. CPU-only, ~$0.

Stage-0 found a VALID gross signal but the same-day (lag0) number was inflated by the cash-13:30 /
futures-13:45 mark mismatch (naive +4.10 -> +1.37 on a 1-day execution delay). This probe rebuilds
the signal's basis with the future priced AT 13:30 (synchronized with the cash close) and asks: does
a real, significant reversion edge survive proper alignment? Only the SIGNAL's basis timing changes;
the traded return stays the Stage-0 roll-safe front-month spread return.

Usage:
    python scripts/research/futures_basis_stage1_intraday_probe.py
"""
from __future__ import annotations

import logging
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# reuse Stage-0 machinery verbatim so signal logic is identical (only the basis timing changes)
from scripts.research.futures_basis_arb_probe import (  # noqa: E402
    W_SET, _boot_p05, _build, _cell, _perm_pvalue, _sharpe,
)

TX_15MIN = ROOT / "data" / "taiwan_intraday" / "TX_15min.parquet"
CASH_NEW = ROOT / "data" / "taiwan_options" / "TAIEX_spot.parquet"

log = logging.getLogger("futures_basis_stage1_intraday_probe")


def _intraday_marks() -> pd.DataFrame:
    """Per-date futures at 13:30 (close of the 13:15->13:30 bar) and 13:45 (close of 13:30->13:45)."""
    tx = pd.read_parquet(TX_15MIN)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["t"] = tx["timestamp"].dt.time
    tx["d"] = pd.to_datetime(tx["timestamp"].dt.date)
    f1330 = tx[tx["t"] == time(13, 15)][["d", "close"]].rename(columns={"close": "F1330", "d": "date"})
    f1345 = tx[tx["t"] == time(13, 30)][["d", "close"]].rename(columns={"close": "F1345", "d": "date"})
    return f1330.merge(f1345, on="date")


def _with_basis(m0: pd.DataFrame, F: np.ndarray) -> pd.DataFrame:
    """Copy the Stage-0 frame but overwrite ann_basis with (ln F - ln S)/tau for the given F mark.
    Keeps spread_ret / rF / rS / tau identical (roll-safe traded return unchanged)."""
    m = m0.copy()
    m["ann_basis"] = (np.log(F) - np.log(m["S"].values)) / m["tau"].values
    return m


def _grid(m: pd.DataFrame) -> dict:
    prim = {W: _cell(m, W, 1.0, lag=1) for W in W_SET}
    naive = {W: _cell(m, W, 1.0, lag=0) for W in W_SET}
    med_l1 = float(np.median([prim[W]["gross"] for W in W_SET]))
    med_l0 = float(np.median([naive[W]["gross"] for W in W_SET]))
    med_W = min(W_SET, key=lambda W: abs(naive[W]["gross"] - med_l0))
    return {"med_lag0": med_l0, "med_lag1": med_l1, "med_W": med_W,
            "cell_lag0": naive[med_W], "per_W": {W: (naive[W]["gross"], prim[W]["gross"]) for W in W_SET}}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("Stage-1a intraday 13:30-alignment (pre-reg §8)")

    m0 = _build("close")                                  # full Stage-0 roll-safe frame (2001-2025)
    marks = _intraday_marks()
    m = m0.merge(marks, on="date", how="inner").sort_values("date").reset_index(drop=True)
    m = m[(m.F1330 > 0) & (m.F1345 > 0) & (m.S > 0)].reset_index(drop=True)
    log.info("aligned overlap n=%d  %s -> %s", len(m), m.date.min().date(), m.date.max().date())

    variants = {
        "EOD ": m,                                        # Stage-0 basis (TX_daily EOD), subsampled
        "1345": _with_basis(m, m["F1345"].values),        # intraday EOD basis
        "1330": _with_basis(m, m["F1330"].values),        # ALIGNED basis (artifact-removed)
    }
    res = {}
    for name, mv in variants.items():
        g = _grid(mv)
        res[name] = g
        pw = "  ".join(f"W{W}:l0={a:+.2f}/l1={b:+.2f}" for W, (a, b) in g["per_W"].items())
        log.info("  %s  med lag0=%+.3f  lag+1=%+.3f  | %s", name, g["med_lag0"], g["med_lag1"], pw)

    # aligned-cell significance
    base = res["1330"]["cell_lag0"]
    p05 = _boot_p05(base["gross_pnl"])
    perm_p = _perm_pvalue(base["pos"], base["fwd"], base["gross"])
    aligned_l0 = res["1330"]["med_lag0"]

    # LEG DECOMPOSITION (harvestability): split the aligned PnL pos*(rF - rS) into the FUTURES leg
    # (pos*rF, tradeable) vs the CASH leg (pos*(-rS), needs an untradeable index / proxy). If the
    # edge lives in the cash leg, it is index-staleness (not harvestable by trading the future);
    # if it lives in the futures leg, it is a real, tradeable convergence move.
    mv = variants["1330"]
    med_W = res["1330"]["med_W"]
    for lag in (0, 1):
        cell = _cell(mv, med_W, 1.0, lag=lag)
        pos = cell["pos"]
        rF = np.roll(mv["rF"].values, -lag)
        rS = np.roll(mv["rS"].values, -lag)
        if lag > 0:
            rF[-lag:] = 0.0
            rS[-lag:] = 0.0
        fut_leg = _sharpe(pos * rF)          # trade ONLY the future on the basis signal
        cash_leg = _sharpe(pos * (-rS))      # short-cash leg (needs proxy)
        log.info("  leg-decomp 1330 lag%+d (W=%d): total=%+.2f  FUTURES-leg(tradeable)=%+.2f  "
                 "CASH-leg(proxy-needed)=%+.2f", lag, med_W, cell["gross"], fut_leg, cash_leg)
    eod_l0 = res["EOD "]["med_lag0"]
    eod_l1 = res["EOD "]["med_lag1"]

    art1 = (eod_l0 - aligned_l0) > 1.0                                   # artifact real & large
    align1 = (aligned_l0 >= 0.50) and (p05 > 0) and (perm_p < 0.05)      # real aligned edge
    consist1 = abs(aligned_l0 - eod_l1) <= 0.5                           # matches Stage-0 lag+1
    log.info("=" * 78)
    log.info("  ART-1  EOD_l0(%.2f) - aligned_l0(%.2f) = %.2f > 1.0 ? %s",
             eod_l0, aligned_l0, eod_l0 - aligned_l0, art1)
    log.info("  ALIGN-1 aligned_l0=%.3f>=0.50 & p05=%.3f>0 & perm_p=%.4f<0.05 ? %s",
             aligned_l0, p05, perm_p, align1)
    log.info("  CONSIST-1 |aligned_l0(%.2f) - EOD_lag+1(%.2f)| = %.2f <= 0.5 ? %s",
             aligned_l0, eod_l1, abs(aligned_l0 - eod_l1), consist1)
    ok = art1 and align1 and consist1
    log.info("VERDICT Stage-1a: %s", "CONFIRMED — timing caveat resolved; proceed to 1b (0050 proxy)"
             if ok else "NOT confirmed — inspect (edge may be timing/staleness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
