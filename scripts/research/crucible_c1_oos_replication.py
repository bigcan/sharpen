"""PRE-REGISTERED out-of-sample replication of the C1 flow-divergence near-miss (cont-151).

WHAT HAPPENED. `crucible_intraday_ic_scan.py` scanned 5 pre-registered signals x 6 horizons on the
2014+ union-grid panel and returned 0 PROMISING under its own rule. One cell missed narrowly:

    C1_flow_divergence, H=21 bars:  holdout IC +0.0508, t = 3.12, net IR +0.91
                                    (Bonferroni bar over 30 cells was t >= 3.403)
    C1_flow_divergence, H=126 bars: holdout IC +0.0893, t = 1.99, net IR +1.31

C1 is -1 * rolling_corr(delta(close), delta(tick_count), L): instruments whose price moves have
DECOUPLED from their trade-count flow score high. A near-miss is a NULL under the pre-registered
rule and it stays a NULL — this script does not re-score the same data with a friendlier threshold,
which is the file-drawer move the whole funnel exists to prevent.

What a near-miss DOES earn is one clean test on data the scan never touched.

THE TEST. Same signal, same construction, same horizons, on the **2008-01-01 -> 2013-12-31** window
— entirely before the scan's 2014 start, so no observation is reused. The instrument set is the FX
majors + metals available across that whole window (the 3 equity index CFDs do not exist before
2011-09 and LIGHTCMDUSD not before 2013, so both are excluded; changing the universe mid-test would
confound a replication failure with a universe change).

PRE-REGISTERED DECISION RULE (written before running, and this file is the record):
  REPLICATES iff, on the 2008-2013 window, for the SAME horizon:
      (a) IC has the SAME SIGN as the 2014+ finding (positive), AND
      (b) t >= 2.576  (one-sided 0.01 Bonferroni-corrected over the 2 horizons tested), AND
      (c) |IC| >= mde_ic for that window (the effect is detectable there at all).
  Anything else = DOES NOT REPLICATE = C1 is dead and the intraday substrate has no surviving
  candidate. There is no third outcome and no "partial" credit.

WHY THIS IS THE RIGHT TEST AND NOT MORE FISHING. Only two cells are examined, both named in advance;
the window is fixed by data availability rather than chosen; the threshold is set before the run; and
a failure is recorded as a kill, not as a reason to try a third window. The pre-2014 period also
contains the GFC aftermath and the 2011-13 QE regime, so it is a genuinely different regime — a real
effect should survive it, and a 2014+ artifact should not.

Usage:
    python scripts/research/crucible_c1_oos_replication.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DUKA = ROOT / "data" / "dukascopy"

# FX majors + metals only: present across the whole 2008-2013 window. LIGHTCMDUSD (from 2013) and the
# 3 index CFDs (from 2011-09, thin until 2014) are EXCLUDED so a replication failure cannot be blamed
# on a universe change.
OOS_TICKERS = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURGBP",
               "EURJPY", "NZDUSD", "USDCAD", "XAGUSD", "XAUUSD")
OOS_START, OOS_END = "2008-01-01", "2014-01-01"
HORIZONS = (21, 126)
T_BAR = float(norm.isf(0.01 / len(HORIZONS)))     # 2.576, fixed before the run
PRIOR = {21: 0.0508, 126: 0.0893}                 # the 2014+ holdout ICs being replicated


def build_union_panel(tickers, start, end):
    frames = {}
    for t in tickers:
        d = pd.read_parquet(DUKA / f"{t}_1h.parquet")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        frames[t] = d[(d.index >= pd.Timestamp(start, tz="UTC"))
                      & (d.index < pd.Timestamp(end, tz="UTC"))]
    idx = None
    for d in frames.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    close = pd.DataFrame({t: frames[t]["close"].reindex(idx) for t in tickers})
    ticks = pd.DataFrame({t: frames[t]["n_ticks"].reindex(idx) for t in tickers})
    return close, ticks, close.notna()


def c1_score(close: pd.DataFrame, ticks: pd.DataFrame, L: int) -> pd.DataFrame:
    """-1 * rolling corr(delta close, delta tick-count) over L bars. Identical to the scan's C1."""
    return -close.diff().rolling(L).corr(ticks.diff())


def ic_series(score, fwd, active, step, min_names=6):
    s = score.where(active)
    s = s.sub(s.mean(axis=1), axis=0)
    out = []
    for t in range(0, len(s) - step, step):
        a, b = s.iloc[t], fwd.iloc[t]
        m = a.notna() & b.notna()
        if int(m.sum()) < min_names:
            continue
        out.append(a[m].rank().corr(b[m].rank()))
    return np.asarray([v for v in out if np.isfinite(v)], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/crucible_power_units/c1_oos_replication.json")
    args = ap.parse_args()

    close, ticks, active = build_union_panel(OOS_TICKERS, OOS_START, OOS_END)
    T = len(close)
    span = (close.index.max() - close.index.min()).days / 365.25
    r = np.log(close).diff()
    ev = np.linalg.eigvalsh(np.nan_to_num(r.corr().to_numpy(), nan=0.0))[::-1]
    n_eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    print(f"OOS window {OOS_START} -> {OOS_END}: N={close.shape[1]} T={T:,} "
          f"span={span:.2f}y n_eff={n_eff:.2f}")
    print(f"pre-registered bar: same sign as 2014+, t >= {T_BAR:.3f}, |IC| >= mde_ic\n")
    print(f"{'H':>5}{'n_obs':>8}{'IC_2014+':>10}{'IC_oos':>10}{'mde_ic':>9}{'t':>8}  verdict")

    rows, n_rep = [], 0
    for H in HORIZONS:
        fwd = close.shift(-H) / close - 1.0
        ic = ic_series(c1_score(close, ticks, H), fwd, active, H)
        if ic.size < 8:
            print(f"{H:>5}{ic.size:>8}  too few observations")
            continue
        m, sd = float(ic.mean()), float(ic.std(ddof=1))
        t = m / (sd / np.sqrt(ic.size)) if sd > 0 else float("nan")
        mde = 3.17 / np.sqrt(max(1.0, n_eff * (T / H)))
        ok = bool(np.sign(m) == np.sign(PRIOR[H]) and np.isfinite(t)
                  and t >= T_BAR and abs(m) >= mde)
        n_rep += int(ok)
        rows.append({"H": H, "n_obs": int(ic.size), "ic_2014plus": PRIOR[H], "ic_oos": m,
                     "mde_ic": float(mde), "t": float(t), "replicates": ok})
        print(f"{H:>5}{ic.size:>8}{PRIOR[H]:>10.4f}{m:>10.4f}{mde:>9.4f}{t:>8.2f}  "
              f"{'REPLICATES' if ok else 'does NOT replicate'}")

    print(f"\n{'=' * 70}")
    if n_rep:
        print(f"C1 REPLICATES at {n_rep}/{len(HORIZONS)} horizon(s) on untouched 2008-2013 data.")
        print("This is a surviving candidate -> forward-incubate in the lockbox; it is NOT a")
        print("deployable strategy until it clears incubation and a Tier-2 audit.")
    else:
        print("C1 DOES NOT REPLICATE. The 2014+ near-miss was noise; the cell is dead.")
        print("No surviving candidate on the intraday substrate. The binding constraint remains")
        print("instrument breadth (n_eff ~5.7), which no amount of further search on this")
        print("instrument set can fix.")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"window": [OOS_START, OOS_END], "tickers": list(OOS_TICKERS),
                               "T": T, "span_years": span, "n_eff": n_eff, "t_bar": T_BAR,
                               "cells": rows, "n_replicates": n_rep}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
