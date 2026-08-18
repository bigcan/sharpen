"""Is there ANY detectable cross-sectional signal on the intraday panel? (pre-registered, cont-151)

WHY THIS, AND WHY NOT ANOTHER MINING TICK
-----------------------------------------
The first intraday tick returned 0 PROMISING, and the power work explains why in a way that makes
re-running it pointless: the funnel's detection floor on that substrate is a calendar dSR of ~1.4-1.9,
while a realistic cross-sectional signal delivers an IR near 0.9 there. The funnel cannot resolve
anything in between, so another tick at the same settings is a coin toss with the coin glued down.

Two facts reframe what to measure instead:

  * Detection and achievable effect are the SAME quantity seen twice. Detectable IC ~ 3.17/sqrt(n_obs)
    with n_obs = n_eff x rebalances/yr x holdout_years; achievable IR = IC x sqrt(n_eff x
    rebalances/yr). Both improve as sqrt(1/hold). There is no free lunch between them — but hold
    horizon moves both together, and it is the one knob not fixed by the data we can get free.
  * Adding instruments to a COMMON-TIMESTAMP inner join is self-defeating, measured: 12 -> 15
    instruments raised n_eff 5.01 -> 5.63 and cut rebalances/yr 271 -> 241, for a net breadth change
    of 1,359 -> 1,358 (zero). This script therefore builds the panel on the UNION grid with an
    `active` mask (no forward-fill: a name simply does not trade on bars it has no bar for), so N can
    grow without T shrinking.

So the question worth money is not "does the funnel promote something" but "does a real
cross-sectional IC exist here at ANY horizon, and does it survive cost". That is measured directly.

PRE-REGISTERED (written before running)
---------------------------------------
Signals (fixed list, no search, so there is no selection to deflate beyond the count below):
  R1  short-horizon reversal: -1 * (close/close[-L] - 1)          (the canonical intraday x-sec effect)
  R2  reversal on vol-normalised returns: R1 / realized_vol
  M1  momentum: +1 * (close/close[-L] - 1)                        (R1's sign flip; both reported)
  V1  -1 * (high-low)/close range rank                            (illiquidity/vol premium proxy)
  C1  -1 * corr(rank(close-change), rank(tick-count), L)          (WQ101-style flow/price divergence)

Horizons L (bars) = holds H (bars) = {3, 5, 10, 21, 63, 126}.

DECISION RULE, per (signal, horizon) cell:
  DETECTABLE   iff |IC_mean| >= mde_ic, where mde_ic = 3.17 / sqrt(n_eff x n_rebalances_holdout).
  PROMISING    iff DETECTABLE **and** net IC-IR t-stat >= 2.33 on the HOLDOUT quarter **and** the
               implied net IR (after the measured cost drag at 2 bps one-way) >= 0.50.
  Everything else is a NULL. A cell that is significant IN-SAMPLE but not on the holdout is a NULL.

Bonferroni over the 5 x 6 = 30 pre-registered cells is applied to the t-threshold (2.33 -> the
0.01/30 one-sided normal quantile) so the multiplicity is priced, not ignored.

FALSIFIER: if no cell is DETECTABLE at any horizon, the intraday substrate has no cross-sectional
edge reachable with this instrument set, and the answer is instrument breadth (or a different
market), not more search.

Usage:
    python scripts/research/crucible_intraday_ic_scan.py
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

# 12 FX/metals/energy + the 3 equity index CFDs. The indices were excluded from CLEAN_12 as "thin";
# measured 2026-08-04, that flag is a START-OF-HISTORY artifact (2012-13 ramp: 2.0-4.7k bars/yr) and
# NOT a session artifact — from 2014 they run 5,766-5,918 bars/yr against FX's 5,888-6,250. Start is
# therefore 2014-01-01 for the wide set.
FX = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURGBP",
      "EURJPY", "NZDUSD", "USDCAD", "XAGUSD", "XAUUSD", "LIGHTCMDUSD")
IDX = ("DEUIDXEUR", "USA500IDXUSD", "USATECHIDXUSD")
START = "2014-01-01"

HORIZONS = (3, 5, 10, 21, 63, 126)
COST_BPS = 0.0002          # one-way, matching configs/intraday_signal_eval.gates.yaml
HOLDOUT_FRAC = 0.25
N_CELLS = 30               # 5 signals x 6 horizons — the pre-registered multiplicity
BOOK_VOL = 0.10            # assumed annual vol of a unit-gross dollar-neutral book (see net_ir)


def build_union_panel(tickers: tuple[str, ...], start: str):
    """Wide close/high/low/ticks frames on the UNION of timestamps, plus an `active` mask.

    No forward-fill anywhere. A name with no bar at t is INACTIVE at t (excluded from that bar's
    cross-section), which is the honest encoding of "it did not trade" — unlike a ffill, which
    invents a price and manufactures spurious relative moves at every session boundary.
    """
    frames = {}
    for t in tickers:
        f = DUKA / f"{t}_1h.parquet"
        d = pd.read_parquet(f)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        frames[t] = d[d.index >= pd.Timestamp(start, tz="UTC")]
    idx = None
    for d in frames.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    close = pd.DataFrame({t: frames[t]["close"].reindex(idx) for t in tickers})
    high = pd.DataFrame({t: frames[t]["high"].reindex(idx) for t in tickers})
    low = pd.DataFrame({t: frames[t]["low"].reindex(idx) for t in tickers})
    ticks = pd.DataFrame({t: frames[t]["n_ticks"].reindex(idx) for t in tickers})
    active = close.notna()
    return close, high, low, ticks, active


def _xs_demean(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally demean each row (a dollar-neutral score) ignoring inactive names."""
    return x.sub(x.mean(axis=1), axis=0)


def signals(close, high, low, ticks, L: int) -> dict[str, pd.DataFrame]:
    """The five pre-registered scores at look-back L. Every one is a function of bars <= t only."""
    ret_L = close / close.shift(L) - 1.0
    r1 = close.pct_change()
    vol = r1.rolling(max(L, 20)).std()
    rng = ((high - low) / close).rolling(L).mean()
    dclose = close.diff()
    c1 = dclose.rolling(L).corr(ticks.diff())
    return {
        "R1_reversal": -ret_L,
        "R2_reversal_volnorm": -ret_L / vol.replace(0.0, np.nan),
        "M1_momentum": ret_L,
        "V1_range": -rng,
        "C1_flow_divergence": -c1,
    }


def ic_series(score: pd.DataFrame, fwd: pd.DataFrame, active: pd.DataFrame,
              step: int, min_names: int = 6) -> np.ndarray:
    """Per-rebalance cross-sectional Spearman IC between score[t] and the forward return t -> t+H.

    Sampled every `step` bars so the observations do NOT overlap — overlapping ICs would inflate the
    apparent sample size, which is exactly the error that makes an underpowered cell look significant.
    """
    s = _xs_demean(score.where(active))
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
    ap.add_argument("--out", default="results/crucible_power_units/intraday_ic_scan.json")
    args = ap.parse_args()

    close, high, low, ticks, active = build_union_panel(FX + IDX, START)
    T = len(close)
    span = (close.index.max() - close.index.min()).days / 365.25
    bpy = T / span
    r = np.log(close).diff()
    corr = r.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    ev = np.linalg.eigvalsh(corr)[::-1]
    n_eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    split = int(T * (1 - HOLDOUT_FRAC))
    print(f"UNION-grid panel: N={close.shape[1]}  T={T:,}  span={span:.2f}y  bars/yr={bpy:,.0f}")
    print(f"  n_eff (participation ratio) = {n_eff:.2f}")
    print(f"  holdout = {T - split:,} bars = {(T - split) / bpy:.2f} calendar years")
    print(f"  active-cell density = {active.to_numpy().mean():.1%}\n")

    t_bar = float(norm.isf(0.01 / N_CELLS))     # Bonferroni over the 30 pre-registered cells
    print(f"Bonferroni t-threshold over {N_CELLS} cells: {t_bar:.3f} (one-sided 0.01/{N_CELLS})\n")
    print(f"{'signal':<22}{'H':>5}{'n_obs':>8}{'IC':>9}{'mde_IC':>9}{'t':>8}"
          f"{'net IR':>9}  verdict")

    rows = []
    for H in HORIZONS:
        fwd = close.shift(-H) / close - 1.0
        sig = signals(close, high, low, ticks, H)
        n_rebal_hold = (T - split) // H
        mde_ic = 3.17 / np.sqrt(max(1.0, n_eff * n_rebal_hold))
        # cost per rebalance in IR terms: a gross-1 book turning over ~1.0 each rebalance pays
        # COST_BPS*turnover; annualised drag ~ COST_BPS * 2 * (bpy/H).
        cost_ann = COST_BPS * 2.0 * (bpy / H)
        for name, sc in sig.items():
            ic_h = ic_series(sc.iloc[split:], fwd.iloc[split:], active.iloc[split:], H)
            if ic_h.size < 8:
                continue
            m, sd = float(ic_h.mean()), float(ic_h.std(ddof=1))
            t = m / (sd / np.sqrt(ic_h.size)) if sd > 0 else np.nan
            gross_ir = m * np.sqrt(n_eff * (bpy / H))
            # Net IR = gross IR minus the cost drag converted into Sharpe units. A drag of `cost_ann`
            # (fraction of notional per year) on a book of annual vol `BOOK_VOL` costs cost_ann/BOOK_VOL
            # of Sharpe. BOOK_VOL is an explicit ASSUMPTION (10% — a unit-gross dollar-neutral book on
            # these instruments), not a measurement; it is stated here rather than buried because the
            # net verdict is linear in it.
            net_ir = gross_ir - cost_ann / BOOK_VOL
            detectable = abs(m) >= mde_ic
            promising = bool(detectable and np.isfinite(t) and t >= t_bar and net_ir >= 0.50)
            verdict = "PROMISING" if promising else ("detectable" if detectable else "null")
            rows.append({"signal": name, "H": H, "n_obs": int(ic_h.size), "ic_mean": m,
                         "mde_ic": float(mde_ic), "t": float(t), "gross_ir": float(gross_ir),
                         "net_ir": float(net_ir), "cost_ann": float(cost_ann),
                         "detectable": bool(detectable), "promising": promising})
            print(f"{name:<22}{H:>5}{ic_h.size:>8}{m:>9.4f}{mde_ic:>9.4f}{t:>8.2f}"
                  f"{net_ir:>9.2f}  {verdict}")

    n_p = sum(r_["promising"] for r_ in rows)
    n_d = sum(r_["detectable"] for r_ in rows)
    print(f"\n{'=' * 78}\n{len(rows)} cells: {n_d} detectable, {n_p} PROMISING")
    if n_p == 0:
        print("VERDICT: no cross-sectional edge on this instrument set at any pre-registered horizon.")
        print("The binding constraint is instrument BREADTH (n_eff), not search effort.")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"panel": {"N": int(close.shape[1]), "T": T, "span_years": span,
                                         "bars_per_year": bpy, "n_eff": n_eff,
                                         "holdout_bars": T - split},
                               "t_bonferroni": t_bar, "cells": rows,
                               "n_detectable": n_d, "n_promising": n_p}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
