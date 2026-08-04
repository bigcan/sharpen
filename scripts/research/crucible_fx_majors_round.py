"""FX-majors round: the one configuration that clears BOTH power constraints (pre-registered, cont-151).

WHY THIS CONFIG, DERIVED NOT GUESSED
------------------------------------
Everything mined this session failed for one of two reasons, and the two are in tension:

  * DETECTION needs observations: n_obs = n_eff x rebalances/yr x holdout_years >= (3.17/IC)^2.
  * PROFITABILITY caps rebalances/yr: cost_IR = 2*c*(bars_per_yr/H)/vol must leave the
    just-detectable gross IR (3.17/sqrt(holdout_years), constant across H) above the economic floor.

Solving the second for R_max and substituting gives the achievable n_obs of a configuration. Run over
what is on disk (vol 0.10, floor 0.50, IC 0.025 => 16,078 observations needed):

    config                        cost bp   Y_hold   R_max/yr   H*    n_obs
    11 instr 2014+   (tested)        1.23     3.14        524   11.9   9,413   FAILS
    11 instr 2008-13 (tested)        3.17     1.50        329   18.9   2,159   FAILS
    9 FX majors 2008+ (this file)    0.54     4.65        898    6.9  16,706   CLEARS
    5 FX majors 2004+                0.40     5.65      1,042    6.0  18,840   CLEARS

The tested configs failed because they mixed in XAGUSD (7.54 bp) and LIGHTCMDUSD (4.51 bp) — 8-14x the
majors' spread — which dominates the cost term and collapses R_max. The FX majors are the ONLY subset
on disk with tight spreads AND long history, and that pairing is exactly what the inequality needs.
This was not tried before because every prior panel was built for INSTRUMENT COUNT, on the assumption
that more names meant more breadth; the measurement that adding names to an inner join yields net-zero
breadth is what made the tight-and-long subset the obvious thing to test instead.

The 9-major cell clears by 4%, so it is genuinely marginal — n_eff is ASSUMED 4.00 above and is
MEASURED below. If the measured n_eff comes in under ~3.85 the configuration does not actually clear,
and that is reported as such rather than quietly accepted.

PRE-REGISTERED (written before running)
---------------------------------------
Universe: the 9 FX majors with continuous 2008+ hourly history. Window 2008-01-01 -> 2026-07-31, union
grid + `active` mask, never forward-filled. Holdout = final 25% (embargoed by one horizon).
Signals: the SAME 5 as the 30-cell scan, no new ones. Horizons: {5, 7, 10, 21} bars, bracketing the
derived H* ~ 7. Costs: MEASURED per-instrument per-bar `mean_spread` (half of it, one-way).

Book: cross-sectionally demeaned rank, dollar-neutral, unit gross, rebalanced every H, held between,
marked each bar on the 1-bar forward return, charged sum_i |dw_i| * halfspread_i(t) on rebalance bars.

DECISION RULE — a cell is PROMISING iff, on the HOLDOUT:
    (a) net calendar Sharpe >= 0.50, AND
    (b) t-stat of the net return series >= 3.530, AND
    (c) the same sign holds in the in-sample portion (consistency, not significance).

The t-bar is Bonferroni over EVERY hypothesis tested on this data this session — 30 (IC scan) + 2 (C1
OOS) + 4 (V1 OOS) + 20 (this file's 5x4) = 56 => one-sided 0.01/56 => 3.530. Accumulated multiplicity
is priced rather than reset, because the alternative is to keep testing until something passes.

FALSIFIER: if nothing clears here, the free-data intraday programme is closed — this is the
best-conditioned configuration the data admits, and it was derived from the constraint rather than
found by search.

Usage:
    python scripts/research/crucible_fx_majors_round.py
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

MAJORS = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY",
          "EURGBP", "EURJPY", "NZDUSD", "USDCAD")
START, END = "2008-01-01", "2026-08-01"
HORIZONS = (5, 7, 10, 21)
HOLDOUT_FRAC = 0.25
N_HYPOTHESES = 56                      # accumulated this session; see docstring
T_BAR = float(norm.isf(0.01 / N_HYPOTHESES))
SHARPE_FLOOR = 0.50


def load():
    frames = {}
    for t in MAJORS:
        d = pd.read_parquet(DUKA / f"{t}_1h.parquet")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        frames[t] = d[(d.index >= pd.Timestamp(START, tz="UTC"))
                      & (d.index < pd.Timestamp(END, tz="UTC"))]
    idx = None
    for d in frames.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    g = lambda c: pd.DataFrame({t: frames[t][c].reindex(idx) for t in MAJORS})  # noqa: E731
    close, high, low, ticks = g("close"), g("high"), g("low"), g("n_ticks")
    hspread = pd.DataFrame(
        {t: (frames[t]["mean_spread"] / frames[t]["close"] / 2.0).reindex(idx) for t in MAJORS})
    return close, high, low, ticks, hspread.ffill().bfill(), close.notna()


def scores(close, high, low, ticks, L):
    """The SAME five pre-registered signals as crucible_intraday_ic_scan.py."""
    ret_L = close / close.shift(L) - 1.0
    vol = close.pct_change().rolling(max(L, 20)).std()
    return {
        "R1_reversal": -ret_L,
        "R2_reversal_volnorm": -ret_L / vol.replace(0.0, np.nan),
        "M1_momentum": ret_L,
        "V1_range": -((high - low) / close).rolling(L).mean(),
        "C1_flow_divergence": -close.diff().rolling(L).corr(ticks.diff()),
    }


def weights(score, active):
    s = score.where(active)
    r = s.rank(axis=1)
    r = r.sub(r.mean(axis=1), axis=0)
    return r.div(r.abs().sum(axis=1).replace(0.0, np.nan), axis=0)


def book(W, F, S, H):
    T, N = W.shape
    out = np.full(T, np.nan)
    w = np.zeros(N)
    for t in range(T - 1):
        if t % H == 0:
            tgt = np.nan_to_num(W[t], nan=0.0)
            c = float(np.nansum(np.abs(tgt - w) * np.nan_to_num(S[t], nan=0.0)))
            w = tgt
        else:
            c = 0.0
        out[t] = float(np.nansum(w * F[t])) - c
    return out


def sr_t(x, bpy):
    m = np.isfinite(x)
    if m.sum() < 32 or np.std(x[m]) == 0:
        return np.nan, np.nan
    return (float(np.mean(x[m]) / np.std(x[m]) * np.sqrt(bpy)),
            float(np.mean(x[m]) / (np.std(x[m], ddof=1) / np.sqrt(m.sum()))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/crucible_power_units/fx_majors_round.json")
    args = ap.parse_args()

    close, high, low, ticks, hspread, active = load()
    T = len(close)
    span = (close.index.max() - close.index.min()).days / 365.25
    bpy = T / span
    r = np.log(close).diff()
    ev = np.linalg.eigvalsh(np.nan_to_num(r.corr().to_numpy(), nan=0.0))[::-1]
    n_eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    split = int(T * (1 - HOLDOUT_FRAC))
    Y = (T - split) / bpy
    mean_bp = float(hspread.mean().mean() * 1e4)

    print(f"FX MAJORS round: N={len(MAJORS)} T={T:,} span={span:.2f}y bars/yr={bpy:,.0f}")
    print(f"  measured mean one-way half-spread = {mean_bp:.2f} bp")
    print(f"  n_eff MEASURED = {n_eff:.2f}   (config clears only if >= ~3.85)")
    print(f"  holdout = {T - split:,} bars = {Y:.2f} calendar years")
    gross_floor = 3.17 / np.sqrt(Y)
    print(f"  just-detectable gross IR = {gross_floor:.2f}")
    print(f"  pre-registered bar: net SR >= {SHARPE_FLOOR}, t >= {T_BAR:.3f} "
          f"(Bonferroni over {N_HYPOTHESES})\n")

    F = (close.shift(-1) / close - 1.0).to_numpy(float)
    S = hspread.to_numpy(float)
    print(f"{'signal':<22}{'H':>4}{'IS SR':>9}{'OOS gross':>11}{'OOS net':>10}{'t':>8}  verdict")
    rows, promising = [], []
    for H in HORIZONS:
        sig = scores(close, high, low, ticks, H)
        for name, sc in sig.items():
            W = weights(sc, active).to_numpy(float)
            net_all = book(W, F, S, H)
            gross_all = book(W, F, S * 0.0, H)
            is_sr, _ = sr_t(net_all[:split], bpy)
            oos_net, oos_t = sr_t(net_all[split:], bpy)
            oos_gross, _ = sr_t(gross_all[split:], bpy)
            ok = bool(np.isfinite(oos_net) and oos_net >= SHARPE_FLOOR
                      and np.isfinite(oos_t) and oos_t >= T_BAR
                      and np.isfinite(is_sr) and np.sign(is_sr) == np.sign(oos_net))
            if ok:
                promising.append((name, H))
            rows.append({"signal": name, "H": H, "is_sr": is_sr, "oos_gross_sr": oos_gross,
                         "oos_net_sr": oos_net, "oos_t": oos_t, "promising": ok})
            print(f"{name:<22}{H:>4}{is_sr:>9.3f}{oos_gross:>11.3f}{oos_net:>10.3f}{oos_t:>8.2f}  "
                  f"{'** PROMISING **' if ok else ''}")

    print(f"\n{'=' * 80}")
    if promising:
        print(f"{len(promising)} PROMISING cell(s): {promising}")
        print("Candidate(s) survive the pre-registered bar on the embargoed holdout, net of MEASURED")
        print("per-instrument friction. Next: forward-incubation in the lockbox + Tier-2 audit.")
        print("NOT deployable and NOT capital-ready on this evidence alone.")
    else:
        print("0 PROMISING. This was the best-conditioned configuration the free data admits —")
        print("derived from the power/cost constraint, not found by search. The free-data intraday")
        print("programme is CLOSED; the remaining lever is acquiring a broader instrument universe.")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"universe": list(MAJORS), "T": T, "span_years": span,
                               "bars_per_year": bpy, "n_eff": n_eff, "holdout_years": Y,
                               "mean_halfspread_bp": mean_bp, "t_bar": T_BAR,
                               "sharpe_floor": SHARPE_FLOOR, "cells": rows,
                               "n_promising": len(promising)}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
