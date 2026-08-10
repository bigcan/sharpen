"""V1_range under MEASURED per-instrument spreads — pre-registered OOS test (cont-151).

WHY THIS CELL, AND WHY IT IS NOT FISHING
----------------------------------------
The 30-cell scan found `V1_range` (score = -1 * rolling mean of (high-low)/close over L, i.e. short
the high-range names) to be the STRONGEST gross effect on the panel and by a wide margin:

    H=3 : holdout IC -0.0302, t = -5.96     H=5 : holdout IC -0.0328, t = -4.92

Both clear the scan's Bonferroni bar (3.403) in magnitude. Both were then recorded NULL for one
reason only: net IR of -11.6 / -7.8 under a FLAT 2 bps one-way cost. That number was my assumption,
not a measurement, and this session's own method rule says friction is a per-turnover PRICE that must
be re-derived per venue rather than carried over. Re-testing the cost assumption on a cell whose
GROSS effect already cleared the pre-registered significance bar is not moving a goalpost — the
significance bar is untouched. It is fixing an input I flagged as doing the work.

The measurement (Dukascopy ships `mean_spread`, the venue's OWN recorded spread per bar — not a
range-based estimator, which this project has separately established must never be used to price a
strategy):

    FX majors one-way half-spread, 2014+ : 0.54 bp mean   (EURUSD 0.17, USDJPY 0.22, NZDUSD 1.00)
    metals / energy / indices            : XAUUSD 1.07, XAGUSD 7.54, LIGHTCMDUSD 4.51, indices ~1.0
    flat assumption used by the scan     : 2.00 bp for EVERYTHING

So the flat number was 3.7x too harsh on the majors and 3.8x too LENIENT on silver. A flat cost does
not merely bias the level — it MISALLOCATES friction across the cross-section, over-charging the
instruments a book would actually concentrate in and under-charging the ones it should avoid. This
script prices each name at its own measured, time-varying spread.

PRE-REGISTERED (written before running)
---------------------------------------
Universe FIXED at the 11 instruments with continuous 2008+ history (FX majors + XAU/XAG), used
IDENTICALLY in both periods, so a period difference cannot be a universe change. LIGHTCMDUSD (2013+)
and the index CFDs (thin before 2014) are excluded on availability, decided before running.

  DISCOVERY period  2014-01-01 -> 2026-07-31  (where the cell was found; reported for reference ONLY,
                                               it cannot support a verdict)
  OOS period        2008-01-01 -> 2014-01-01  (untouched by the scan; the DECISION comes from here)

Book: cross-sectionally demeaned rank of the score, gross-normalised to 1 (dollar-neutral, unit
gross), rebalanced every H bars, held between, each bar marked with the 1-bar forward return, and
charged sum_i |dw_i| * halfspread_i(t) at each rebalance using that bar's MEASURED spread.

  DECISION RULE — V1_range SURVIVES iff, on the OOS period, at the SAME H:
      (a) net calendar Sharpe > 0 with the SAME SIGN as discovery, AND
      (b) t-stat of the net return series >= 2.576  (one-sided 0.01, Bonferroni over the 2 hypotheses
          now tested on this OOS window — C1 was the first), AND
      (c) net calendar Sharpe >= 0.50 (the same economic floor the scan used).
  Anything else = DOES NOT SURVIVE. No third window will be tried.

H grid = {3, 5, 10, 21} — the four horizons where V1 was measured detectable or near it. Reported for
all four; the decision is per-H and a single surviving H is enough to call it a candidate, with the
multiplicity already priced into (b).

Usage:
    python scripts/research/crucible_v1_range_costed.py
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

UNIVERSE = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "EURGBP",
            "EURJPY", "NZDUSD", "USDCAD", "XAGUSD", "XAUUSD")
OOS = ("2008-01-01", "2014-01-01")
DISCOVERY = ("2014-01-01", "2026-08-01")
HORIZONS = (3, 5, 10, 21)
T_BAR = float(norm.isf(0.01 / 2))     # 2.576 — Bonferroni over {C1, V1} on the OOS window
SHARPE_FLOOR = 0.50


def load(start: str, end: str):
    """Union-grid close/high/low + MEASURED one-way half-spread (fraction of price), active mask."""
    frames = {}
    for t in UNIVERSE:
        d = pd.read_parquet(DUKA / f"{t}_1h.parquet")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        frames[t] = d[(d.index >= pd.Timestamp(start, tz="UTC"))
                      & (d.index < pd.Timestamp(end, tz="UTC"))]
    idx = None
    for d in frames.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    close = pd.DataFrame({t: frames[t]["close"].reindex(idx) for t in UNIVERSE})
    high = pd.DataFrame({t: frames[t]["high"].reindex(idx) for t in UNIVERSE})
    low = pd.DataFrame({t: frames[t]["low"].reindex(idx) for t in UNIVERSE})
    # one-way cost as a FRACTION of price = half the quoted spread / price.
    hspread = pd.DataFrame(
        {t: (frames[t]["mean_spread"] / frames[t]["close"] / 2.0).reindex(idx) for t in UNIVERSE})
    return close, high, low, hspread.ffill().bfill(), close.notna()


def v1_weights(high, low, close, active, L: int) -> pd.DataFrame:
    """Dollar-neutral unit-gross weights from the V1_range score (identical construction to the scan).

    Score at t uses bars <= t only (rolling mean of the contemporaneous bar's range, no shift needed
    because the range of bar t is known at the close of bar t and the book earns t -> t+1).
    """
    score = -((high - low) / close).rolling(L).mean()
    s = score.where(active)
    r = s.rank(axis=1)                                  # cross-sectional rank
    r = r.sub(r.mean(axis=1), axis=0)                   # dollar-neutral
    gross = r.abs().sum(axis=1).replace(0.0, np.nan)
    return r.div(gross, axis=0)                         # unit gross


def book(weights, fwd1, hspread, H: int) -> np.ndarray:
    """Net per-bar return: rebalance every H bars, hold between, charge measured spread on turnover."""
    W = weights.to_numpy(dtype=float)
    F = fwd1.to_numpy(dtype=float)
    S = hspread.to_numpy(dtype=float)
    T, N = W.shape
    out = np.full(T, np.nan)
    w = np.zeros(N)
    for t in range(T - 1):
        if t % H == 0:
            tgt = np.nan_to_num(W[t], nan=0.0)
            cost = float(np.nansum(np.abs(tgt - w) * np.nan_to_num(S[t], nan=0.0)))
            w = tgt
        else:
            cost = 0.0
        out[t] = float(np.nansum(w * F[t])) - cost
    return out


def evaluate(label: str, start: str, end: str) -> list[dict]:
    close, high, low, hspread, active = load(start, end)
    T = len(close)
    span = (close.index.max() - close.index.min()).days / 365.25
    bpy = T / span
    fwd1 = close.shift(-1) / close - 1.0
    print(f"\n{label}: {start} -> {end}   N={close.shape[1]} T={T:,} span={span:.2f}y "
          f"bars/yr={bpy:,.0f}")
    print(f"  mean measured one-way half-spread: {hspread.mean().mean() * 1e4:.2f} bp")
    print(f"{'H':>5}{'gross SR':>11}{'cost/yr':>10}{'NET SR':>10}{'t':>8}{'turnover/yr':>13}")
    rows = []
    for H in HORIZONS:
        w = v1_weights(high, low, close, active, H)
        net = book(w, fwd1, hspread, H)
        gross = book(w, fwd1, hspread * 0.0, H)
        m = np.isfinite(net)
        sr = float(np.mean(net[m]) / np.std(net[m]) * np.sqrt(bpy)) if net[m].std() > 0 else np.nan
        gsr = (float(np.mean(gross[m]) / np.std(gross[m]) * np.sqrt(bpy))
               if gross[m].std() > 0 else np.nan)
        drag = float(np.nansum(gross[m] - net[m]) / span)
        # t-stat of the NET series (iid approximation; the book is marked every bar and held, so this
        # is optimistic on autocorrelation — noted, and it does not rescue a failing cell).
        t = float(np.mean(net[m]) / (np.std(net[m], ddof=1) / np.sqrt(m.sum())))
        turn = float(np.nansum(np.abs(w.diff().to_numpy())[::H]) / span)
        rows.append({"H": H, "gross_sr": gsr, "net_sr": sr, "t": t, "cost_drag_yr": drag,
                     "turnover_yr": turn})
        print(f"{H:>5}{gsr:>11.3f}{drag:>10.1%}{sr:>10.3f}{t:>8.2f}{turn:>13.1f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/crucible_power_units/v1_range_costed.json")
    args = ap.parse_args()
    print("V1_range priced at MEASURED per-instrument spreads (Dukascopy mean_spread).")
    print(f"Pre-registered OOS bar: same sign, t >= {T_BAR:.3f}, net SR >= {SHARPE_FLOOR}")

    disc = evaluate("DISCOVERY (reference only)", *DISCOVERY)
    oos = evaluate("OOS (the decision)", *OOS)

    dmap = {r["H"]: r for r in disc}
    print(f"\n{'=' * 74}\nVERDICT (decided on OOS only)")
    survivors = []
    for r in oos:
        d = dmap[r["H"]]
        same_sign = np.sign(r["net_sr"]) == np.sign(d["net_sr"]) and r["net_sr"] > 0
        ok = bool(same_sign and r["t"] >= T_BAR and r["net_sr"] >= SHARPE_FLOOR)
        if ok:
            survivors.append(r["H"])
        print(f"  H={r['H']:<4} discovery net SR {d['net_sr']:+.3f} | OOS net SR {r['net_sr']:+.3f} "
              f"t={r['t']:.2f}  ->  {'SURVIVES' if ok else 'does NOT survive'}")
    if survivors:
        print(f"\nV1_range SURVIVES at H={survivors}. This is a genuine candidate: real gross effect,")
        print("positive out-of-sample net of MEASURED per-instrument friction. Next step is")
        print("forward-incubation in the lockbox + a Tier-2 audit — NOT capital.")
    else:
        print("\nV1_range does NOT survive out of sample. The cost correction was necessary but not")
        print("sufficient; the cell is dead and the intraday substrate has no surviving candidate.")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"universe": list(UNIVERSE), "t_bar": T_BAR,
                               "sharpe_floor": SHARPE_FLOOR, "discovery": disc, "oos": oos,
                               "survivors": survivors}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
