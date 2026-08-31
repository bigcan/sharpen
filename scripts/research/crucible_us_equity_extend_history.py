"""Extend the PIT US-equity union panel further back. (S553-cont-152)

The panel starts 2014-06 for one reason only: that is where `xlg_pit_validation.py` set START_FETCH.
The free fja05680 membership history runs back to **1996**, and yfinance prices surviving names just
as far, so the 12.07-year span is an artifact of a prior probe's scope, not a data limit.

WHY IT MATTERS — it is the difference between a marginal mine and a powered one. The holdout's
CALENDAR SPAN is the only term that sets the detection floor (SE(annualized Sharpe) = 1/sqrt(calendar
years); neither bar frequency nor hold moves it, both measured). At the measured breadth of
BR = 21.5 x 252 = 5,418 independent bets/yr, the IC a candidate must have to be detectable is:

    panel start   span    holdout    MDE     IC needed
    2014-06       12.0y    4.20y     1.782    0.0242     <- marginal: only the UPPER half of the band
    2010-01       16.5y    5.77y     1.463    0.0199
    2007-01       19.5y    6.82y     1.315    0.0179     <- real margin inside 0.02-0.03
    2004-01       22.5y    7.87y     1.197    0.0163

THE TRADE-OFF THIS SCRIPT MEASURES, and it is not free: reaching further back crosses the GFC and
picks up more names that delisted before yfinance coverage begins, so the UNPRICEABLE count — the
residual survivorship bias, 144 names at the 2014 start — should rise. Power and survivorship pull in
opposite directions here. The script reports both so the choice is made on numbers.

Writes `_pit_union_<start-year>.pkl` alongside the existing cache; it never overwrites it.

Usage:
    python scripts/research/crucible_us_equity_extend_history.py --start 2007-01-01
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.data import cross_asset_loader as cal  # noqa: E402
from sharpen.signals.features import Panel  # noqa: E402

MEMBERS = ROOT / "data" / "raw" / "equity_panel" / "sp500_pit_members.csv"
OUTDIR = ROOT / "data" / "raw" / "equity_panel"


def load_union(start: str) -> list[str]:
    df = pd.read_csv(MEMBERS)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start) - pd.Timedelta(days=400)]
    union: set[str] = set()
    for row in df["tickers"]:
        union |= {t.strip().replace(".", "-") for t in str(row).split(",") if t.strip()}
    return sorted(union)


def batched_fetch(tickers: list[str], start: str, batch: int = 120) -> dict:
    frames: dict[str, list] = {f: [] for f in ("open", "high", "low", "close", "volume")}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  fetch {i + 1}-{i + len(chunk)}/{len(tickers)} ...", flush=True)
        try:
            wide = cal.fetch_ohlcv_wide(chunk, start, None)
        except Exception as exc:  # noqa: BLE001
            print(f"   [warn] chunk failed: {exc!r}"[:140])
            continue
        for f in frames:
            frames[f].append(wide[f])
    return {f: pd.concat(frames[f], axis=1).sort_index() for f in frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007-01-01", help="fetch start (warmup before eval)")
    args = ap.parse_args()

    union = load_union(args.start)
    print(f"[members] {len(union)} unique tickers ever-in-index since {args.start}")
    wide = batched_fetch(union, args.start)

    close = wide["close"]
    keep = [c for c in close.columns if close[c].notna().any()]
    unpriceable = len(close.columns) - len(keep)
    wide = {f: wide[f][keep] for f in wide}
    close = wide["close"]

    dates = close.index.to_numpy(dtype="datetime64[ns]")
    cols = list(close.columns)
    T, N = len(dates), len(cols)

    def arr(f: str) -> np.ndarray:
        return np.asarray(wide[f].to_numpy(), dtype=np.float64)

    o, h, lo, c, v = (arr(f) for f in ("open", "high", "low", "close", "volume"))
    # adv_usd is recomputed (shifted) by `us_equity_panel.build_us_equity_panel`; a placeholder here
    # keeps the Panel contract without baking in the unshifted convention the old cache carries.
    adv = np.full((T, N), np.nan)
    meta = {"survivorship_free": True, "source": "yfinance+fja05680_PIT",
            "universe_def": f"PIT S&P500 union since {args.start} (active mask built downstream)",
            "n_universe_union": N, "unpriceable_dropped": unpriceable,
            "fetch_start": args.start}
    panel = Panel(dates, tuple(cols), o, h, lo, c, v,
                  np.ones((T, N), dtype=bool), adv, np.zeros(N, dtype=int), meta)

    year = args.start[:4]
    out = OUTDIR / f"_pit_union_{year}.pkl"
    with open(out, "wb") as fh:
        pickle.dump(panel, fh)

    span = float((dates[-1] - dates[0]) / np.timedelta64(365, "D"))
    print(f"\n{'=' * 62}")
    print(f"  priced names      {N}")
    print(f"  UNPRICEABLE       {unpriceable}  ({100.0 * unpriceable / (N + unpriceable):.1f}% of union)")
    print(f"  bars              {T}  ({span:.2f} calendar years)")
    print(f"  vs 2014 baseline  633 priced / 144 unpriceable (18.5%) / 12.07y")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
