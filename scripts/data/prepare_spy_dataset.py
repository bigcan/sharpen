"""Stage-0 data prep: Dukascopy SPY tick-bars -> GMGP1 multiscale handler format.

The raw pull (`fetch_dukascopy_equity.py`) emits open/high/low/close plus `n_ticks` and
`mean_spread`. The handler requires an OHLC**V** frame (multiscale_handler.py:282), so this
script supplies `volume` and, along the way, MEASURES the transaction cost instead of
assuming one.

VOLUME IS TICK COUNT, AND THAT IS A SUBSTITUTION WORTH STATING. Dukascopy is a CFD feed;
it does not publish consolidated-tape share volume for SPY. `n_ticks` (quote updates per
bar) is the standard activity proxy for such feeds and is what feature 8 (`volume_z`,
SymLog -> EMA-Z -> tanh, multiscale_handler.py:182) actually needs — a normalized
participation signal, not a share count. It is a real measured quantity, not synthetic,
but it is NOT tape volume and any finding that leans on volume levels must say so.

FEES ARE MEASURED, NOT ASSUMED. `mean_spread` is the realized bid-ask on the instrument the
bars come from, so the cost model is derived from the same feed as the prices rather than
from a table. Reported as bps of price so it can go straight into `env.taker_fee` as a
steady-state fee from step 0 (protocol v2 s3.5.4 forbids curriculum schedules outside HPO
sensitivity runs; BUG-01).

Usage:
    python scripts/data/prepare_spy_dataset.py \
        --raw data/equities/SPYUSUSD_1min.parquet --out data/processed/spy_1min.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("prepare-spy")

RTH_BARS_PER_DAY = 390          # 09:30-16:00 ET inclusive of the open minute


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Dukascopy SPY bars -> handler OHLCV parquet")
    ap.add_argument("--raw", default="data/equities/SPYUSUSD_1min.parquet")
    ap.add_argument("--out", default="data/processed/spy_1min.parquet")
    ap.add_argument("--min-ticks", type=int, default=1,
                    help="drop bars built from fewer than this many ticks")
    ap.add_argument("--start", default=None,
                    help="drop bars before this UTC date. The raw file also holds the "
                         "Feb-May 2017 coverage probe; leaving those in would put a 20-MONTH "
                         "hole in the middle of the series, which the manifest reads as a "
                         "gap and the EMA-Z warmup would silently normalize across.")
    args = ap.parse_args()

    raw_p = (ROOT / args.raw) if not Path(args.raw).is_absolute() else Path(args.raw)
    out_p = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    log.info("read %s: %d bars (%s .. %s)", raw_p.name, len(df),
             df["timestamp"].min(), df["timestamp"].max())

    if args.start:
        n_before = len(df)
        df = df[df["timestamp"] >= pd.Timestamp(args.start, tz="UTC")]
        log.info("dropped %d bars before %s", n_before - len(df), args.start)

    n0 = len(df)
    df = df[df["n_ticks"] >= args.min_ticks]
    if len(df) != n0:
        log.info("dropped %d bars below --min-ticks=%d", n0 - len(df), args.min_ticks)

    # --- OHLC integrity. A bar whose high/low does not bracket its open/close is corrupt;
    # CLAUDE.md forbids using mid_price without this check, and PF-XCHECK marks at (H+L)/2.
    bad = ((df["high"] < df[["open", "close"]].max(axis=1) - 1e-9)
           | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9))
    if bad.any():
        log.warning("dropping %d bars with high/low not bracketing open/close", int(bad.sum()))
        df = df[~bad]

    df["volume"] = df["n_ticks"].astype(np.float64)     # tick-count activity proxy (see docstring)

    # --- Measured transaction cost -------------------------------------------------
    spread_bps = (df["mean_spread"] / df["close"] * 10_000.0).replace([np.inf, -np.inf], np.nan)
    q = spread_bps.quantile([0.5, 0.9, 0.99])
    log.info("MEASURED SPY CFD spread (bps of price): median=%.3f p90=%.3f p99=%.3f mean=%.3f",
             q.loc[0.5], q.loc[0.9], q.loc[0.99], spread_bps.mean())
    log.info("  -> suggested one-way env.taker_fee = half-spread = %.6f (%.3f bps)",
             q.loc[0.5] / 2 / 10_000.0, q.loc[0.5] / 2)

    # --- Session shape / coverage ---------------------------------------------------
    df["_date"] = df["timestamp"].dt.date
    per_day = df.groupby("_date").size()
    log.info("sessions: %d days | bars/day median=%d min=%d max=%d",
             len(per_day), int(per_day.median()), int(per_day.min()), int(per_day.max()))
    short = per_day[per_day < RTH_BARS_PER_DAY * 0.5]
    log.info("  %d short sessions (<50%% of %d bars) — half-days/holidays are expected here",
             len(short), RTH_BARS_PER_DAY)
    if (per_day > RTH_BARS_PER_DAY).any():
        log.warning("  %d sessions EXCEED %d bars — extended-hours contamination?",
                    int((per_day > RTH_BARS_PER_DAY).sum()), RTH_BARS_PER_DAY)

    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    if out.isna().any().any():
        log.error("NaNs present after prep — manifest gate requires nan_count==0")
        return 1
    out.to_parquet(out_p, index=False)
    log.info("wrote %s: %d bars (%s .. %s)", out_p, len(out),
             out["timestamp"].min(), out["timestamp"].max())
    log.info("close range %.2f .. %.2f", out["close"].min(), out["close"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
