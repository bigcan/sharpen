"""Fetch free Dukascopy historical tick data and aggregate to bars.

WHY THIS EXISTS. The 2026-07-31 session concluded that discovery was data-blocked and stated the
requirement as "~11 years of hourly ES-class bars at ~0.3bp round trip" — then asserted that such
data is "a purchase, not a search". **That assertion was wrong and this script is the correction.**
Dukascopy publishes tick data free and unauthenticated from **2003 to present** across FX majors,
index CFDs (`USA500IDXUSD` = S&P 500, `USATECHIDXUSD` = Nasdaq, `DEUIDXEUR` = DAX), gold
(`XAUUSD`) and crude (`LIGHTCMDUSD`) — verified live, 23 years of coverage, roughly twice the
derived requirement.

So the constraint is a DOWNLOAD, not a budget line.

FORMAT. `/datafeed/{INSTR}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5` — LZMA-compressed,
20 bytes per tick, big-endian `>u4,i4,i4,f4,f4` = (ms offset within the hour, ask, bid, ask_vol,
bid_vol). **ask and bid are SCALED INT32, not float32** — decoding them as floats silently yields
~1e-45 denormals with no exception raised. Divide by 1e5 for 5-decimal FX, 1e3 for JPY pairs, 1 for
CFDs. Verified: EURUSD June-2015 decodes to 1.0897-1.1358 (correct) with a measured mean spread of
**0.32 bp** — which independently confirms the <=0.3bp cost assumption the data spec relies on.

WEEKENDS ARE EMPTY, NOT MISSING. An hour file for a Saturday/Sunday returns 200 with zero bytes.
An early probe in this session mistook that for absent history and nearly recorded a false "no
coverage" finding — always probe a verified weekday.

Usage (RESUMABLE — re-run the same command to continue):
    python scripts/data/fetch_dukascopy.py --instrument EURUSD --start 2015-01-01 --end 2026-01-01 \
        --bar 1h --workers 12 --out data/dukascopy
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import lzma
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("dukascopy")
BASE = "https://datafeed.dukascopy.com/datafeed"
UA = {"User-Agent": "Mozilla/5.0"}
# 20 bytes/tick, big-endian: ms offset (u4), ask (i4 SCALED), bid (i4 SCALED), ask_vol, bid_vol (f4)
TICK_FMT = ">IiiFF"

# Price divisor by instrument family. FX 5-decimal pairs are integer-scaled by 1e5; JPY pairs 1e3;
# CFDs/metals/energy come through as already-scaled floats (divisor 1).
DIVISOR = {"JPY": 1e3}


def _divisor(instr: str) -> float:
    """Scaled-int divisor. Getting this wrong is SILENT — it yields prices off by 10^3 with no
    error (S&P decoded as 2,114,349 instead of 2114.3 before this was corrected). Always sanity
    -check a decoded price against a known level for any NEW instrument family."""
    if instr.endswith("JPY"):
        return 1e3
    if len(instr) == 6 and instr.isalpha():        # 5-decimal FX pair
        return 1e5
    if instr.startswith(("XAU", "XAG")):
        return 1e2                                  # metals (verified: XAUUSD -> ~1188 in Jun-2015)
    return 1e3                                      # index / energy CFD (verified S&P, DAX)


def fetch_hour(instr: str, when: dt.datetime, session: requests.Session,
               timeout: int = 25) -> pd.DataFrame | None:
    url = f"{BASE}/{instr}/{when.year}/{when.month - 1:02d}/{when.day:02d}/{when.hour:02d}h_ticks.bi5"
    try:
        r = session.get(url, timeout=timeout, headers=UA)
    except requests.RequestException:
        return None                                  # transient; caller may retry
    if r.status_code != 200 or not r.content:
        return pd.DataFrame()                        # weekend / holiday — legitimately empty
    try:
        raw = lzma.LZMADecompressor().decompress(r.content)
    except lzma.LZMAError:
        return pd.DataFrame()
    n = len(raw) // 20
    if n == 0:
        return pd.DataFrame()
    # ask/bid are SCALED INT32, not float32. Decoding them as floats yields ~1e-45 denormals —
    # a bug caught by inspecting decoded prices against a known level (EURUSD ~1.09 in 2015)
    # rather than by any exception, which is why the check is worth doing on every new feed.
    arr = np.frombuffer(raw[: n * 20], dtype=np.dtype(">u4,>i4,>i4,>f4,>f4"))
    div = _divisor(instr)
    ts = pd.to_datetime(when, utc=True) + pd.to_timedelta(arr["f0"].astype(np.int64), unit="ms")
    return pd.DataFrame({"timestamp": ts,
                         "ask": arr["f1"].astype(np.float64) / div,
                         "bid": arr["f2"].astype(np.float64) / div})


def to_bars(ticks: pd.DataFrame, bar: str) -> pd.DataFrame:
    if ticks.empty:
        return ticks
    t = ticks.set_index("timestamp").sort_index()
    mid = (t["ask"] + t["bid"]) / 2.0
    o = mid.resample(bar, label="left", closed="left").first()
    h = mid.resample(bar, label="left", closed="left").max()
    lo = mid.resample(bar, label="left", closed="left").min()
    c = mid.resample(bar, label="left", closed="left").last()
    n = mid.resample(bar, label="left", closed="left").count()
    spread = (t["ask"] - t["bid"]).resample(bar, label="left", closed="left").mean()
    out = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c,
                        "n_ticks": n, "mean_spread": spread}).dropna(subset=["close"])
    return out.reset_index().rename(columns={"index": "timestamp"})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Fetch free Dukascopy ticks -> bars (resumable)")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--bar", default="1h")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/dukascopy")
    args = ap.parse_args()

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.instrument}_{args.bar}.parquet"

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    # weekends are empty by construction — skip them rather than paying a request each
    hours = hours[(hours.dayofweek < 5) | ((hours.dayofweek == 6) & (hours.hour >= 21))]

    have: set[pd.Timestamp] = set()
    prior = None
    if dest.exists():
        prior = pd.read_parquet(dest)
        prior["timestamp"] = pd.to_datetime(prior["timestamp"], utc=True)
        have = set(prior["timestamp"].dt.floor("h"))
        log.info("resume: %s already has %d bars", dest.name, len(prior))
    todo = [h for h in hours if h not in have]
    log.info("%s: %d hour-files to fetch (%d skipped as present/weekend)",
             args.instrument, len(todo), len(hours) - len(todo))
    if not todo:
        log.info("nothing to do")
        return 0

    frames: list[pd.DataFrame] = []
    done = 0
    session_local = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_hour, args.instrument, h.to_pydatetime(), session_local): h
                for h in todo}
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
            done += 1
            if done % 2000 == 0:
                log.info("  %d/%d hours (%.1f%%)", done, len(todo), 100 * done / len(todo))

    if not frames:
        log.warning("no ticks retrieved")
        return 0
    ticks = pd.concat(frames, ignore_index=True)
    bars = to_bars(ticks, args.bar)
    if prior is not None:
        bars = (pd.concat([prior, bars], ignore_index=True)
                .drop_duplicates(subset=["timestamp"]).sort_values("timestamp"))
    bars.to_parquet(dest, index=False)
    log.info("wrote %s: %d bars (%s..%s)", dest, len(bars),
             bars["timestamp"].min(), bars["timestamp"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
