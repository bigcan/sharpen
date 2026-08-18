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
import time
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
    # ORDER IS LOAD-BEARING. Metals must be tested BEFORE the generic 6-letter FX rule: "XAUUSD"
    # is six alpha characters, so it used to fall into the 1e5 FX branch and the metals branch
    # below was unreachable dead code. Gold then decoded to 6.89-10.27 for 2008 (true range
    # ~$688-1027) — a silent 100x error that raised nothing, exactly the failure mode this
    # docstring warns about. The dead branch's own value (1e2) was ALSO wrong; measured against
    # known levels the correct metals divisor is 1e3: raw 688564 -> $688.56 (gold's Oct-2008 low)
    # and raw 1026945 -> $1026.9 (its Mar-2008 high). Verified again at Jun-2015 -> ~$1180.
    if instr.startswith(("XAU", "XAG")):
        return 1e3                                  # metals (verified 2008 lows/highs + Jun-2015)
    if instr.endswith("JPY"):
        return 1e3
    if len(instr) == 6 and instr.isalpha():        # 5-decimal FX pair
        return 1e5
    return 1e3                                      # index / energy CFD (verified S&P, DAX)


def fetch_hour(instr: str, when: dt.datetime, session: requests.Session,
               timeout: int = 25, retries: int = 4) -> pd.DataFrame | None:
    """Returns a tick frame, an EMPTY frame for a legitimately closed hour, or None if every
    retry failed.

    RETRIES ARE LOAD-BEARING, not politeness. The first version of this function returned None on
    the first RequestException with a comment saying the caller "may retry" — and no caller ever
    did. Transient failures were then silently indistinguishable from weekends, and a full
    2004-2026 pull came back with 71,467 of ~140,000 bars: whole years at ~1,400 bars instead of
    ~6,200, with NO error surfaced. Five sampled missing hours all returned data on a plain retry,
    proving the loss was transient rather than absent history.
    """
    url = f"{BASE}/{instr}/{when.year}/{when.month - 1:02d}/{when.day:02d}/{when.hour:02d}h_ticks.bi5"
    r = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout, headers=UA)
        except requests.RequestException:
            r = None
        # STATUS CODES ARE NOT INTERCHANGEABLE. 200 is usable (possibly a legitimately empty
        # closed hour); 404 means this hour genuinely has no file. ANY OTHER status — 429 throttle,
        # 5xx — is TRANSIENT and must be retried, never booked as "market closed".
        #
        # The previous line `if r.status_code != 200 or not r.content: return DataFrame()` made a
        # throttled response indistinguishable from a weekend, silently and without incrementing
        # the failure counter. It surfaced on 2026-08-01 when five fetch blocks were run in
        # parallel (40 concurrent requests): the FIRST year of every block came back ~43% short
        # (3,286-3,614 bars against 5,890-6,288 for every other year) with ZERO failures logged.
        # One thin year per block, always the block's first — a pattern, not noise. Caught by the
        # per-year uniformity check, which is the same check that caught the 2004-2026 FX pull
        # losing half its history to an unimplemented retry.
        if r is not None and r.status_code in (200, 404):
            break
        if attempt == retries - 1:
            return None                          # exhausted — caller COUNTS this, never silently drops
        time.sleep(0.4 * (2 ** attempt) + 0.1 * attempt)
    if r is None or r.status_code == 404 or not r.content:
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

    # RESUME PROVENANCE GUARD. The resume path merges new bars into an existing file, so if the
    # decoding divisor changes between runs the two halves land on DIFFERENT PRICE SCALES and the
    # merge is silent -- no exception, monotone timestamps, plausible bars. That happened on
    # 2026-08-01: a XAUUSD pull begun under the buggy 1e5 divisor was resumed after the fix to
    # 1e3, producing a file whose early years were 100x low. Record the divisor beside the data
    # and refuse to resume across a change.
    sidecar = dest.with_suffix(".manifest.json")
    divisor_now = _divisor(args.instrument)
    if dest.exists() and sidecar.exists():
        import json as _json
        prev_div = _json.loads(sidecar.read_text(encoding="utf-8")).get("divisor")
        if prev_div is not None and float(prev_div) != float(divisor_now):
            log.error("REFUSING TO RESUME: %s was written with divisor %s, this run uses %s. "
                      "Delete %s and re-fetch -- merging would mix price scales silently.",
                      dest.name, prev_div, divisor_now, dest.name)
            return 2

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
    failed = 0
    session_local = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_hour, args.instrument, h.to_pydatetime(), session_local): h
                for h in todo}
        for fut in as_completed(futs):
            df = fut.result()
            if df is None:
                failed += 1                          # exhausted retries — surfaced below, never silent
            elif not df.empty:
                frames.append(df)
            done += 1
            if done % 2000 == 0:
                log.info("  %d/%d hours (%.1f%%)", done, len(todo), 100 * done / len(todo))

    if failed:
        log.warning("%d/%d hours FAILED after retries — re-run to fill (resumable)", failed, len(todo))
    if not frames:
        log.warning("no ticks retrieved")
        return 0
    ticks = pd.concat(frames, ignore_index=True)
    bars = to_bars(ticks, args.bar)
    if prior is not None:
        bars = (pd.concat([prior, bars], ignore_index=True)
                .drop_duplicates(subset=["timestamp"]).sort_values("timestamp"))
    bars.to_parquet(dest, index=False)
    import json as _json
    sidecar.write_text(_json.dumps({"instrument": args.instrument, "bar": args.bar,
                                    "divisor": divisor_now, "n_bars": int(len(bars)),
                                    "px_min": float(bars["close"].min()),
                                    "px_max": float(bars["close"].max())}, indent=1),
                       encoding="utf-8")
    log.info("wrote %s: %d bars (%s..%s) px %.4f-%.4f divisor %g", dest, len(bars),
             bars["timestamp"].min(), bars["timestamp"].max(),
             bars["close"].min(), bars["close"].max(), divisor_now)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
