"""Fetch free Dukascopy tick data for CASH-EQUITY / ETF CFDs and aggregate to bars.

WHY A SEPARATE SCRIPT FROM `fetch_dukascopy.py`. The FX/index sibling is correct for
round-the-clock instruments but has two properties that break on equities:

  1. IT REQUESTS ALL 24 HOURS OF EVERY WEEKDAY. Equity CFDs trade RTH only — a measured
     probe of `SPYUSUSD` on Wed 2024-06-12 returned ticks in 13:00-19:59 UTC and exactly
     zero in the other 17 hours. Pulling 24h/day would spend ~70% of its requests on hours
     that are empty by construction, against a feed that already throttles with 503.
  2. IT ACCUMULATES EVERY TICK IN MEMORY AND BARS ONCE AT THE END. SPY runs ~10k ticks/hour;
     9.5 years x 6.5h/day x ~250d/yr is ~1.6e8 ticks, several GB before the first bar is
     written. This script bars each DAY as it lands and discards the ticks, so peak memory
     is one day (~65k ticks) regardless of span.

RESUMABILITY IS PER-DAY AND FAILURE-AWARE. A day is recorded complete only when every one
of its requested hours returned a definite answer (data or a genuine empty). If any hour
exhausted its retries, the day is left absent so a re-run re-fetches it. This is the
difference between "the market was closed" and "we failed to ask", which the feed does not
distinguish for you: a throttled 503 and a holiday both look like no bytes.

COVERAGE / SANITY, measured 2026-08-15 against the live feed:
  * `SPYUSUSD` history begins in 2017 (2015 and 2016 return nothing on a verified weekday).
  * Divisor is 1e3: 2024-06-12 15:00 UTC decodes to bid 543.096 and 2026-06-17 to 749.026,
    both matching SPY's actual level. `--expect-price-range` enforces this rather than
    trusting it, because a wrong divisor is SILENT (see the sibling's `_divisor` docstring).
  * 2024-06-19 returns zero ticks — Juneteenth. Holidays are absent, not zero-filled.

Usage (RESUMABLE — re-run the same command to continue / repair):
    python scripts/data/fetch_dukascopy_equity.py --instrument SPYUSUSD \
        --start 2017-01-01 --end 2026-08-15 --bar 1min --hours 13-20 \
        --workers 8 --out data/equities --expect-price-range 50 2000
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.fetch_dukascopy import fetch_hour, to_bars  # noqa: E402

log = logging.getLogger("dukascopy-equity")


def _parse_hours(spec: str) -> list[int]:
    """'13-20' -> [13..20]; '13,14,15' -> [13,14,15]."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


class Pacer:
    """Adaptive request pacing against Dukascopy's IP rate limiter.

    MEASURED 2026-08-15, and the reason this class exists. Concurrency is punished
    immediately: 6 workers -> 0/48 requests returned data, 3 workers -> 1/48, 1 worker ->
    48/48. But sequential is not sufficient either — after ~25 minutes of steady pulling at
    ~1.1 req/s the feed began returning 503 to EVERY request including single sequential
    ones, i.e. there is a sustained-rate budget on top of the concurrency limit.

    So the throttle has to be discovered at runtime rather than hardcoded: additive-decrease
    on success, multiplicative-increase on failure (the inverse of TCP's AIMD, since here
    the controlled variable is delay rather than rate). A fixed sleep would either waste
    hours when the feed is healthy or keep tripping the limiter when it is not.
    """

    def __init__(self, min_delay: float = 0.35, max_delay: float = 30.0):
        self.min_delay, self.max_delay = min_delay, max_delay
        self.delay = min_delay
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            d = self.delay
        time.sleep(d)

    def on_failure(self) -> None:
        with self._lock:
            self.delay = min(self.delay * 2.0, self.max_delay)

    def on_success(self) -> None:
        with self._lock:
            self.delay = max(self.delay * 0.90, self.min_delay)


_local = threading.local()


def _session() -> requests.Session:
    """One requests.Session PER THREAD.

    A single shared Session handed to a ThreadPoolExecutor is not thread-safe: its urllib3
    connection pool can wedge, and this pull did exactly that — 100 days in 3.5 min, then
    SIX MINUTES of zero progress with the process alive and burning no CPU, while a freshly
    opened session fetched the same hours in 0.4s. It never errors, so no retry logic can
    see it; it just stops. Per-thread sessions keep connection reuse without the sharing.
    """
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


def fetch_day(instr: str, day: dt.date, hours: list[int], ex: ThreadPoolExecutor,
              pacer: "Pacer") -> tuple[pd.DataFrame | None, int]:
    """Fetch one day's requested hours and bar them immediately.

    The executor is owned by the CALLER and lives for the whole pull, so its worker threads
    (and therefore their per-thread Sessions, and therefore their kept-alive connections)
    survive across days. A per-day executor would rebuild both every iteration.

    Returns (tick frame or None, n_failed_hours). A day with ANY failed hour returns its
    partial ticks alongside a non-zero failure count; the caller must NOT mark it complete,
    because a partial day silently becomes a fabricated gap in the bar series.
    """
    def _one(h: int):
        pacer.wait()
        return fetch_hour(instr, dt.datetime(day.year, day.month, day.day, h), _session())

    frames: list[pd.DataFrame] = []
    failed = 0
    for df in ex.map(_one, hours):
        if df is None:
            failed += 1          # exhausted retries — NOT a holiday, do not let it read as one
        elif not df.empty:
            frames.append(df)
    # Steer the pacer on the day's outcome. A failure means the limiter is biting; a clean
    # day means we can creep back toward the floor.
    pacer.on_failure() if failed else pacer.on_success()
    if not frames:
        return None, failed
    return pd.concat(frames, ignore_index=True), failed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Dukascopy equity/ETF CFD ticks -> bars (resumable)")
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--bar", default="1min")
    ap.add_argument("--hours", default="13-20", help="UTC hours to request, e.g. '13-20'")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent hour requests. MEASURED: >1 is counterproductive — 3 workers "
                         "got 1/48 requests through, 6 got 0/48, 1 got 48/48.")
    ap.add_argument("--min-delay", type=float, default=0.35,
                    help="pacing floor in seconds; the Pacer adapts upward from here")
    ap.add_argument("--out", default="data/equities")
    ap.add_argument("--expect-price-range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="fail if decoded closes fall outside [LO,HI] — catches a wrong divisor, "
                         "which is otherwise SILENT")
    args = ap.parse_args()

    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.instrument}_{args.bar}.parquet"
    state_path = out_dir / f"{args.instrument}_{args.bar}.progress.json"

    hours = _parse_hours(args.hours)
    days = pd.date_range(args.start, args.end, freq="D", inclusive="left")
    days = [d.date() for d in days if d.dayofweek < 5]   # weekends are empty by construction

    done: set[str] = set()
    if state_path.exists():
        done = set(json.loads(state_path.read_text()).get("complete_days", []))
    prior = pd.read_parquet(dest) if dest.exists() else None
    if prior is not None:
        prior["timestamp"] = pd.to_datetime(prior["timestamp"], utc=True)
        log.info("resume: %s has %d bars, %d days marked complete", dest.name, len(prior), len(done))

    todo = [d for d in days if d.isoformat() not in done]
    log.info("%s: %d weekdays to fetch (%d already complete), hours=%s",
             args.instrument, len(todo), len(days) - len(todo), hours)
    if not todo:
        log.info("nothing to do")
        return 0

    new_bars: list[pd.DataFrame] = []
    total_failed = 0
    n_empty = 0
    pacer = Pacer(min_delay=args.min_delay)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, day in enumerate(todo, 1):
            ticks, failed = fetch_day(args.instrument, day, hours, ex, pacer)
            total_failed += failed
            if ticks is not None and not ticks.empty:
                new_bars.append(to_bars(ticks, args.bar))
            elif failed == 0:
                n_empty += 1                   # genuine market holiday
            if failed == 0:
                done.add(day.isoformat())      # ONLY a fully-answered day counts as complete
            if i % 25 == 0 or i == len(todo):
                got = sum(len(b) for b in new_bars)
                log.info("  %d/%d days (%.1f%%) | %d new bars | %d holidays | %d failed hours "
                         "| pace %.2fs",
                         i, len(todo), 100 * i / len(todo), got, n_empty, total_failed,
                         pacer.delay)
            if i % 100 == 0 or i == len(todo):  # checkpoint so a kill does not lose the pull
                _flush(dest, state_path, prior, new_bars, done)

    _flush(dest, state_path, prior, new_bars, done)
    final = pd.read_parquet(dest)
    log.info("wrote %s: %d bars (%s .. %s)", dest, len(final),
             final["timestamp"].min(), final["timestamp"].max())
    if total_failed:
        log.warning("%d hour-requests FAILED after retries — their days are NOT marked complete; "
                    "re-run the same command to repair", total_failed)

    if args.expect_price_range:
        lo, hi = args.expect_price_range
        c = final["close"]
        if c.min() < lo or c.max() > hi:
            log.error("PRICE SANITY FAILED: closes span %.4f..%.4f, expected within %.1f..%.1f "
                      "— almost certainly a wrong scaled-int divisor", c.min(), c.max(), lo, hi)
            return 1
        log.info("price sanity OK: closes span %.4f..%.4f", c.min(), c.max())
    return 0


def _flush(dest: Path, state_path: Path, prior: pd.DataFrame | None,
           new_bars: list[pd.DataFrame], done: set[str]) -> None:
    if new_bars:
        bars = pd.concat(new_bars, ignore_index=True)
        if prior is not None:
            bars = pd.concat([prior, bars], ignore_index=True)
        bars = bars.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        bars.to_parquet(dest, index=False)
    state_path.write_text(json.dumps({"complete_days": sorted(done)}))


if __name__ == "__main__":
    raise SystemExit(main())
