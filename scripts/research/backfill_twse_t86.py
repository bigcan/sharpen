"""One-shot backfill of the TWSE T86 three-institutional-investors accumulation store (P0,
S553-cont-117).

The Crucible Taiwan overlay slots were power-starved: the connector's old 400-day lookback cap left
each ``twse_inst:*`` series with only ~267 of the panel's ~2,796 bars (~9.5%), so a flow overlay would
have needed an implausibly large marginal edge to clear the pre-registered gates. T86 IS range-
queryable arbitrarily far back (live-verified: full snapshots for 2014 and 2018), so this script fills
the persistent day store (``finrl_pro_ds/crucible/data/twse_institutional._AccumulationStore``) once,
from the panel start (2015-01-05) to today, lifting coverage to ~2,750 bars (SE(SR) ~1.0 -> ~0.30).

It is idempotent and resumable: the store persists per-day and the connector skips already-stored days,
so a re-run continues where an interrupted run left off (checkpointed every ``save_every_days``). The
orchestrator's own connector uses the SAME default store path, so once this has run a nightly tick
re-fetches only genuinely new days.

Causality: the store caches raw reference-day values only; the +1-calendar-day release stamp and the
as_of cutoff are applied at READ time in the connector's ``fetch`` (LEAK-2 preserved, parity-tested).

Usage:
  python scripts/research/backfill_twse_t86.py                       # 2015-01-05 -> today, default store
  python scripts/research/backfill_twse_t86.py --start 2020-01-01    # narrower range
  python scripts/research/backfill_twse_t86.py --sleep 0.5           # gentler politeness delay
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.crucible.data.twse_institutional import (  # noqa: E402
    _DEFAULT_STORE_PATH,
    _DEFAULT_TICKERS,
    _FIELD_COLUMNS,
    TwseInstitutionalConnector,
)

log = logging.getLogger("backfill_twse_t86")


def _report(store_path: Path) -> dict:
    """Coverage summary read straight off the persisted store (never the connector internals)."""
    if not store_path.exists():
        return {"days_total": 0, "days_with_data": 0, "per_ticker": {}}
    store = json.loads(store_path.read_text(encoding="utf-8"))
    per_ticker: dict[str, int] = {tk: 0 for tk in _DEFAULT_TICKERS}
    days_with_data = 0
    for day_rows in store.values():
        if day_rows:
            days_with_data += 1
        for tk in day_rows:
            per_ticker[tk] = per_ticker.get(tk, 0) + 1
    return {"days_total": len(store), "days_with_data": days_with_data, "per_ticker": per_ticker}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Backfill the TWSE T86 institutional accumulation store")
    ap.add_argument("--start", default="2015-01-05", help="ISO date; default = Taiwan panel start")
    ap.add_argument("--end", default=None, help="ISO date; default = today (UTC+8 Taiwan)")
    ap.add_argument("--sleep", type=float, default=0.3, help="politeness delay between live day calls")
    ap.add_argument("--store-path", default=None,
                    help="override the store file (default = the connector's shared default path)")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    # Taiwan is UTC+8; "today" there is the operative trading-calendar day for a fresh publication.
    end = (date.fromisoformat(args.end) if args.end
           else (datetime.now(timezone.utc) + timedelta(hours=8)).date())
    if end < start:
        log.error("end %s precedes start %s — nothing to do", end, start)
        return 1
    store_path = Path(args.store_path) if args.store_path else _DEFAULT_STORE_PATH
    span_days = (end - start).days + 1

    before = _report(store_path)
    log.info("backfill T86 %s -> %s (%d calendar days) into %s", start, end, span_days, store_path)
    log.info("store BEFORE: %d days polled (%d with data)", before["days_total"],
             before["days_with_data"])

    conn = TwseInstitutionalConnector(
        tickers=_DEFAULT_TICKERS,
        sleep_seconds=args.sleep,
        store_path=store_path,
        max_lookback_days=span_days + 10,        # never floor the requested start
        max_live_days_per_fetch=span_days + 10,  # fill the WHOLE range in this one run
    )
    ref = next(r for r in conn.discover() if r.series_id.endswith(":foreign_net"))

    # Year-chunked so a long run emits progress; the store is shared + idempotent across chunks, and
    # the per-instance live budget is set to cover the full span, so chunking never re-fetches a day.
    for year in range(start.year, end.year + 1):
        y_start = max(start, date(year, 1, 1))
        y_end = min(end, date(year, 12, 31))
        conn.fetch(ref, y_start.isoformat(), y_end.isoformat())
        rpt = _report(store_path)
        log.info("  through %d: store now %d days polled, %d with data (0050 coverage=%d)",
                 year, rpt["days_total"], rpt["days_with_data"], rpt["per_ticker"].get("0050", 0))

    after = _report(store_path)
    log.info("store AFTER: %d days polled (%d with data); +%d new days this run",
             after["days_total"], after["days_with_data"],
             after["days_total"] - before["days_total"])
    log.info("per-ticker bar coverage: %s",
             {tk: after["per_ticker"][tk] for tk in _DEFAULT_TICKERS})
    fields = ", ".join(_FIELD_COLUMNS)
    log.info("each covered (ticker, day) carries fields: %s", fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
