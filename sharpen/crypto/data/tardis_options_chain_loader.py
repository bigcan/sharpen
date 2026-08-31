"""Free real-chain loader: Tardis.dev first-of-month Deribit options snapshots.

Tardis serves the **first day of each month** for free (no API key) at
``datasets.tardis.dev``. The ``options_chain`` dataset for Deribit contains, per
instrument, the **real** bid/ask price + IV, mark price + IV, full greeks, strike,
expiration and underlying price. This is the Phase-3 "real chain" fidelity source
that the free Deribit public API could NOT provide (it prunes expired instruments
and serves no per-option history).

We only need ONE cross-section per month, so we **stream** each daily gz file and
**break early** once the first snapshot window is captured — pulling a few MB, not
the full ~700 MB/day tick file. Each instrument's first quote of the day (≈00:00
UTC) is kept (dedupe by symbol).

This is causally clean for a monthly sell-and-hold backtest: the snapshot at the
1st is the entry; the option is held to its own expiry and settled at intrinsic
from daily spot (no future-snapshot look-ahead). Option prices are in COIN terms
(Deribit convention); USD premium = price * underlying_price.

Note: a monthly cross-section is, by construction, survivorship-free for the
sell-and-hold use — we trade only instruments quoted at entry and settle them at
their own expiry; we never need a future chain that could drop dead instruments.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CSV_URL = "https://datasets.tardis.dev/v1/deribit/options_chain/{y}/{m:02d}/01/OPTIONS.csv.gz"
USER_AGENT = "finrl-pro-ds/vrp-harvest (free first-of-month chain loader)"
DEFAULT_CACHE_DIR = Path("data/processed/deribit_chain")

# Columns kept from the (24-column) options_chain schema.
_KEEP = ["symbol", "timestamp", "type", "strike_price", "expiration",
         "underlying_price", "bid_price", "bid_iv", "ask_price", "ask_iv",
         "mark_price", "mark_iv", "open_interest", "delta", "gamma", "vega", "theta"]
_FLOAT = ["strike_price", "underlying_price", "bid_price", "bid_iv", "ask_price",
          "ask_iv", "mark_price", "mark_iv", "open_interest", "delta", "gamma",
          "vega", "theta"]

_SNAPSHOT_WINDOW_US = 60_000_000   # keep rows within first 60s of the day
_MAX_ROWS = 400_000                # safety cap on rows streamed per file


def fetch_month_snapshot(year: int, month: int, *, currency: str | None = None,
                         window_us: int = _SNAPSHOT_WINDOW_US,
                         max_rows: int = _MAX_ROWS) -> pd.DataFrame:
    """Stream the free first-of-month Deribit options_chain and return ONE snapshot.

    Returns an empty DataFrame if the dataset is absent (404/403) for that month.
    ``currency`` optionally filters to symbols starting with e.g. "BTC".
    """
    url = CSV_URL.format(y=year, m=month)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    rows: dict[str, dict] = {}
    t0: int | None = None
    n = 0
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            logger.warning("No free chain for %d-%02d (HTTP %d) — skipping", year, month, exc.code)
            return pd.DataFrame(columns=_KEEP)
        raise
    try:
        gz = gzip.GzipFile(fileobj=resp)
        reader = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8"))
        for row in reader:
            n += 1
            if n > max_rows:
                break
            ts = int(row["timestamp"])
            if t0 is None:
                t0 = ts
            if ts > t0 + window_us:
                break
            sym = row["symbol"]
            if currency and not sym.startswith(currency):
                continue
            if sym not in rows:  # keep FIRST quote per instrument
                rows[sym] = {k: row[k] for k in _KEEP}
    finally:
        resp.close()

    if not rows:
        return pd.DataFrame(columns=_KEEP)
    df = pd.DataFrame(list(rows.values()))
    for c in _FLOAT:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["expiration"] = pd.to_numeric(df["expiration"], errors="coerce")
    df["snapshot_date"] = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    df["expiry_dt"] = pd.to_datetime(df["expiration"], unit="us", utc=True)
    logger.info("Snapshot %d-%02d: %d instruments (streamed %d rows)", year, month, len(df), n)
    return df


def _month_iter(start_ym: str, end_ym: str):
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def load_chain_snapshots(start_ym: str = "2021-04", end_ym: str = "2026-06",
                         *, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                         currency: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download/cache free first-of-month chain snapshots and return them combined."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    months_loaded = []
    for y, m in _month_iter(start_ym, end_ym):
        p = cache_dir / f"chain_{y}_{m:02d}.parquet"
        if p.exists() and not refresh:
            df = pd.read_parquet(p)
        else:
            df = fetch_month_snapshot(y, m, currency=currency)
            if not df.empty:
                df.to_parquet(p)
        if not df.empty:
            frames.append(df)
            months_loaded.append(f"{y}-{m:02d}")
    if not frames:
        return pd.DataFrame(columns=_KEEP)
    combined = pd.concat(frames, ignore_index=True)

    manifest = {
        "source": "tardis_free_first_of_month",
        "dataset": "deribit/options_chain",
        "start_ym": start_ym, "end_ym": end_ym,
        "months_loaded": months_loaded,
        "n_months": len(months_loaded),
        "n_rows": int(len(combined)),
        "currency_filter": currency or "all",
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Loaded %d monthly snapshots, %d instrument-rows", len(months_loaded), len(combined))
    return combined


def _cli() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Cache free Tardis first-of-month Deribit option chains")
    ap.add_argument("--start", default="2021-04")
    ap.add_argument("--end", default="2026-06")
    ap.add_argument("--currency", default=None, help="filter e.g. BTC or ETH")
    ap.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = load_chain_snapshots(args.start, args.end, cache_dir=args.cache_dir,
                              currency=args.currency, refresh=args.refresh)
    logger.info("Combined rows: %d", len(df))


if __name__ == "__main__":
    _cli()
