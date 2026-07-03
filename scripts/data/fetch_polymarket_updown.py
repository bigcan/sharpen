"""Polymarket 5-min up/down market trade history + resolution outcomes.

Free, on-chain-derived trade data for Polymarket's "Up or Down" 5-minute binary
markets (BTC/ETH/SOL/XRP/DOGE/BNB/HYPE). Polymarket runs on Polygon; every
executed trade is a permanent public blockchain event (CTF Exchange
``OrderFilled``), independent of Polymarket's own REST history endpoint
(which stopped serving order-book history 2026-02-20 — see
``.agent/artifacts/polymarket_vilkov_blocker_audit_2026-07-02.md``). This
script does NOT hit the blockchain directly; it reads the community-maintained
mirror ``SII-WANGZJ/Polymarket_data`` (HuggingFace, MIT license, ~171GB total,
continuously updated), which already decodes those events into a flat trades
table. Only the rows inside the requested date range are downloaded, via
targeted parquet row-group reads (fsspec range requests) — never the full file.

What this does NOT give you: resting order-book depth. Only executed trades
are on-chain; unfilled quotes are off-chain and were never recorded anywhere
before you start watching live. A market-making backtest that needs book
depth for fill simulation cannot be built from this data alone.

Trade resolution (which outcome token paid out $1) is NOT reliable in the
bulk dataset's ``markets.parquet`` (its ``outcome_prices`` field showed
placeholder-looking values clustered at 0.5 for many already-closed 5-min
markets — do not trust it). This script instead queries Polymarket's live
CLOB API (``GET /markets/{condition_id}``, free, no auth) per market, which
returns the authoritative on-chain settlement (``tokens[].winner``).

Findings from the first run of this script (2026-07-02, session S553-cont-102/103):
period 2026-04-04 to 2026-05-04, top-10 most active two-sided ("dual-sided
maker") wallets by market count = 98.4% of that period's entire 5-min
up/down market universe. Realized P&L (maker-side only, true resolution,
BEFORE the 20-25% maker fee-rebate program): 9/10 wallets net profitable,
aggregate +$269,882 on $30.6M maker volume = +88.2 bps blended. One wallet
(0x76D4D4703ADD6e94cFDb1107F3d991D85fF2c512) was a consistent loser across
both an 8-day and a 1-month sample.

Usage:
    python scripts/data/fetch_polymarket_updown.py --selftest
    python scripts/data/fetch_polymarket_updown.py \
        --start 2026-04-04 --end 2026-05-04 --out data/polymarket_updown

Output (under --out):
    markets.parquet                     full up/down market metadata (all horizons, all history)
    trades_<start>_<end>.parquet        maker/taker trade fills for 5-min up/down markets in range
    resolutions_<start>_<end>.parquet   condition_id -> winning token_id, fetched live from CLOB API
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

log = logging.getLogger("fetch_polymarket_updown")

REPO_ID = "SII-WANGZJ/Polymarket_data"
CLOB_MARKET_URL = "https://clob.polymarket.com/markets/{condition_id}"


def fetch_markets(out_dir: Path) -> pd.DataFrame:
    """Full market metadata table (~170MB) — small enough to always fetch whole."""
    p = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="markets.parquet", local_dir=str(out_dir))
    return pd.read_parquet(p)


def _row_groups_for_range(pf: pq.ParquetFile, start_ts: int, end_ts: int) -> list[int]:
    """Binary-search-free linear scan of row-group timestamp stats (cheap: metadata only)."""
    ts_idx = pf.schema_arrow.get_field_index("timestamp")
    keep = []
    for i in range(pf.num_row_groups):
        stats = pf.metadata.row_group(i).column(ts_idx).statistics
        if stats.max >= start_ts and stats.min <= end_ts:
            keep.append(i)
    return keep


def fetch_trades(
    updown5m_condition_ids: set[str], start_ts: int, end_ts: int, retries: int = 6
) -> pa.Table:
    """Stream trades.parquet row groups in [start_ts, end_ts], filtering to the given
    condition_ids as each row group lands so memory stays bounded (full trades.parquet
    is ~28GB; the 5-min up/down family alone can still be tens of millions of rows)."""
    cid_array = pa.array(list(updown5m_condition_ids))
    url = hf_hub_url(REPO_ID, "trades.parquet", repo_type="dataset")

    def open_pf():
        for attempt in range(retries):
            try:
                return pq.ParquetFile(fsspec.open(url, "rb").open())
            except Exception:
                log.warning("open trades.parquet failed, retry %d", attempt)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("could not open trades.parquet after retries")

    pf = open_pf()
    row_groups = _row_groups_for_range(pf, start_ts, end_ts)
    log.info("date range spans %d row groups", len(row_groups))

    filtered = []
    for n, i in enumerate(row_groups):
        tbl = None
        for attempt in range(retries):
            try:
                tbl = pf.read_row_group(i)
                break
            except Exception as e:
                log.warning("row group %d read failed (attempt %d): %s", i, attempt, e)
                time.sleep(3 * (attempt + 1))
                try:
                    pf = open_pf()
                except Exception:
                    pass
        if tbl is None:
            log.error("row group %d permanently failed, skipping (data gap)", i)
            continue
        mask = pc.is_in(tbl.column("condition_id"), value_set=cid_array)
        kept = tbl.filter(mask)
        if kept.num_rows:
            filtered.append(kept)
        if n % 10 == 0:
            log.info("row group %d/%d done", n, len(row_groups))

    if not filtered:
        return pa.table({})
    return pa.concat_tables(filtered)


def fetch_resolutions(condition_ids: list[str], workers: int = 8) -> pd.DataFrame:
    """Authoritative settlement outcome per market, from the live CLOB API.

    IMPORTANT: keep workers modest (<=12). At 25 concurrent workers this endpoint
    silently rate-limits — requests come back with no data and no error surfaced,
    which looks exactly like "market unresolved" and will corrupt P&L math if not
    caught. Verified empirically 2026-07-02: same requests succeed 100% at 8-12
    workers with retry+backoff, ~0/10332 failures.
    """

    def fetch_one(cid: str, session: requests.Session):
        last_err = "unknown"
        for attempt in range(4):
            try:
                r = session.get(CLOB_MARKET_URL.format(condition_id=cid), timeout=15)
                if r.status_code == 200:
                    tokens = r.json().get("tokens", [])
                    if len(tokens) == 2:
                        winner = next((t["token_id"] for t in tokens if t.get("winner")), None)
                        return (cid, winner, "ok")
                    return (cid, None, "bad_shape")
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                return (cid, None, f"http_{r.status_code}")
            except Exception as e:
                last_err = type(e).__name__
                time.sleep(1.0 * (attempt + 1))
        return (cid, None, f"failed_{last_err}")

    def worker(cid: str):
        return fetch_one(cid, requests.Session())

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, c): c for c in condition_ids}
        for n, fut in enumerate(as_completed(futs)):
            results.append(fut.result())
            if n % 3000 == 0:
                log.info("resolution fetch %d/%d", n, len(condition_ids))

    df = pd.DataFrame(results, columns=["condition_id", "winner_token", "status"])
    n_ok = df.winner_token.notna().sum()
    log.info("resolved %d/%d (status counts: %s)", n_ok, len(df), df.status.value_counts().to_dict())
    if n_ok < 0.95 * len(df):
        log.error(
            "resolution rate <95%% — likely rate-limited, NOT a genuine data gap. "
            "Lower `workers` and retry before trusting this output for P&L."
        )
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--out", default="data/polymarket_updown")
    ap.add_argument("--resolution-workers", type=int, default=8)
    ap.add_argument("--selftest", action="store_true", help="Sanity-check imports/API reachability, no download")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        r = requests.get("https://gamma-api.polymarket.com/markets?limit=1", timeout=10)
        assert r.status_code == 200, f"Gamma API unreachable: {r.status_code}"
        log.info("selftest OK - Gamma API reachable, no download performed")
        return

    if not args.start or not args.end:
        ap.error("--start/--end required unless --selftest")

    start_ts = int(datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400 - 1

    log.info("fetching markets.parquet")
    markets = fetch_markets(out_dir)
    u5 = markets[markets.slug.str.contains("updown-5m", na=False)]
    u5_cids = set(u5.condition_id)
    log.info("5-min up/down family: %d markets since inception", len(u5_cids))

    log.info("streaming trades.parquet for %s..%s", args.start, args.end)
    trades_tbl = fetch_trades(u5_cids, start_ts, end_ts)
    trades_path = out_dir / f"trades_{args.start}_{args.end}.parquet"
    pq.write_table(trades_tbl, trades_path)
    log.info("wrote %d trade rows -> %s", trades_tbl.num_rows, trades_path)

    cids_in_range = pc.unique(trades_tbl.column("condition_id")).to_pylist()
    log.info("fetching resolutions for %d markets", len(cids_in_range))
    res_df = fetch_resolutions(cids_in_range, workers=args.resolution_workers)
    res_path = out_dir / f"resolutions_{args.start}_{args.end}.parquet"
    res_df[["condition_id", "winner_token"]].to_parquet(res_path)
    log.info("wrote resolutions -> %s", res_path)


if __name__ == "__main__":
    main()
