"""CLI: Create a database-backed data snapshot.

Parses provider parameters, fetches data (to be implemented), upserts
into DB, records a snapshot row, and prints JSON with `snapshot_id`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from uuid import uuid4
import time
from typing import Iterable

import pandas as pd

from finrl_pro_ds.data.db import DatabaseClient, MarketBar
from finrl_pro_ds.data.yahoo_loader import YahooLoader
from finrl_pro_ds.data.alpaca_loader import AlpacaLoader
from finrl_pro_ds.mlops.logger import MLOpsLogger


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _lib_versions(provider: str) -> dict[str, str]:
    versions: dict[str, str] = {"pandas": pd.__version__}
    if provider == "yahoo":
        import yfinance as yf  # type: ignore

        versions["yfinance"] = getattr(yf, "__version__", "unknown")
    return versions


def _fetch_with_retry(loader: YahooLoader, *, tickers: list[str], start: str, end: str, interval: str, max_retries: int = 3, base_delay: float = 1.0, logger: MLOpsLogger | None = None) -> pd.DataFrame:
    attempt = 0
    while True:
        try:
            df = loader.fetch(tickers=tickers, start=start, end=end, interval=interval)
            if logger:
                logger.log_event(
                    "finrl_pro_ds.snapshot.fetch_success",
                    context={"rows": int(df.shape[0]), "tickers": tickers, "interval": interval},
                )
            return df
        except Exception as e:  # noqa: BLE001
            attempt += 1
            if logger:
                logger.log_event(
                    "finrl_pro_ds.snapshot.fetch_error",
                    context={"attempt": attempt, "error": str(e)},
                )
            if attempt > max_retries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro_ds.data.snapshot",
        description="Create a database-backed market data snapshot.",
    )
    parser.add_argument("--provider", required=True, choices=["yahoo", "alpaca"], help="Data provider")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g., SPY,AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1d", help="Bar interval (provider-specific)")
    parser.add_argument("--out", default="data/snapshots", help="Output directory (used by export workflows)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided")

    if args.provider == "alpaca":
        api_key = os.getenv("ALPACA_API_KEY_ID")
        api_secret = os.getenv("ALPACA_API_SECRET_KEY")
        if not api_key or not api_secret:
            raise SystemExit("Alpaca credentials not set (ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY)")

    # Fetch data
    if args.provider == "yahoo":
        loader = YahooLoader()
    else:
        loader = AlpacaLoader(api_key=api_key, api_secret=api_secret)  # type: ignore[name-defined]
    logger = MLOpsLogger()
    logger.log_event(
        "finrl_pro_ds.snapshot.start",
        context={"provider": args.provider, "tickers": tickers, "start": args.start, "end": args.end, "interval": args.interval},
    )
    df = _fetch_with_retry(
        loader,
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        max_retries=3,
        base_delay=0.1,
        logger=logger,
    )
    if df.empty:
        raise SystemExit("No data returned for given parameters")

    # Prepare records
    bars: list[MarketBar] = []
    for row in df.itertuples(index=False):
        bars.append(
            MarketBar(
                timestamp=str(getattr(row, "timestamp")),
                ticker=str(getattr(row, "ticker")),
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=float(getattr(row, "volume", 0.0)),
                source=args.provider,
                vendor_rev=1,
            )
        )

    snapshot_id = str(uuid4())
    dsn = os.getenv("FINRL_PRO_DB_DSN", "")
    db = DatabaseClient(dsn=dsn)
    try:
        inserted = db.upsert_bars(bars)
    except Exception as e:  # noqa: BLE001
        logger.log_event("finrl_pro_ds.snapshot.db_upsert_error", context={"error": str(e), "rows": len(bars)})
        raise
    logger.log_event("finrl_pro_ds.snapshot.db_upsert", context={"rows": inserted})

    params = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
    }
    libs = _lib_versions(args.provider)
    try:
        db.insert_snapshot(
            snapshot_id=snapshot_id,
            provider=args.provider,
            params_json=json.dumps(params, sort_keys=True),
            code_hash=_git_commit_hash(),
            lib_versions_json=json.dumps(libs, sort_keys=True),
            tickers=tickers,
            row_count=inserted,
        )
    except Exception as e:  # noqa: BLE001
        logger.log_event("finrl_pro_ds.snapshot.db_snapshot_insert_error", context={"error": str(e)})
        raise
    logger.log_event("finrl_pro_ds.snapshot.db_snapshot_inserted", context={"snapshot_id": snapshot_id, "row_count": inserted})

    print(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "row_count": inserted,
                "provider": args.provider,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
