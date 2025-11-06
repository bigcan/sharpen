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
from typing import Iterable

import pandas as pd

from finrl_pro.data.db import DatabaseClient, MarketBar
from finrl_pro.data.yahoo_loader import YahooLoader


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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.data.snapshot",
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
        raise SystemExit("Alpaca provider not implemented yet")

    # Fetch data
    loader = YahooLoader()
    df = loader.fetch(tickers=tickers, start=args.start, end=args.end, interval=args.interval)
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
    inserted = db.upsert_bars(bars)

    params = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
    }
    libs = _lib_versions(args.provider)
    db.insert_snapshot(
        snapshot_id=snapshot_id,
        provider=args.provider,
        params_json=json.dumps(params, sort_keys=True),
        code_hash=_git_commit_hash(),
        lib_versions_json=json.dumps(libs, sort_keys=True),
        tickers=tickers,
        row_count=inserted,
    )

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
