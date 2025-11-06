"""CLI: Create a database-backed data snapshot.

Parses provider parameters, fetches data (to be implemented), upserts
into DB, records a snapshot row, and prints JSON with `snapshot_id`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable


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

    # Placeholder behavior per spec: wire CLI and message until DB implemented
    payload = {
        "message": "Snapshot CLI stub. DB interactions not yet implemented.",
        "provider": args.provider,
        "tickers": args.tickers.split(","),
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    main()

