"""CLI: Export a database-backed snapshot to file.

Materializes a snapshot to Parquet/CSV (future), verifying row counts
match the DB slice. Currently a stub until DB plumbing is implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.data.export_snapshot",
        description="Export a snapshot to a file format (Parquet/CSV).",
    )
    parser.add_argument("--id", required=True, help="Snapshot identifier")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="File format")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = {
        "message": "Export CLI stub. DB interactions not yet implemented.",
        "snapshot_id": args.id,
        "out": args.out,
        "format": args.format,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    main()

