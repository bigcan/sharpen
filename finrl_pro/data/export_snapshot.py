"""CLI: Export a database-backed snapshot to file.

Materializes a snapshot to Parquet/CSV (future), verifying row counts
match the DB slice. Currently a stub until DB plumbing is implemented.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from finrl_pro.data.db import DatabaseClient


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finrl_pro.data.export_snapshot",
        description="Export a snapshot to a file format (Parquet/CSV).",
    )
    parser.add_argument("--id", required=True, help="Snapshot identifier")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="File format")
    args = parser.parse_args(list(argv) if argv is not None else None)

    dsn = os.getenv("FINRL_PRO_DB_DSN", "")
    db = DatabaseClient(dsn=dsn)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.id}.{args.format}"
    path = db.export_snapshot(args.id, fmt=args.format, out_path=str(out_path))
    print(json.dumps({"snapshot_id": args.id, "path": path, "format": args.format}, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
