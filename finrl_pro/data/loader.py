"""Module: loader
Purpose: Define data ingestion and preprocessing scaffolds for FinRL Pro.

Adds a resolver for dataset references. If a reference uses the
`snapshot://<snapshot_id>` scheme, it is resolved via the
database-backed snapshots per specs/001-db-snapshots.
"""

from __future__ import annotations


from typing import Optional
from dataclasses import asdict, is_dataclass
import os

import pandas as pd

from finrl_pro.data.db import DatabaseClient

SNAPSHOT_SCHEME = "snapshot://"


class DataLoader:
    """Placeholder for future data loading pipelines."""

    def load(self) -> None:
        """Execute the data loading routine once implemented."""
        raise NotImplementedError("Data loading pipeline not yet implemented.")

    @staticmethod
    def resolve_dataset(dataset_hash: str, *, client: Optional[DatabaseClient] = None) -> pd.DataFrame:
        """Resolve a dataset reference to a concrete source.

        Supported:
        - snapshot://<snapshot_id> → load from DB
        - dvc://... or file paths → to be implemented as needed
        """
        if dataset_hash.startswith(SNAPSHOT_SCHEME):
            snapshot_id = dataset_hash[len(SNAPSHOT_SCHEME) :]
            db = client or DatabaseClient(dsn=os.getenv("FINRL_PRO_DB_DSN", ""))
            bars = db.load_snapshot(snapshot_id)
            if not bars:
                raise ValueError(f"No data found for snapshot '{snapshot_id}'")
            rows = []
            for b in bars:  # accept dataclass or mapping
                if is_dataclass(b):
                    rows.append(asdict(b))
                else:
                    try:
                        rows.append(dict(b))
                    except Exception:
                        rows.append({
                            'timestamp': getattr(b, 'timestamp'),
                            'ticker': getattr(b, 'ticker'),
                            'open': getattr(b, 'open'),
                            'high': getattr(b, 'high'),
                            'low': getattr(b, 'low'),
                            'close': getattr(b, 'close'),
                            'volume': getattr(b, 'volume'),
                            'source': getattr(b, 'source'),
                            'vendor_rev': getattr(b, 'vendor_rev'),
                        })
            df = pd.DataFrame(rows)
            # ensure logical column ordering when present
            cols = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "source", "vendor_rev"]
            present = [c for c in cols if c in df.columns]
            return df[present]
        raise NotImplementedError(f"Unknown dataset reference: {dataset_hash}")
