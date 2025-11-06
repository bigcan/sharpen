"""Database helpers for FinRL Pro data snapshots.

This module defines a thin client interface for interacting with a
TimescaleDB/PostgreSQL backend. Implementations are placeholders to
be filled in per specs/001-db-snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(slots=True)
class MarketBar:
    timestamp: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    vendor_rev: int


class DatabaseClient:
    """Placeholder DB client for TimescaleDB interactions."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn

    def init_schema(self) -> None:
        """Create schema objects (hypertable and tables).

        Per spec: create `market_bars` hypertable and `snapshots`,
        `snapshot_assets` tables.
        """
        raise NotImplementedError("DB schema initialization not yet implemented.")

    def upsert_bars(self, bars: Iterable[MarketBar]) -> int:
        """Batch upsert OHLCV bars, returning affected row count."""
        raise NotImplementedError("Upsert not yet implemented.")

    def insert_snapshot(
        self,
        *,
        snapshot_id: str,
        provider: str,
        params_json: str,
        code_hash: str,
        lib_versions_json: str,
        tickers: Sequence[str],
        row_count: int,
    ) -> None:
        """Insert a snapshot record and its assets mapping."""
        raise NotImplementedError("Snapshot insert not yet implemented.")

    def load_snapshot(self, snapshot_id: str) -> Sequence[MarketBar]:
        """Load bars for a given snapshot id."""
        raise NotImplementedError("Snapshot query not yet implemented.")

    def export_snapshot(self, snapshot_id: str, *, fmt: str, out_path: str) -> str:
        """Materialize a snapshot to `out_path` and return the file path."""
        raise NotImplementedError("Snapshot export not yet implemented.")

