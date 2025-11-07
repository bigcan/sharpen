"""Database helpers for FinRL Pro data snapshots.

This module defines a thin client interface for interacting with a
TimescaleDB/PostgreSQL backend. Implementations are placeholders to
be filled in per specs/001-db-snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Callable, Any
import json
import os

import pandas as pd
import psycopg


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

    def __init__(self, dsn: str | None = None, *, connect: Callable[..., Any] | None = None) -> None:
        self._dsn = dsn or os.getenv("FINRL_PRO_DB_DSN") or ""
        self._connect = connect or psycopg.connect

    def init_schema(self) -> None:
        """Create schema objects (hypertable and tables).

        Per spec: create `market_bars` hypertable and `snapshots`,
        `snapshot_assets` tables.
        """
        stmts = [
            # Enable Timescale extension if present; ignore if not installed.
            "CREATE EXTENSION IF NOT EXISTS timescaledb",
            # market_bars table with unique constraint for upsert
            """
            CREATE TABLE IF NOT EXISTS market_bars (
              timestamp TIMESTAMPTZ NOT NULL,
              ticker TEXT NOT NULL,
              open DOUBLE PRECISION NOT NULL,
              high DOUBLE PRECISION NOT NULL,
              low DOUBLE PRECISION NOT NULL,
              close DOUBLE PRECISION NOT NULL,
              volume DOUBLE PRECISION NOT NULL,
              source TEXT NOT NULL,
              vendor_rev INTEGER NOT NULL DEFAULT 1,
              ingest_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              CONSTRAINT market_bars_uniq UNIQUE (timestamp, ticker)
            )
            """,
            # Create hypertable if Timescale is available; ignore errors otherwise
            "SELECT 1 FROM create_hypertable('market_bars', by_range('timestamp'), if_not_exists => TRUE)",
            # snapshots table
            """
            CREATE TABLE IF NOT EXISTS snapshots (
              snapshot_id UUID PRIMARY KEY,
              provider TEXT NOT NULL,
              params_json JSONB NOT NULL,
              created_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              code_hash TEXT NOT NULL,
              lib_versions_json JSONB NOT NULL,
              row_count INTEGER NOT NULL
            )
            """,
            # snapshot_assets table
            """
            CREATE TABLE IF NOT EXISTS snapshot_assets (
              snapshot_id UUID NOT NULL,
              ticker TEXT NOT NULL,
              PRIMARY KEY (snapshot_id, ticker),
              FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
            )
            """,
        ]
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                for stmt in stmts:
                    try:
                        cur.execute(stmt)
                    except Exception:
                        # Be tolerant if Timescale functions are not installed.
                        conn.rollback()
                        continue
                conn.commit()

    def upsert_bars(self, bars: Iterable[MarketBar]) -> int:
        """Batch upsert OHLCV bars, returning affected row count."""
        if not bars:
            return 0
        with self._connect(self._dsn) as conn:
            from dataclasses import asdict
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO market_bars (timestamp, ticker, open, high, low, close, volume, source, vendor_rev) "
                    "VALUES (%(timestamp)s, %(ticker)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s, %(vendor_rev)s) "
                    "ON CONFLICT (timestamp, ticker) DO UPDATE SET "
                    "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, "
                    "volume=EXCLUDED.volume, source=EXCLUDED.source, vendor_rev=EXCLUDED.vendor_rev"
                )
                payloads = []
                for bar in bars:
                    try:
                        payloads.append(asdict(bar))
                    except Exception:
                        # Fallback if already a Mapping-like
                        payloads.append(dict(bar))
                cur.executemany(sql, payloads)
            conn.commit()
            return len(payloads)

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
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO snapshots (snapshot_id, provider, params_json, code_hash, lib_versions_json, row_count)
                    VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s)
                    ON CONFLICT (snapshot_id) DO NOTHING
                    """,
                    (snapshot_id, provider, params_json, code_hash, lib_versions_json, row_count),
                )
                if tickers:
                    cur.executemany(
                        "INSERT INTO snapshot_assets (snapshot_id, ticker) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        [(snapshot_id, t) for t in tickers],
                    )
            conn.commit()

    def load_snapshot(self, snapshot_id: str) -> Sequence[MarketBar]:
        """Load bars for a given snapshot id."""
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT params_json FROM snapshots WHERE snapshot_id=%s", (snapshot_id,))
                row = cur.fetchone()
                if not row:
                    return []
                params = row[0]
                start = params.get("start") if isinstance(params, dict) else None
                end = params.get("end") if isinstance(params, dict) else None
                cur.execute("SELECT ticker FROM snapshot_assets WHERE snapshot_id=%s", (snapshot_id,))
                tickers = [r[0] for r in cur.fetchall()]
                if not tickers:
                    return []
                placeholders = ",".join(["%s"] * len(tickers))
                where = [f"ticker IN ({placeholders})"]
                args = [*tickers]
                if start:
                    where.append("timestamp >= %s")
                    args.append(start)
                if end:
                    where.append("timestamp <= %s")
                    args.append(end)
                sql = (
                    "SELECT timestamp, ticker, open, high, low, close, volume, source, vendor_rev "
                    "FROM market_bars WHERE " + " AND ".join(where) + " ORDER BY timestamp ASC, ticker ASC"
                )
                cur.execute(sql, args)
                results = []
                for r in cur.fetchall():
                    results.append(
                        MarketBar(
                            timestamp=r[0].isoformat(),
                            ticker=r[1],
                            open=float(r[2]),
                            high=float(r[3]),
                            low=float(r[4]),
                            close=float(r[5]),
                            volume=float(r[6]),
                            source=str(r[7]),
                            vendor_rev=int(r[8]),
                        )
                    )
                return results

    def export_snapshot(self, snapshot_id: str, *, fmt: str, out_path: str) -> str:
        """Materialize a snapshot to `out_path` and return the file path."""
        bars = self.load_snapshot(snapshot_id)
        if not bars:
            raise ValueError(f"No bars found for snapshot {snapshot_id}")
        from dataclasses import asdict, is_dataclass
        rows = []
        for b in bars:
            if is_dataclass(b):
                rows.append(asdict(b))
            else:
                try:
                    rows.append(dict(b))
                except Exception:
                    # Fallback: attribute access
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
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(out_path)
        elif fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        return out_path
