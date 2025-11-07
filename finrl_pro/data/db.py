"""Database helpers for FinRL Pro data snapshots.

This module defines a thin client interface for interacting with a
TimescaleDB/PostgreSQL backend. Implementations are placeholders to
be filled in per specs/001-db-snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Callable, Any
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
    """DB client for TimescaleDB/PostgreSQL interactions.

    Provides snapshot storage, OHLCV upserts, and a simple feature store API
    for custom engineered features keyed by a cache key and linked to a base
    data snapshot.
    """

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

    # ----------------------------- Feature Store -----------------------------

    def init_feature_store(self) -> None:
        """Initialize feature store tables: feature_sets and feature_values."""
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS feature_sets (
              feature_set_id UUID PRIMARY KEY,
              snapshot_id UUID NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
              cache_key TEXT NOT NULL,
              config_json JSONB NOT NULL,
              code_hash TEXT NOT NULL,
              lib_versions_json JSONB NOT NULL,
              created_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              CONSTRAINT feature_sets_uniq UNIQUE (snapshot_id, cache_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feature_values (
              timestamp TIMESTAMPTZ NOT NULL,
              ticker TEXT NOT NULL,
              feature_set_id UUID NOT NULL REFERENCES feature_sets(feature_set_id) ON DELETE CASCADE,
              name TEXT NOT NULL,
              value DOUBLE PRECISION NOT NULL,
              PRIMARY KEY (timestamp, ticker, feature_set_id, name)
            )
            """,
            # If Timescale is available, convert feature_values to hypertable
            "SELECT 1 FROM create_hypertable('feature_values', by_range('timestamp'), if_not_exists => TRUE)",
        ]
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                for stmt in stmts:
                    try:
                        cur.execute(stmt)
                    except Exception:
                        conn.rollback()
                        continue
                conn.commit()

    def get_feature_set(self, *, snapshot_id: str, cache_key: str) -> str | None:
        """Return feature_set_id if present for (snapshot_id, cache_key)."""
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT feature_set_id FROM feature_sets WHERE snapshot_id=%s AND cache_key=%s",
                    (snapshot_id, cache_key),
                )
                row = cur.fetchone()
                return row[0] if row else None

    def insert_feature_set(
        self,
        *,
        feature_set_id: str,
        snapshot_id: str,
        cache_key: str,
        config_json: str,
        code_hash: str,
        lib_versions_json: str,
    ) -> None:
        """Insert a feature set record if not existing."""
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feature_sets (feature_set_id, snapshot_id, cache_key, config_json, code_hash, lib_versions_json)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (snapshot_id, cache_key) DO NOTHING
                    """,
                    (feature_set_id, snapshot_id, cache_key, config_json, code_hash, lib_versions_json),
                )
            conn.commit()

    def upsert_feature_values(self, feature_set_id: str, rows: Iterable[Mapping[str, object]], *, batch: int = 5000) -> int:
        """Upsert feature values (timestamp,ticker,feature_set_id,name,value)."""
        total = 0
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                sql = (
                    "INSERT INTO feature_values (timestamp, ticker, feature_set_id, name, value) "
                    "VALUES (%(timestamp)s, %(ticker)s, %(feature_set_id)s, %(name)s, %(value)s) "
                    "ON CONFLICT (timestamp, ticker, feature_set_id, name) DO UPDATE SET value=EXCLUDED.value"
                )
                batch_rows: list[Mapping[str, object]] = []
                for r in rows:
                    batch_rows.append(r)
                    if len(batch_rows) >= batch:
                        cur.executemany(sql, batch_rows)
                        total += len(batch_rows)
                        batch_rows.clear()
                if batch_rows:
                    cur.executemany(sql, batch_rows)
                    total += len(batch_rows)
            conn.commit()
        return total

    def fetch_features(
        self,
        *,
        feature_set_id: str,
        tickers: Sequence[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch feature values for a set of tickers and optional date range."""
        placeholders = ",".join(["%s"] * len(tickers))
        where = ["ticker IN (" + placeholders + ")", "feature_set_id = %s"]
        args: list[object] = [*tickers, feature_set_id]
        if start:
            where.append("timestamp >= %s")
            args.append(start)
        if end:
            where.append("timestamp <= %s")
            args.append(end)
        sql = (
            "SELECT timestamp, ticker, name, value FROM feature_values WHERE "
            + " AND ".join(where)
            + " ORDER BY timestamp ASC, ticker ASC"
        )
        with self._connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "tic", "name", "value"])
        df = pd.DataFrame(rows, columns=["timestamp", "tic", "name", "value"])
        return df

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
