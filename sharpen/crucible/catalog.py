"""Central data catalog — the map of "what can I hypothesize over today" (spec §4.4).

The inventory flagged the missing piece: no central catalog recording which data domains are
available, how fresh, and under what point-in-time policy. Stage 1 (ACQUIRE) registers every
dataset here after it passes the data-quality gate; Stage 2 (HYPOTHESIZE) queries it. Each row
carries a ``snapshot_hash`` so a ``crucible-vN`` run pins an exact panel snapshot (CR-5), and the
catalog-wide :meth:`DataCatalog.snapshot_hash` feeds ``run_manifest.data_snapshot_hash``.

P0 stands up the SQLite schema + registration/query/hash API only. The connectors that FILL it
(FRED/ALFRED, CFTC COT, …) and the non-OHLCV quality gate arrive in P1b.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

# The data domains the small-operator breadth strategy targets (spec §4). 'market' is the already-
# wired OHLCV world; the others are the gaps this subsystem fills.
ASSET_CLASSES = frozenset(
    {"macro", "positioning", "sentiment", "onchain", "fundamental", "market", "weather", "attention"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS catalog (
    source_id      TEXT NOT NULL,        -- e.g. 'fred', 'cftc_cot', 'gdelt'
    series         TEXT NOT NULL,        -- series/ticker key within the source
    asset_class    TEXT NOT NULL,        -- one of ASSET_CLASSES
    date_start     TEXT,
    date_end       TEXT,
    freshness      TEXT,                 -- ISO stamp of the last successful refresh
    as_of_policy   TEXT,                 -- PIT policy: vintage api | release-lag | none-rejected
    snapshot_hash  TEXT,                 -- content hash of the pinned series snapshot
    license        TEXT,
    PRIMARY KEY (source_id, series)
);
CREATE INDEX IF NOT EXISTS ix_catalog_asset_class ON catalog(asset_class);
"""


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One registered data series (spec §4.4)."""

    source_id: str
    series: str
    asset_class: str
    date_start: str | None = None
    date_end: str | None = None
    freshness: str | None = None
    as_of_policy: str | None = None
    snapshot_hash: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(
                f"asset_class must be one of {sorted(ASSET_CLASSES)}; got {self.asset_class!r}"
            )


class DataCatalog:
    """SQLite-backed catalog of point-in-time-gated data series (spec §4.4)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DataCatalog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def register(self, entry: CatalogEntry) -> None:
        """Upsert a series by (source_id, series). Re-registering refreshes the snapshot/freshness
        rather than duplicating — the catalog is keyed on the series identity."""
        cols = [
            "source_id", "series", "asset_class", "date_start", "date_end", "freshness",
            "as_of_policy", "snapshot_hash", "license",
        ]
        vals = [getattr(entry, c) for c in cols]
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("source_id", "series"))
        self._conn.execute(
            f"INSERT INTO catalog ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(source_id, series) DO UPDATE SET {updates}",
            vals,
        )
        self._conn.commit()

    def list_series(self, asset_class: str | None = None) -> list[dict]:
        """All registered series, optionally filtered to one asset class. Deterministically ordered
        so the catalog snapshot is reproducible."""
        if asset_class is not None:
            rows = self._conn.execute(
                "SELECT * FROM catalog WHERE asset_class = ? ORDER BY source_id, series",
                (asset_class,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM catalog ORDER BY source_id, series"
            ).fetchall()
        return [dict(r) for r in rows]

    def snapshot_hash(self) -> str:
        """Deterministic 12-hex SHA-256 over the catalog rows (spec §4.4/§5). Pins the exact data
        surface a run saw; feeds ``run_manifest.data_snapshot_hash``. Empty catalog → a stable hash
        of ``[]`` (a run over no registered series is still reproducibly identified)."""
        payload = json.dumps([asdict(CatalogEntry(**{
            k: r[k] for k in (
                "source_id", "series", "asset_class", "date_start", "date_end", "freshness",
                "as_of_policy", "snapshot_hash", "license",
            )
        })) for r in self.list_series()], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
