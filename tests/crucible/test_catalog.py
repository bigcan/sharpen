"""Data-catalog tripwires — the panel-snapshot pin (spec §4.4/§5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sharpen.crucible.catalog import CatalogEntry, DataCatalog


def _entry(source: str, series: str, ac: str = "macro") -> CatalogEntry:
    return CatalogEntry(source_id=source, series=series, asset_class=ac,
                        date_start="2010-01-01", date_end="2026-06-30",
                        as_of_policy="alfred_vintage", license="public-domain")


def test_bad_asset_class_rejected() -> None:
    with pytest.raises(ValueError):
        CatalogEntry(source_id="fred", series="T10Y2Y", asset_class="not_a_class")


def test_register_upsert_and_list(tmp_path: Path) -> None:
    with DataCatalog(tmp_path / "catalog.db") as cat:
        cat.register(_entry("fred", "T10Y2Y", "macro"))
        cat.register(_entry("cftc_cot", "comm_net_z", "positioning"))
        assert len(cat.list_series()) == 2
        assert [r["series"] for r in cat.list_series("macro")] == ["T10Y2Y"]
        cat.register(_entry("fred", "T10Y2Y", "macro"))          # same key → upsert, not duplicate
        assert len(cat.list_series()) == 2


def test_snapshot_hash_stable_and_row_sensitive(tmp_path: Path) -> None:
    with DataCatalog(tmp_path / "catalog.db") as cat:
        empty = cat.snapshot_hash()
        assert len(empty) == 12                                  # stable hash of an empty catalog
        cat.register(_entry("fred", "T10Y2Y", "macro"))
        h1 = cat.snapshot_hash()
        assert h1 != empty                                       # a registered row moves the hash
        assert h1 == cat.snapshot_hash()                         # ... and is deterministic
        cat.register(_entry("cftc_cot", "comm_net_z", "positioning"))
        assert cat.snapshot_hash() != h1
