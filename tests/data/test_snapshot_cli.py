from __future__ import annotations

import io
import json
from typing import Any, Iterable, List, Sequence

import pandas as pd

import finrl_pro.data.snapshot as snap


class _FakeDB:
    def __init__(self):
        self.bars: List[Any] = []
        self.snapshots: List[tuple] = []

    def upsert_bars(self, bars):
        self.bars.extend(bars)
        return len(bars)

    def insert_snapshot(self, *, snapshot_id: str, provider: str, params_json: str, code_hash: str, lib_versions_json: str, tickers: Sequence[str], row_count: int):
        self.snapshots.append((snapshot_id, provider, params_json, code_hash, lib_versions_json, list(tickers), row_count))


class _FakeLoader:
    def fetch(self, *, tickers: list[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        rows = []
        for t in tickers:
            rows.append({
                "timestamp": pd.Timestamp("2020-01-01"),
                "ticker": t,
                "open": 1.0,
                "high": 1.5,
                "low": 0.9,
                "close": 1.2,
                "volume": 100.0,
            })
        return pd.DataFrame(rows)


def test_snapshot_cli_minimal(monkeypatch, capsys):
    fake_db = _FakeDB()
    monkeypatch.setattr(snap, "DatabaseClient", lambda dsn=None: fake_db)
    monkeypatch.setattr(snap, "YahooLoader", lambda: _FakeLoader())

    args = [
        "--provider", "yahoo",
        "--tickers", "SPY,AAPL",
        "--start", "2020-01-01",
        "--end", "2020-01-05",
        "--interval", "1d",
    ]
    snap.main(args)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "snapshot_id" in payload and payload["row_count"] == 2
    # DB received 2 bars and one snapshot insert
    assert len(fake_db.bars) == 2
    assert len(fake_db.snapshots) == 1

