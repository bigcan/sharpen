from __future__ import annotations

import pandas as pd

from finrl_pro.data.loader import DataLoader


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def load_snapshot(self, snapshot_id: str):
        return self._rows


def test_resolve_snapshot_returns_dataframe():
    rows = [
        {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "ticker": "SPY",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 100.0,
            "source": "yahoo",
            "vendor_rev": 1,
        },
        {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "ticker": "AAPL",
            "open": 2.0,
            "high": 2.2,
            "low": 1.8,
            "close": 2.1,
            "volume": 200.0,
            "source": "yahoo",
            "vendor_rev": 1,
        },
    ]
    df = DataLoader.resolve_dataset("snapshot://abc", client=_FakeDB(rows))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)[:3] == ["timestamp", "ticker", "open"]
    assert len(df) == 2


def test_resolve_snapshot_empty_raises():
    try:
        DataLoader.resolve_dataset("snapshot://empty", client=_FakeDB([]))
    except ValueError as e:
        assert "No data found for snapshot" in str(e)
    else:
        assert False, "Expected ValueError for empty snapshot"

