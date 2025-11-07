from __future__ import annotations

import json

import pandas as pd

import finrl_pro.data.snapshot as snap


class _FlakyLoader:
    def __init__(self):
        self.calls = 0

    def fetch(self, *, tickers, start, end, interval):
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("temporary failure")
        return pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01"), "ticker": tickers[0], "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
        ])


class _NoopDB:
    def upsert_bars(self, bars):
        return len(bars)

    def insert_snapshot(self, **kwargs):
        return None


def test_snapshot_cli_retries_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(snap, "YahooLoader", lambda: _FlakyLoader())
    monkeypatch.setattr(snap, "DatabaseClient", lambda dsn=None: _NoopDB())
    monkeypatch.setattr(snap.time, "sleep", lambda *_args, **_kwargs: None)

    snap.main([
        "--provider", "yahoo",
        "--tickers", "SPY",
        "--start", "2020-01-01",
        "--end", "2020-01-02",
    ])
    out = json.loads(capsys.readouterr().out)
    assert "snapshot_id" in out and out["row_count"] == 1

