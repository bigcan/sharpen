from __future__ import annotations

import json
import pandas as pd

import finrl_pro_ds.data.snapshot as snap


class _FakeAlpaca:
    def fetch(self, *, tickers, start, end, interval):
        return pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01"), "ticker": t, "open": 1, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 100}
            for t in tickers
        ])


class _FakeDB:
    def upsert_bars(self, bars):
        return len(bars)

    def insert_snapshot(self, **kwargs):
        return None


def test_snapshot_cli_alpaca_provider(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    monkeypatch.setattr(snap, "AlpacaLoader", lambda **_: _FakeAlpaca())
    monkeypatch.setattr(snap, "DatabaseClient", lambda dsn=None: _FakeDB())
    monkeypatch.setattr(snap.time, "sleep", lambda *_: None)

    snap.main([
        "--provider", "alpaca",
        "--tickers", "SPY,AAPL",
        "--start", "2020-01-01",
        "--end", "2020-01-02",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["row_count"] == 2 and out["snapshot_id"]

