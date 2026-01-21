from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import finrl_pro_ds.data.snapshot as snapshot_cli
import finrl_pro_ds.data.export_snapshot as export_cli
from finrl_pro_ds.data.loader import DataLoader


class _FakeDB:
    def __init__(self):
        self._bars = []
        self._snapshots = {}

    # Snapshot CLI calls
    def upsert_bars(self, bars):
        self._bars.extend(bars)
        return len(bars)

    def insert_snapshot(self, *, snapshot_id: str, provider: str, params_json: str, code_hash: str, lib_versions_json: str, tickers, row_count: int):
        self._snapshots[snapshot_id] = {
            "tickers": list(tickers),
            "row_count": row_count,
        }

    # Loader uses this
    def load_snapshot(self, snapshot_id: str):
        # Turn stored bars into dicts
        rows = []
        for b in self._bars:
            rows.append(b.__dict__ if hasattr(b, "__dict__") else dict(b))
        return rows

    # Export CLI calls this, emulate writing a file
    def export_snapshot(self, snapshot_id: str, *, fmt: str, out_path: str) -> str:
        df = pd.DataFrame(self.load_snapshot(snapshot_id))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(out_path)
        elif fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError("unknown format")
        return out_path


class _FakeLoader:
    def fetch(self, *, tickers: list[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        return pd.DataFrame([
            {"timestamp": pd.Timestamp("2020-01-01"), "ticker": t, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 100.0}
            for t in tickers
        ])


def test_snapshot_export_and_resolve(tmp_path, monkeypatch, capsys):
    fake_db = _FakeDB()
    monkeypatch.setattr(snapshot_cli, "DatabaseClient", lambda dsn=None: fake_db)
    monkeypatch.setattr(snapshot_cli, "YahooLoader", lambda: _FakeLoader())

    # 1) Create a snapshot via CLI
    args = [
        "--provider", "yahoo",
        "--tickers", "SPY,AAPL",
        "--start", "2020-01-01",
        "--end", "2020-01-02",
    ]
    snapshot_cli.main(args)
    out = json.loads(capsys.readouterr().out)
    snap_id = out["snapshot_id"]

    # 2) Export snapshot via CLI
    monkeypatch.setattr(export_cli, "DatabaseClient", lambda dsn=None: fake_db)
    export_cli.main(["--id", snap_id, "--out", str(tmp_path), "--format", "csv"])
    exported = list(tmp_path.glob(f"{snap_id}.csv"))
    assert exported and exported[0].exists()

    # 3) Resolve via loader using same fake DB
    df = DataLoader.resolve_dataset(f"snapshot://{snap_id}", client=fake_db)
    assert len(df) == 2
    assert set(df["ticker"]) == {"SPY", "AAPL"}

