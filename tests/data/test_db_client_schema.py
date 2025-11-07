from __future__ import annotations

from typing import List

import types

from finrl_pro.data.db import DatabaseClient


class _Cur:
    def __init__(self, executed: List[str]):
        self._executed = executed

    def execute(self, stmt: str, *args, **kwargs):
        # record the statement
        self._executed.append(stmt.strip())

    def executemany(self, stmt: str, seq):
        self._executed.append(stmt.strip())

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Conn:
    def __init__(self):
        self.executed: List[str] = []

    def cursor(self):
        return _Cur(self.executed)

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_init_schema_executes_core_statements(monkeypatch):
    fake_conn = _Conn()

    def _connect(_dsn: str):
        return fake_conn

    client = DatabaseClient(dsn="postgresql://test", connect=_connect)
    client.init_schema()

    joined = "\n".join(fake_conn.executed)
    assert "CREATE TABLE IF NOT EXISTS market_bars" in joined
    assert "CREATE TABLE IF NOT EXISTS snapshots" in joined
    assert "CREATE TABLE IF NOT EXISTS snapshot_assets" in joined

