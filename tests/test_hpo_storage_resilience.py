"""Tests for `_resolve_hpo_storage` — the Neon/serverless-Postgres survival fix.

An overnight run of six workers was lost because Optuna holds a DB connection idle for the
whole of a ~104-minute trial and the serverless backend terminates it, so the commit at
trial end raised `AdminShutdown` and Optuna turned that into `assert False, "Should not
reach."`, killing the process after the full 400K training steps had already run.

These assert the defences are configured. They capture the RDBStorage constructor kwargs
rather than building one, because RDBStorage connects (and creates tables) during __init__ —
a real instance would need a live database, which would make these tests useless in CI and
is beside the point: what is being verified is the CONFIGURATION we ask for.
"""
import pytest

import scripts.run_full_pipeline as rfp

PG = "postgresql+psycopg://u:p@example.invalid/db"


@pytest.fixture
def captured(monkeypatch):
    """Capture the kwargs `_resolve_hpo_storage` passes to RDBStorage."""
    seen = {}

    class _FakeRDBStorage:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(rfp.optuna.storages, "RDBStorage", _FakeRDBStorage)
    return seen


def test_non_postgres_passes_through_untouched():
    """sqlite default and None must be unchanged for every pre-existing workstream."""
    assert rfp._resolve_hpo_storage(None) is None
    assert rfp._resolve_hpo_storage("sqlite:///hpo.db") == "sqlite:///hpo.db"


def test_already_constructed_storage_passes_through():
    sentinel = object()
    assert rfp._resolve_hpo_storage(sentinel) is sentinel


def test_pool_pre_ping_enabled(captured):
    """THE core fix: revalidate a connection before use, so one killed while idle is
    transparently reopened instead of raising at commit time."""
    rfp._resolve_hpo_storage(PG)
    assert captured["engine_kwargs"]["pool_pre_ping"] is True


def test_pool_recycle_under_serverless_idle_cutoff(captured):
    rfp._resolve_hpo_storage(PG)
    recycle = captured["engine_kwargs"]["pool_recycle"]
    assert 0 < recycle <= 300, f"pool_recycle={recycle} too long for a serverless cutoff"


def test_heartbeat_configured(captured):
    """Keeps the serverless instance warm through a long trial, and lets a dead worker's
    trial be reclaimed instead of sitting RUNNING forever (6 such orphans were left behind
    by the failure this fixes)."""
    rfp._resolve_hpo_storage(PG)
    assert captured["heartbeat_interval"] > 0


def test_heartbeat_interval_shorter_than_grace_period(captured):
    """If grace <= interval, Optuna would reclaim trials that are still alive."""
    rfp._resolve_hpo_storage(PG)
    assert captured["heartbeat_interval"] < captured["grace_period"]


def test_failed_trials_are_retried(captured):
    rfp._resolve_hpo_storage(PG)
    assert captured["failed_trial_callback"] is not None


@pytest.mark.parametrize("url", ["postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"])
def test_both_postgres_url_forms_are_hardened(url, captured):
    """A bare `postgresql://` URL must not slip past unhardened."""
    rfp._resolve_hpo_storage(url)
    assert captured["engine_kwargs"]["pool_pre_ping"] is True
