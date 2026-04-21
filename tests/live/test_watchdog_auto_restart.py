"""Auto-restart tests for scripts/watchdog_docker.sweep.

Regression for S489 (2026-04-21, issue #4): watchdog previously only
restarted containers on `not running`. For `running / unhealthy` it
alert-looped without action — gmgp1-gold received 24 Telegram UNHEALTHY
pings over 2h10m with no recovery.

Fix: after N consecutive unhealthy sweeps on a non-suppressed strategy,
`docker restart <name>` is issued (rate-limited to avoid loops).

Tests verify:
  1. No restart before the threshold is reached.
  2. Restart fires on the Nth consecutive unhealthy sweep.
  3. Streak resets when container returns to healthy.
  4. Streak does NOT advance during TradFi weekend suppression.
  5. Per-hour rate limit prevents runaway restart loops.
  6. AUTO_RESTART_ENABLED=false disables the feature entirely.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# scripts/ is not a package — add it to sys.path so we can import watchdog_docker.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


@pytest.fixture
def wd(monkeypatch):
    """Import watchdog_docker with a fresh module state per test.

    Patches `alert` and `get_health_detail` to avoid Telegram / docker calls.
    Patches `_tradfi_market_closed_utc` to default False (market open).
    Clears streak + restart history.
    """
    # Force re-import so module-level dicts are empty each test
    sys.modules.pop("watchdog_docker", None)
    mod = importlib.import_module("watchdog_docker")

    # Reset module state
    mod._unhealthy_streak.clear()
    mod._auto_restart_history.clear()

    # Patch alert + detail fetch to no-op
    monkeypatch.setattr(mod, "alert", lambda name, msg: None)
    monkeypatch.setattr(mod, "get_health_detail", lambda c: "detail")
    monkeypatch.setattr(mod, "_tradfi_market_closed_utc", lambda: False)

    # Enforce deterministic config
    monkeypatch.setattr(mod, "AUTO_RESTART_ENABLED", True)
    monkeypatch.setattr(mod, "AUTO_RESTART_AFTER_N", 3)
    monkeypatch.setattr(mod, "AUTO_RESTART_MAX_PER_HOUR", 3)

    yield mod


def _make_container(name: str, health: str, running: bool = True) -> MagicMock:
    """Mock docker container + the info dict inspect_container would produce."""
    c = MagicMock(name=f"container-{name}")
    c._info = {
        "name": name,
        "strategy": name,
        "status": "running" if running else "exited",
        "health": health,
        "running": running,
    }
    return c


def _make_client(containers: list[MagicMock]) -> MagicMock:
    """Mock docker client whose containers.list returns the given set."""
    client = MagicMock()
    client.containers.list.return_value = containers
    return client


def _patch_inspect(wd, containers: list[MagicMock]):
    """Route inspect_container to each container's bundled info dict."""
    wd.inspect_container = lambda c: c._info  # type: ignore[assignment]


def _run_sweeps(wd, containers: list[MagicMock], n: int) -> None:
    client = _make_client(containers)
    _patch_inspect(wd, containers)
    for _ in range(n):
        wd.sweep(client)


def test_no_restart_before_threshold(wd):
    c = _make_container("gmgp1-gold", health="unhealthy")
    _run_sweeps(wd, [c], n=2)
    assert c.restart.call_count == 0
    assert wd._unhealthy_streak["gmgp1-gold"] == 2


def test_restart_fires_on_nth_unhealthy_sweep(wd):
    c = _make_container("gmgp1-gold", health="unhealthy")
    _run_sweeps(wd, [c], n=3)
    assert c.restart.call_count == 1
    # Streak reset after restart so we don't double-fire next sweep
    assert "gmgp1-gold" not in wd._unhealthy_streak
    assert len(wd._auto_restart_history["gmgp1-gold"]) == 1


def test_streak_resets_when_container_returns_to_healthy(wd):
    c = _make_container("gmgp1-gold", health="unhealthy")
    _run_sweeps(wd, [c], n=2)
    assert wd._unhealthy_streak["gmgp1-gold"] == 2

    c._info["health"] = "healthy"
    _run_sweeps(wd, [c], n=1)
    assert "gmgp1-gold" not in wd._unhealthy_streak
    assert c.restart.call_count == 0


def test_streak_does_not_advance_during_tradfi_weekend(wd, monkeypatch):
    monkeypatch.setattr(wd, "_tradfi_market_closed_utc", lambda: True)
    c = _make_container("gmgp1-gold", health="unhealthy")
    _run_sweeps(wd, [c], n=5)
    assert "gmgp1-gold" not in wd._unhealthy_streak
    assert c.restart.call_count == 0


def test_rate_limit_blocks_fourth_restart_within_hour(wd):
    c = _make_container("gmgp1-gold", health="unhealthy")
    # Three successful auto-restarts — each takes 3 sweeps to trigger
    _run_sweeps(wd, [c], n=3)
    _run_sweeps(wd, [c], n=3)
    _run_sweeps(wd, [c], n=3)
    assert c.restart.call_count == 3

    # Fourth attempt — rate limit must block restart; streak keeps advancing
    _run_sweeps(wd, [c], n=3)
    assert c.restart.call_count == 3
    assert wd._unhealthy_streak["gmgp1-gold"] == 3


def test_rate_limit_clears_after_one_hour(wd):
    c = _make_container("gmgp1-gold", health="unhealthy")
    # Back-date three restart timestamps to >1h ago
    wd._auto_restart_history["gmgp1-gold"] = [1000.0, 1001.0, 1002.0]
    # Current time is much later — should allow a fresh restart
    _run_sweeps(wd, [c], n=3)
    assert c.restart.call_count == 1


def test_disabled_flag_prevents_any_restart(wd, monkeypatch):
    monkeypatch.setattr(wd, "AUTO_RESTART_ENABLED", False)
    c = _make_container("gmgp1-gold", health="unhealthy")
    _run_sweeps(wd, [c], n=10)
    assert c.restart.call_count == 0
    # Streak still tracked for observability even when restart is disabled
    assert wd._unhealthy_streak["gmgp1-gold"] == 10


def test_not_running_container_clears_streak(wd):
    """Container that dies mid-streak: docker's restart policy handles it,
    watchdog should clear the unhealthy streak so it doesn't double-act."""
    c = _make_container("gmgp1-gold", health="unhealthy")
    _run_sweeps(wd, [c], n=2)
    assert wd._unhealthy_streak["gmgp1-gold"] == 2

    c._info["running"] = False
    c._info["status"] = "exited"
    _run_sweeps(wd, [c], n=1)
    assert "gmgp1-gold" not in wd._unhealthy_streak


def test_restart_failure_does_not_record_success(wd):
    c = _make_container("gmgp1-gold", health="unhealthy")
    c.restart.side_effect = RuntimeError("docker API down")
    _run_sweeps(wd, [c], n=3)
    assert c.restart.call_count == 1
    # Failed restart not recorded — rate limit not consumed
    assert wd._auto_restart_history.get("gmgp1-gold", []) == []
