"""Tests for scripts/watchdog_docker ibgateway-rebind hook (Fix D, S535-cont-2).

Regression for the gmgp1-gold daily 65-min outage caused by ibgateway's
TWS nightly auto-logoff at 23:40 UTC restarting the container, which
orphans dependents' kernel netns reference (network_mode: container:<id>).

The hook detects ibgateway start events and, after a short grace period
to let IBC complete login, restarts each dependent strategy container so
its netns reference rebinds to the new ibgateway PID's namespace.

See project_gmgp1_gold_daily_ib_outage_s535.md and
project_ibgateway_restart_orphan_netns.md.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


@pytest.fixture
def wd(monkeypatch):
    """Fresh watchdog_docker module per test with Timer patched to fire synchronously."""
    sys.modules.pop("watchdog_docker", None)
    mod = importlib.import_module("watchdog_docker")

    # Reset module state
    mod._last_ibgateway_rebind_fired = 0.0

    # Patch alert + outbound calls
    monkeypatch.setattr(mod, "alert", lambda name, msg: None)

    # Patch Timer to invoke immediately (no real wait, no thread)
    fire_log: list[float] = []

    class _SyncTimer:
        def __init__(self, interval, function, *args, **kwargs):
            fire_log.append(interval)
            self._function = function
            self._args = args
            self._kwargs = kwargs
            self.daemon = True

        def start(self):
            self._function(*self._args, **self._kwargs)

    monkeypatch.setattr(mod, "Timer", _SyncTimer)
    mod._fire_log = fire_log

    monkeypatch.setattr(mod, "IBGATEWAY_REBIND_ENABLED", True)
    monkeypatch.setattr(mod, "IBGATEWAY_REBIND_GRACE_S", 20)
    monkeypatch.setattr(mod, "IBGATEWAY_CONTAINER_NAME", "ibgateway")

    yield mod


def _ibgw_start_event(name: str = "ibgateway") -> dict:
    return {
        "Action": "start",
        "Actor": {
            "ID": "deadbeef" * 4,
            "Attributes": {"name": name},
        },
    }


def test_hook_fires_on_ibgateway_start(wd, monkeypatch):
    """Start event for ibgateway schedules a dependent rebind."""
    rebind_calls = []
    monkeypatch.setattr(
        wd,
        "_ibgateway_rebind_dependents",
        lambda: rebind_calls.append(True),
    )

    wd.handle_container_event(_ibgw_start_event())

    assert wd._fire_log == [20], "Timer scheduled with the configured grace period"
    assert rebind_calls == [True], "Rebind function executed (synchronously via patched Timer)"


def test_hook_dedupes_rapid_repeats(wd, monkeypatch):
    """Multiple ibgateway start events within the dedupe window fire only once."""
    rebind_calls = []
    monkeypatch.setattr(
        wd,
        "_ibgateway_rebind_dependents",
        lambda: rebind_calls.append(True),
    )

    wd.handle_container_event(_ibgw_start_event())
    wd.handle_container_event(_ibgw_start_event())
    wd.handle_container_event(_ibgw_start_event())

    assert len(rebind_calls) == 1, "Only the first event in the dedupe window fires"


def test_hook_skipped_when_disabled(wd, monkeypatch):
    """IBGATEWAY_REBIND_ENABLED=False suppresses the hook entirely."""
    rebind_calls = []
    monkeypatch.setattr(wd, "IBGATEWAY_REBIND_ENABLED", False)
    monkeypatch.setattr(
        wd,
        "_ibgateway_rebind_dependents",
        lambda: rebind_calls.append(True),
    )

    wd.handle_container_event(_ibgw_start_event())

    assert rebind_calls == [], "Disabled hook does not invoke rebind"
    assert wd._fire_log == [], "No Timer scheduled when disabled"


def test_hook_only_on_target_container_name(wd, monkeypatch):
    """Start events for non-ibgateway containers without finrl.monitor are no-ops here."""
    rebind_calls = []
    monkeypatch.setattr(
        wd,
        "_ibgateway_rebind_dependents",
        lambda: rebind_calls.append(True),
    )

    # Some random other container starting up (no finrl.monitor label) — should
    # fall through to the existing early-return at the `finrl.monitor` check.
    wd.handle_container_event(
        {
            "Action": "start",
            "Actor": {
                "ID": "abc" * 13,
                "Attributes": {"name": "prometheus"},
            },
        }
    )

    assert rebind_calls == [], "Non-ibgateway start does not fire the rebind"


def test_find_dependents_matches_network_mode_by_id(wd):
    """_find_ibgateway_dependents returns containers using container:<ibgateway-id>."""
    fake_client = MagicMock()
    ibg = MagicMock()
    ibg.id = "deadbeef" * 8
    fake_client.containers.get.return_value = ibg

    matching_dep = MagicMock()
    matching_dep.name = "gmgp1-gold"
    matching_dep.attrs = {"HostConfig": {"NetworkMode": f"container:{ibg.id}"}}

    matching_by_name_dep = MagicMock()
    matching_by_name_dep.name = "alt-ib-strategy"
    matching_by_name_dep.attrs = {"HostConfig": {"NetworkMode": "container:ibgateway"}}

    unrelated_dep = MagicMock()
    unrelated_dep.name = "gmgp1-xauusd"
    unrelated_dep.attrs = {"HostConfig": {"NetworkMode": "live_finrl-net"}}

    fake_client.containers.list.return_value = [matching_dep, unrelated_dep, matching_by_name_dep]

    deps = wd._find_ibgateway_dependents(fake_client)
    names = sorted(c.name for c in deps)
    assert names == ["alt-ib-strategy", "gmgp1-gold"], (
        "Dependents matched by exact NetworkMode (container:<id> or container:ibgateway); "
        "containers on a docker bridge / named network are not dependents"
    )


def test_find_dependents_handles_lookup_failure(wd):
    """If looking up ibgateway raises (not found / API error / etc.), return [] gracefully.

    The docker SDK isn't installed in the dev .venv (it's container-only), so
    we use a generic Exception which the function's broad `except Exception`
    branch catches identically to the more specific `docker.errors.NotFound`.
    """
    fake_client = MagicMock()
    fake_client.containers.get.side_effect = Exception("not found")

    deps = wd._find_ibgateway_dependents(fake_client)
    assert deps == [], "Lookup failure returns empty list, not exception"
