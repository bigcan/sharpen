"""NOW-7 (audit C9-05): wall-clock-honest per-tick proposal timestamps.

The old loop stamped tick_ts = start + i days, so a multi-night burst run in one process fabricated
FUTURE nights (07-03..07-05 while executing on 07-02) — future-dating CR-2 proposal_ts and pushing the
lockbox forward boundary out. The fix: default to the actual UTC instant per tick; a pinned --start-ts
(reproduce) steps +1d but caps at wall-clock now and round-trips a single past ts verbatim.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.research.crucible_orchestrator import _tick_timestamps


def test_default_is_wall_clock_now_never_future() -> None:
    ts = _tick_timestamps(None, 4)
    parsed = [datetime.fromisoformat(t) for t in ts]
    now = datetime.now(timezone.utc)
    assert all(p.tzinfo is not None for p in parsed)                 # tz-aware UTC
    assert all(p <= now + timedelta(seconds=2) for p in parsed)     # never future
    assert (max(parsed) - min(parsed)) < timedelta(seconds=5)       # NOT +1d stepped


def test_start_ts_steps_past_but_caps_at_now() -> None:
    ts = _tick_timestamps("2020-01-01T00:00:00+00:00", 4)           # all in the past
    assert len(set(ts)) == 4 and ts[0] == "2020-01-01T00:00:00+00:00"
    now = datetime.now(timezone.utc)
    ts2 = _tick_timestamps((now - timedelta(days=1)).isoformat(), 5)  # crosses now on later nights
    assert all(datetime.fromisoformat(t) <= now + timedelta(seconds=2) for t in ts2)   # clamped


def test_reproduce_recipe_bit_identical() -> None:
    recorded = "2026-07-02T23:53:17.374661+00:00"                   # a recipe's pinned --start-ts
    assert _tick_timestamps(recorded, 1)[0] == recorded            # exact string round-trip
