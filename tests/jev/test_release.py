"""LEAK-2 release-row tripwires (pre-registration §3). Every case that matters is a NEGATIVE test: it fails if a
filing can reach a row before it is usable at that row's close — including the DST case a fixed UTC offset gets
wrong."""
from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml

from sharpen.jev.release import load_release_rule, release_dates

# Trading days around: MLK holiday (Mon 2024-01-15), US DST start (Sun 2024-03-10), July 4 (Thu 2024-07-04),
# US DST end (Sun 2024-11-03).
CAL = np.array(["2024-01-12", "2024-01-16", "2024-03-08", "2024-03-11", "2024-03-12", "2024-07-03",
                "2024-07-05", "2024-07-08", "2024-11-01", "2024-11-04"], dtype="datetime64[D]")
PHASE1 = {"release_row": {"timezone": "America/New_York", "cutoff": "15:30"}}
RULE = load_release_rule(PHASE1)


def _rd(utc: str, rule=RULE) -> str:
    out = release_dates(np.array([utc], dtype="datetime64[ns]"), CAL, rule)[0]
    return "NaT" if np.isnat(out) else str(out)


@pytest.mark.parametrize("utc, expect, why", [
    ("2024-07-03T19:29:00", "2024-07-03", "15:29 EDT: before the cutoff -> same day"),
    ("2024-07-03T19:30:00", "2024-07-05", "exactly 15:30:00 EDT is not BEFORE the cutoff -> next day"),
    ("2024-07-03T19:31:00", "2024-07-05", "15:31 EDT: after the cutoff -> next trading day (skips July 4)"),
    ("2024-07-03T20:30:00", "2024-07-05", "16:30 EDT: after the close"),
    ("2024-03-08T21:00:00", "2024-03-11", "Friday 16:00 EST -> Monday"),
    ("2024-03-09T15:00:00", "2024-03-11", "Saturday -> Monday"),
    ("2024-01-15T14:00:00", "2024-01-16", "MLK holiday -> next trading day"),
    ("2024-03-12T11:00:00", "2024-03-12", "07:00 EDT pre-open -> same day (usable at its close)"),
])
def test_release_day(utc, expect, why):
    assert _rd(utc) == expect, why


def test_dst_uses_the_local_offset_not_a_fixed_one():
    """19:45Z is 14:45 EST in January (before the cutoff) but 15:45 EDT in July (after it). A fixed UTC-5
    conversion would put the July filing on the same day — a look-ahead of one full session."""
    assert _rd("2024-01-12T19:45:00") == "2024-01-12"
    assert _rd("2024-07-05T19:45:00") == "2024-07-08"
    # DST end: 20:15Z on 2024-11-04 is 15:15 EST (UTC-5) -> same day; a stale UTC-4 would say 16:15 -> next
    assert _rd("2024-11-04T20:15:00") == "2024-11-04"


def test_beyond_the_calendar_fails_closed():
    assert _rd("2024-11-04T21:00:00") == "NaT"          # 16:00 EST after cutoff; no later day in the calendar
    assert np.isnat(release_dates(np.array(["NaT"], dtype="datetime64[ns]"), CAL, RULE)[0])


def test_invariant_on_random_times_never_same_day_after_cutoff():
    """Property: a release day is a calendar day at or after the local date, and equals the local date ONLY
    when the local time is before the cutoff."""
    rng = np.random.default_rng(7)
    lo, hi = np.datetime64("2024-01-10T00:00", "s").astype(np.int64), np.datetime64("2024-11-04T12:00", "s").astype(np.int64)
    acc = rng.integers(lo, hi, size=3000).astype("datetime64[s]").astype("datetime64[ns]")
    out = release_dates(acc, CAL, RULE)
    tz = ZoneInfo("America/New_York")
    for a, d in zip(acc, out):
        if np.isnat(d):
            continue
        local = a.astype("datetime64[us]").astype(datetime).replace(tzinfo=timezone.utc).astimezone(tz)
        ld = np.datetime64(local.date(), "D")
        assert d in CAL and d >= ld
        if d == ld:
            assert local.time() < time(15, 30)


def test_the_cutoff_is_read_from_the_gates_not_hardcoded():
    later = load_release_rule({"release_row": {"timezone": "America/New_York", "cutoff": "16:00"}})
    assert _rd("2024-07-03T19:31:00", later) == "2024-07-03"      # 15:31 is now before the (moved) cutoff


@pytest.mark.parametrize("phase1", [
    {},
    {"release_row": {"timezone": "America/New_York"}},
    {"release_row": {"timezone": "Mars/Olympus", "cutoff": "15:30"}},
    {"release_row": {"timezone": "America/New_York", "cutoff": "3:30pm"}},
    {"release_row": {"timezone": "America/New_York", "cutoff": "25:00"}},
])
def test_loader_refuses_anything_it_cannot_honour(phase1):
    with pytest.raises(ValueError):
        load_release_rule(phase1)


def test_calendar_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        release_dates(np.array(["2024-07-03T12:00"], dtype="datetime64[ns]"), CAL[::-1], RULE)
    with pytest.raises(ValueError):
        release_dates(np.array(["2024-07-03T12:00"], dtype="datetime64[ns]"), np.r_[CAL, CAL[-1:]], RULE)


def test_frozen_gates_file_gives_15_30_new_york():
    g = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "atl_jev.gates.yaml").read_text(encoding="utf-8"))
    rule = load_release_rule(g["phase1"])
    assert rule.timezone == "America/New_York" and rule.cutoff == time(15, 30)
