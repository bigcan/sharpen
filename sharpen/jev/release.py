"""LEAK-2 release row for SEC filings (ATL x Jev pre-registration §3; Phase 2 architecture ADR-2).

A filing EDGAR accepted at τ (UTC) may first move a position at the CLOSE of its release row: the trading day τ
falls on, if τ is before the cutoff in exchange-local time, otherwise the next trading day. The funnel pairs row t
with ``close[t+h]/close[t] - 1``, so a score used at row t must be knowable in time to trade t's close; the cutoff
(15:30 America/New_York, frozen in ``configs/atl_jev.gates.yaml`` ``phase1.release_row``) keeps a margin before the
16:00 close.

Local time comes from ``zoneinfo`` and is DST-aware. A fixed UTC-5 offset would put a July filing accepted at
19:45Z (15:45 EDT, after the cutoff) on the SAME day — the look-ahead ``tests/jev/test_release.py`` pins.

The calendar is the price panel's own row dates (trading days only), so no external exchange calendar is needed.
A filing whose release day would fall after the calendar's last day maps to NaT: fail closed, never a guess.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")


@dataclass(frozen=True)
class ReleaseRule:
    timezone: str        # IANA zone name, kept for provenance
    cutoff: time         # accepted strictly before this local time on a trading day -> that day's row
    tz: ZoneInfo


def load_release_rule(phase1: Mapping) -> ReleaseRule:
    """Read ``phase1.release_row`` from the gates file. Refuses a missing key, an unknown zone or a malformed
    cutoff — there is no default, because a wrong cutoff is a silent look-ahead."""
    try:
        rr = phase1["release_row"]
        zone, cutoff = str(rr["timezone"]), str(rr["cutoff"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"phase1.release_row.{{timezone, cutoff}} required: {exc!r}") from exc
    m = _HHMM.fullmatch(cutoff)
    if m is None:
        raise ValueError(f"phase1.release_row.cutoff must be 'HH:MM' (24h), got {cutoff!r}")
    try:
        tz = ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {zone!r}") from exc
    return ReleaseRule(zone, time(int(m.group(1)), int(m.group(2))), tz)


def release_dates(accepted_utc: np.ndarray, calendar: np.ndarray, rule: ReleaseRule) -> np.ndarray:
    """(K,) ``datetime64[D]`` release day per filing, or NaT.

    ``accepted_utc``: (K,) datetime64, UTC, naive (EDGAR ``acceptanceDateTime``). ``calendar``: trading days,
    strictly increasing. Release day = the first calendar day ``d`` with ``d == local_date(τ)`` and
    ``local_time(τ) < cutoff``, or else ``d > local_date(τ)``.
    """
    acc = np.asarray(accepted_utc, dtype="datetime64[ns]")
    cal = np.asarray(calendar, dtype="datetime64[D]")
    if cal.ndim != 1 or (cal.size > 1 and not np.all(np.diff(cal) > np.timedelta64(0, "D"))):
        raise ValueError("calendar must be a strictly increasing 1-D array of trading days")
    out = np.full(acc.shape, np.datetime64("NaT"), dtype="datetime64[D]")
    for k, a in enumerate(acc):
        if np.isnat(a):
            continue
        local = a.astype("datetime64[us]").astype(datetime).replace(tzinfo=timezone.utc).astimezone(rule.tz)
        day = np.datetime64(local.date(), "D")
        i = int(np.searchsorted(cal, day, side="left"))
        if i < cal.size and cal[i] == day and local.time() < rule.cutoff:
            out[k] = cal[i]                       # a trading day, and in time for its close
        else:
            j = int(np.searchsorted(cal, day, side="right"))
            if j < cal.size:
                out[k] = cal[j]                   # the next trading day strictly after the local date
    return out
