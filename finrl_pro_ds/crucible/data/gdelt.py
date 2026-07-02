"""GDELT 2.0 connector (spec §4.2 Tier-A news tone / sentiment) — Crucible P5.

GDELT is a free, deep-history global news database the current stack lacks entirely (spec §4.2). Its
DOC 2.0 API exposes a **daily average-tone timeline** per query — a genuine alt-data sentiment domain,
uncorrelated with price. As a :class:`DataConnector` each theme is one value series (the day's average
tone), released a day after the day it summarizes so it enters a Panel feature slot PIT-safe.

PIT honesty (CR-4): GDELT ingests continuously (every 15 min), so a full day's average tone is only
settled once the day closes. The connector stamps ``release_timestamp = reference_period + release_lag``
(default 1 day) — a bar on day ``t`` can only ever read tone through day ``t-1``. Gotchas spec §4.2
flags: tone ≠ causation; volume is heavy; timezone/dedup matters — so a theme MUST be a single lexer-
safe token (``gold``, ``inflation``), because the DSL terminal is ``gdelt:<theme>`` and the grammar
admits only one ``source:series`` segment with no spaces.

Testability mirrors FRED/COT: inject a ``transport`` callable ``(url) -> dict`` (parsed JSON) and no
network is needed. The live path hits ``api.gdeltproject.org`` over HTTPS (keyless).
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_LICENSE = "GDELT Project — free, open data (attribution requested)"
_RELEASE_LAG_DAYS = 1      # a day's average tone settles after the day closes (spec §4.2)

# Single-token themes (lexer-safe — see module docstring). discover() returns these unless a custom
# list is supplied; the Data Scout (P5) proposes additions. Each token is BOTH the series id and the
# GDELT query, so the emitted terminal (gdelt:<token>) never contains a space.
_DEFAULT_THEMES: tuple[tuple[str, str], ...] = (
    ("gold", "news tone for 'gold'"),
    ("inflation", "news tone for 'inflation'"),
    ("recession", "news tone for 'recession'"),
)


def _default_transport(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:   # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


class GdeltConnector:
    """DataConnector for GDELT DOC 2.0 tone timelines. ``asset_class = 'sentiment'``."""

    source_id = "gdelt"
    asset_class = "sentiment"

    def __init__(
        self,
        *,
        transport: Callable[[str], dict] | None = None,
        themes: tuple[tuple[str, str], ...] | None = None,
        release_lag_days: int = _RELEASE_LAG_DAYS,
    ) -> None:
        self._transport = transport
        self._themes = themes or _DEFAULT_THEMES
        self._release_lag = np.timedelta64(release_lag_days, "D")

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, theme, self.asset_class, title=title, frequency="D")
            for theme, title in self._themes
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        return Provenance(
            source_id=self.source_id,
            url=f"{_BASE}?query={urllib.parse.quote(ref.series_id)}&mode=timelinetone",
            license=_LICENSE,
            as_of_policy="release-lag",       # stamped reference_period + lag (CR-4)
            release_lag_days=int(self._release_lag / np.timedelta64(1, "D")),
            revision_policy="final",
        )

    def fetch(
        self,
        ref: SeriesRef,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        as_of: np.datetime64 | str | None = None,
    ) -> SeriesData:
        payload = self._request(ref, start, end)
        return self._parse(ref, payload, as_of=as_of)

    # -- internals -----------------------------------------------------------------
    def _request(self, ref: SeriesRef, start, end) -> dict:
        params = {
            "query": ref.series_id,
            "mode": "timelinetone",
            "format": "json",
            "startdatetime": _gdelt_stamp(start),
            "enddatetime": _gdelt_stamp(end),
        }
        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        transport = self._transport or _default_transport
        return transport(url)

    def _parse(self, ref: SeriesRef, payload: dict, *, as_of) -> SeriesData:
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None
        # GDELT returns {"timeline": [{"series": "...", "data": [{"date","value"}, ...]}, ...]}.
        # Take the first (Average Tone) series; be defensive about the shape.
        timeline = payload.get("timeline") or []
        data = timeline[0].get("data", []) if timeline else []
        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for point in data:
            raw_date = point.get("date")
            raw_val = point.get("value")
            if raw_date is None or raw_val is None:
                continue
            try:
                reference = _parse_gdelt_date(str(raw_date))
                value = float(raw_val)
            except (TypeError, ValueError):
                continue
            release = reference + self._release_lag         # tone settles after the day (CR-4)
            if cutoff is not None and release > cutoff:      # vintage: only what was public by as_of
                continue
            refs.append(reference)
            vals.append(value)
            rels.append(release)
        return SeriesData(
            ref=ref,
            reference_period=np.array(refs, dtype="datetime64[ns]"),
            value=np.array(vals, dtype=np.float64),
            release_timestamp=np.array(rels, dtype="datetime64[ns]"),
            provenance=self.provenance(ref),
            meta={"n_raw": len(data)},
        )


def _gdelt_stamp(when: np.datetime64 | str) -> str:
    """GDELT's ``YYYYMMDDHHMMSS`` datetime format (UTC), from a date-ish input."""
    d = np.datetime64(when, "D")
    return str(d).replace("-", "") + "000000"


def _parse_gdelt_date(raw: str) -> np.datetime64:
    """Parse a GDELT timeline date. Handles ``YYYYMMDDT......Z`` and ``YYYY-MM-DD``; the value is a
    DAILY tone so we normalize to the day (reference_period is period-end == that day)."""
    s = raw.strip()
    if "T" in s:                                            # e.g. 20200101T000000Z
        s = s.split("T", 1)[0]
    s = s.replace("-", "")
    if len(s) < 8:
        raise ValueError(f"unrecognized GDELT date {raw!r}")
    return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}", "ns")
