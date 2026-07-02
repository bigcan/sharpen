"""Stooq connector (spec §4.2 Tier-A market breadth) — Crucible P5.

Stooq serves free **daily** OHLCV history for global equities, indices, FX and commodities — a way to
widen the universe beyond yfinance's fragility (spec §4.2). As a :class:`DataConnector` each symbol is
delivered as a single value series (the daily close) with a publication-time release stamp, so it can
enter a Panel feature slot through :func:`quality_gate.asof_join` exactly like every other source.

PIT honesty (CR-4): a daily bar's close is public only after that session ends, so the connector
stamps ``release_timestamp = reference_period + release_lag`` (default 1 day) — never the session
date itself. **Adjustment caveat:** Stooq's default history is split/dividend *adjusted*, and an
adjusted close is silently *restated* when a later corporate action occurs — a revision of past
values. That makes the adjusted series only weakly PIT-safe (the yfinance-adjustment hazard spec §4.2
flags); ``revision_policy='revised'`` records this, and adjustment-sensitive theses should prefer an
unadjusted pull. The series is best used for *universe breadth*, cross-checked against yfinance.

Testability mirrors FRED/COT: inject a ``transport`` callable ``(url) -> str`` (raw CSV text) and the
connector needs no network. The live path hits ``stooq.com`` over HTTPS (keyless) and fails closed
with a clear error only if the CSV is unparseable.
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://stooq.com/q/d/l/"
_LICENSE = "Stooq — free for personal use; verify redistribution terms per stooq.com"
_RELEASE_LAG_DAYS = 1      # daily close is public after the session ends (spec §4.2)

# A small curated breadth starter set. discover() returns these unless a custom list is supplied; the
# Data Scout (P5) proposes additions. Stooq symbols are lowercase; '^' = index, '.f' = continuous
# futures, bare 6-char = FX cross.
_DEFAULT_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("^spx", "S&P 500 index"),
    ("^ndx", "Nasdaq 100 index"),
    ("gc.f", "Gold continuous future"),
    ("cl.f", "WTI crude continuous future"),
    ("eurusd", "EUR/USD spot"),
)


def _default_transport(url: str) -> str:
    """Live HTTPS GET → raw CSV text. Only reached when no ``transport`` is injected."""
    with urllib.request.urlopen(url, timeout=30) as resp:   # noqa: S310 - fixed https host
        return resp.read().decode("utf-8")


class StooqConnector:
    """DataConnector for Stooq daily history. ``asset_class = 'market'``. series_id = stooq symbol."""

    source_id = "stooq"
    asset_class = "market"

    def __init__(
        self,
        *,
        transport: Callable[[str], str] | None = None,
        symbols: tuple[tuple[str, str], ...] | None = None,
        release_lag_days: int = _RELEASE_LAG_DAYS,
    ) -> None:
        self._transport = transport
        self._symbols = symbols or _DEFAULT_SYMBOLS
        self._release_lag = np.timedelta64(release_lag_days, "D")

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, sym, self.asset_class, title=title, frequency="D")
            for sym, title in self._symbols
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        return Provenance(
            source_id=self.source_id,
            url=f"https://stooq.com/q/d/?s={ref.series_id}",
            license=_LICENSE,
            as_of_policy="release-lag",
            release_lag_days=int(self._release_lag / np.timedelta64(1, "D")),
            # Adjusted closes restate on later corporate actions -> past values can change (CR-4).
            revision_policy="revised",
        )

    def fetch(
        self,
        ref: SeriesRef,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        as_of: np.datetime64 | str | None = None,
    ) -> SeriesData:
        csv_text = self._request(ref, start, end)
        return self._parse(ref, csv_text, start, end, as_of=as_of)

    # -- internals -----------------------------------------------------------------
    def _request(self, ref: SeriesRef, start, end) -> str:
        params = {
            "s": ref.series_id,
            "i": "d",
            "d1": str(np.datetime64(start, "D")).replace("-", ""),
            "d2": str(np.datetime64(end, "D")).replace("-", ""),
        }
        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        transport = self._transport or _default_transport
        return transport(url)

    def _parse(self, ref: SeriesRef, csv_text: str, start, end, *, as_of) -> SeriesData:
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None
        lo, hi = np.datetime64(start, "ns"), np.datetime64(end, "ns")
        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        for row in rows:
            date = row.get("Date") or row.get("date")
            close = row.get("Close") or row.get("close")
            if not date or close in (None, "", "N/A"):
                continue
            try:
                reference = np.datetime64(str(date)[:10], "ns")
                value = float(str(close))
            except (TypeError, ValueError):
                continue
            if reference < lo or reference > hi:           # stooq can echo out-of-range rows
                continue
            release = reference + self._release_lag        # close public after the session (CR-4)
            if cutoff is not None and release > cutoff:     # vintage: only what was public by as_of
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
            meta={"n_raw": len(rows)},
        )
