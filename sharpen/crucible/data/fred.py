"""FRED / ALFRED connector (spec §4.2 Tier-A macro) — Crucible P1b.

FRED is vast, reliable and free. **PIT caveat (C1-03):** the DEFAULT fetch path is a RELEASE-LAG
model, NOT a true vintage read — it pulls the latest published observations and stamps each
``release_timestamp`` as ``reference_period + release_lag_days``. That is PIT-safe ONLY for series
effectively never revised after first release (the curated default list is chosen to be such); for a
genuinely revised series it would present today's revised value as if it were public at first release.
True **ALFRED** vintages (``realtime_start`` per observation; ``output_type=2`` for the full revision
history that :func:`quality_gate.asof_join` replays) are the correct fix for revised series and the
``as_of`` param is plumbed for it — but that path is NOT the default and still has a known stamping gap
for revised values (C1-05). Do not rely on FRED for a genuinely revised series until ALFRED vintages
are wired end-to-end (roadmap NEXT-9); prefer non-revised series (rates, spreads) until then.

Testability: pass a ``transport`` callable ``(url) -> dict`` (parsed JSON) and the connector needs
NO network and NO key — fixtures drive it. The live path requires ``FRED_API_KEY`` (free) and
outbound HTTPS to ``api.stlouisfed.org``; it fails closed with a clear error otherwise.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from collections.abc import Callable

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred"
_LICENSE = "FRED/ALFRED — free, attribution to Federal Reserve Bank of St. Louis"
# A small curated Tier-A macro starter set (spec §10 decision 3). discover() returns these unless a
# custom list is supplied; the agent's Data Scout (P5) proposes additions.
_DEFAULT_SERIES: tuple[tuple[str, str, str], ...] = (
    ("T10Y2Y", "10Y-2Y Treasury spread (term structure)", "D"),
    ("DGS10", "10Y Treasury constant maturity yield", "D"),
    ("BAMLH0A0HYM2", "ICE BofA US High Yield OAS (credit stress)", "D"),
    ("T10YIE", "10Y breakeven inflation", "D"),
    ("VIXCLS", "CBOE VIX", "D"),
    ("WALCL", "Fed total assets (liquidity)", "W"),
)


def _default_transport(url: str) -> dict:
    """Live HTTPS GET → parsed JSON. Only reached when no ``transport`` is injected."""
    with urllib.request.urlopen(url, timeout=30) as resp:   # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


class FredConnector:
    """DataConnector for FRED/ALFRED. ``asset_class = 'macro'``."""

    source_id = "fred"
    asset_class = "macro"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: Callable[[str], dict] | None = None,
        series: tuple[tuple[str, str, str], ...] | None = None,
        release_lag_days: int = 1,
    ) -> None:
        self._api_key: str = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")
        self._transport = transport
        self._series = series or _DEFAULT_SERIES
        # Publication lag: FRED's plain endpoint stamps realtime_start = *today* (useless as a
        # release time), and the all-vintages endpoint (output_type=2) is a pivoted format capped
        # at 2000 vintages — so we model the release as reference_period + lag, exactly like COT.
        # Default 1d suits the non-revised daily/weekly series (Treasury rates, spreads, VIX). For
        # heavily-revised monthly series (GDP, payrolls) pass a larger lag or use the `as_of`
        # vintage path — reference+lag never leaks (over-lagging is safe), it can only be stale.
        self._release_lag = np.timedelta64(release_lag_days, "D")

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, sid, self.asset_class, title=title, frequency=freq)
            for sid, title, freq in self._series
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        return Provenance(
            source_id=self.source_id,
            url=f"https://fred.stlouisfed.org/series/{ref.series_id}",
            license=_LICENSE,
            # Default stamping is release-lag; a true ALFRED vintage snapshot is available via
            # fetch(..., as_of=<date>), which pins realtime_start/end to that date (CR-4).
            as_of_policy="release-lag",
            release_lag_days=int(self._release_lag / np.timedelta64(1, "D")),
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
        payload = self._request(ref, start, end, as_of=as_of)
        return self._parse(ref, payload, as_of=as_of)

    # -- internals -----------------------------------------------------------------
    def _request(self, ref: SeriesRef, start, end, *, as_of) -> dict:
        params = {
            "series_id": ref.series_id,
            "file_type": "json",
            "observation_start": str(np.datetime64(start, "D")),
            "observation_end": str(np.datetime64(end, "D")),
        }
        if as_of is not None:
            # ALFRED vintage snapshot: the series values exactly as they stood on `as_of`.
            aod = str(np.datetime64(as_of, "D"))
            params["realtime_start"] = aod
            params["realtime_end"] = aod
        transport = self._transport or self._live_transport()
        query = dict(params)
        query["api_key"] = self._api_key
        url = f"{_BASE}/series/observations?{urllib.parse.urlencode(query)}"
        return transport(url)

    def _live_transport(self) -> Callable[[str], dict]:
        if not self._api_key:
            raise RuntimeError(
                "FredConnector live fetch requires FRED_API_KEY (free: fredaccount.stlouisfed.org). "
                "For offline/test use, inject a `transport` callable instead.")
        return _default_transport

    def _parse(self, ref: SeriesRef, payload: dict, *, as_of) -> SeriesData:
        obs = payload.get("observations", [])
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None
        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for o in obs:
            raw = o.get("value", ".")
            if raw in (".", "", None):          # FRED marks missing as "."
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            reference = np.datetime64(o["date"], "ns")
            release = reference + self._release_lag        # publication lag (see __init__)
            if cutoff is not None and release > cutoff:    # vintage: only what was public by as_of
                continue
            refs.append(reference)
            vals.append(v)
            rels.append(release)
        return SeriesData(
            ref=ref,
            reference_period=np.array(refs, dtype="datetime64[ns]"),
            value=np.array(vals, dtype=np.float64),
            release_timestamp=np.array(rels, dtype="datetime64[ns]"),
            provenance=self.provenance(ref),
            meta={"n_raw": len(obs)},
        )
