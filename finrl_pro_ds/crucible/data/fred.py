"""FRED / ALFRED connector (spec §4.2 Tier-A macro) — Crucible P1b.

FRED is vast, reliable and free; **ALFRED** is the reason it is PIT-safe: it exposes true *vintage*
series so a backtest never sees a revised value early (CR-4). This connector therefore reads the
vintage-aware endpoint and stamps each observation's ``release_timestamp`` from ALFRED's
``realtime_start`` (the date that reading first became the published value). Requesting all vintages
(``output_type=2``) yields the full release history incl. revisions, which :func:`quality_gate.asof_join`
replays correctly.

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
    ) -> None:
        self._api_key: str = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")
        self._transport = transport
        self._series = series or _DEFAULT_SERIES

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
            as_of_policy="vintage-api",       # ALFRED vintages → no revision look-ahead (CR-4)
            release_lag_days=0,               # per-observation release stamped from realtime_start
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
        return self._parse(ref, payload)

    # -- internals -----------------------------------------------------------------
    def _request(self, ref: SeriesRef, start, end, *, as_of) -> dict:
        params = {
            "series_id": ref.series_id,
            "file_type": "json",
            "observation_start": str(np.datetime64(start, "D")),
            "observation_end": str(np.datetime64(end, "D")),
        }
        if as_of is not None:
            # ALFRED vintage snapshot: the series exactly as it stood on `as_of`.
            aod = str(np.datetime64(as_of, "D"))
            params["realtime_start"] = aod
            params["realtime_end"] = aod
        else:
            # All vintages → full release history (each row carries its own realtime_start).
            params["realtime_start"] = "1776-07-04"
            params["realtime_end"] = "9999-12-31"
            params["output_type"] = "2"
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

    @staticmethod
    def _parse(ref: SeriesRef, payload: dict) -> SeriesData:
        obs = payload.get("observations", [])
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
            refs.append(np.datetime64(o["date"], "ns"))
            vals.append(v)
            # realtime_start = the date this reading became the published value (ALFRED release ts).
            rels.append(np.datetime64(o.get("realtime_start", o["date"]), "ns"))
        return SeriesData(
            ref=ref,
            reference_period=np.array(refs, dtype="datetime64[ns]"),
            value=np.array(vals, dtype=np.float64),
            release_timestamp=np.array(rels, dtype="datetime64[ns]"),
            provenance=Provenance(
                source_id=ref.source_id,
                url=f"https://fred.stlouisfed.org/series/{ref.series_id}",
                license=_LICENSE,
                as_of_policy="vintage-api",
                revision_policy="revised",
            ),
            meta={"n_raw": len(obs)},
        )
