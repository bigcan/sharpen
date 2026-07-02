"""CFTC Commitments of Traders connector (spec §4.2 Tier-A positioning) — Crucible P1b.

Large-spec / commercial positioning is a classic uncrowded signal — big money's own footprint.
COT is public data (NO API key). The PIT crux (spec §4.2 gotcha, CR-4): the report describes
**Tuesday** positions but only publishes the following **Friday** — a ~3-day release lag. Joining
Tuesday's report to Tuesday's bar is look-ahead. This connector therefore stamps
``release_timestamp = report_date + release_lag`` so :func:`quality_gate.asof_join` binds it to the
first bar on/after the Friday release, never the Tuesday it describes.

Testability: inject a ``transport`` callable ``(url) -> list[dict]`` (parsed Socrata JSON) and it
needs no network. Live path hits ``publicreporting.cftc.gov`` over HTTPS (keyless; an optional free
Socrata app token only lifts rate limits).
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

# Legacy futures-only report (Socrata resource id). publicreporting.cftc.gov is keyless.
_RESOURCE = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_LICENSE = "CFTC Commitments of Traders — U.S. Government public data (no redistribution limit)"
_RELEASE_LAG_DAYS = 3      # Tuesday report_date → Friday public release (spec §4.2)

# Derived positioning fields → the Socrata columns they combine. Keeps the mined terminal economic
# (net positioning), not a raw column the DSL would have to recombine.
_FIELDS: dict[str, tuple[str, ...]] = {
    "comm_net": ("comm_positions_long_all", "comm_positions_short_all"),
    "noncomm_net": ("noncomm_positions_long_all", "noncomm_positions_short_all"),
    "comm_net_pct_oi": ("comm_positions_long_all", "comm_positions_short_all", "open_interest_all"),
}


def _default_transport(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as resp:   # noqa: S310 - fixed https host
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, list) else [data]


class CftcCotConnector:
    """DataConnector for CFTC COT. ``asset_class = 'positioning'``. series_id = ``<market_code>:<field>``."""

    source_id = "cot"
    asset_class = "positioning"

    def __init__(
        self,
        *,
        transport: Callable[[str], list[dict]] | None = None,
        markets: tuple[tuple[str, str], ...] = (),
        release_lag_days: int = _RELEASE_LAG_DAYS,
    ) -> None:
        self._transport = transport
        # markets: (cftc_contract_market_code, human_name). Empty by default — the operator/Data
        # Scout supplies the contracts of interest (e.g. ('067651','GOLD - COMMODITY EXCHANGE')).
        self._markets = markets
        self._release_lag = np.timedelta64(release_lag_days, "D")

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, f"{code}:{field}", self.asset_class,
                      title=f"{name} — {field}", frequency="W")
            for code, name in self._markets
            for field in _FIELDS
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        return Provenance(
            source_id=self.source_id,
            url=_RESOURCE,
            license=_LICENSE,
            as_of_policy="release-lag",       # stamped report_date + lag (CR-4)
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
        code, field = self._split(ref.series_id)
        rows = self._request(code, start, end)
        return self._parse(ref, field, rows, as_of=as_of)

    # -- internals -----------------------------------------------------------------
    @staticmethod
    def _split(series_id: str) -> tuple[str, str]:
        code, _, field = series_id.partition(":")
        if field not in _FIELDS:
            raise ValueError(f"unknown COT field {field!r}; supported: {sorted(_FIELDS)}")
        return code, field

    def _request(self, code: str, start, end) -> list[dict]:
        cols = "report_date_as_yyyy_mm_dd,cftc_contract_market_code," + ",".join(
            sorted({c for combo in _FIELDS.values() for c in combo}))
        where = (f"cftc_contract_market_code='{code}' "
                 f"AND report_date_as_yyyy_mm_dd between '{np.datetime64(start, 'D')}' "
                 f"and '{np.datetime64(end, 'D')}'")
        query = urllib.parse.urlencode({
            "$select": cols, "$where": where,
            "$order": "report_date_as_yyyy_mm_dd", "$limit": "50000"})
        url = f"{_RESOURCE}?{query}"
        transport = self._transport or _default_transport
        return transport(url)

    def _parse(self, ref: SeriesRef, field: str, rows: list[dict], *, as_of) -> SeriesData:
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None
        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for r in rows:
            try:
                report = np.datetime64(str(r["report_date_as_yyyy_mm_dd"])[:10], "ns")
                value = self._field_value(field, r)
            except (KeyError, TypeError, ValueError):
                continue
            if value is None:
                continue
            release = report + self._release_lag        # Tuesday report → Friday public (CR-4)
            if cutoff is not None and release > cutoff:  # vintage: only what was public by as_of
                continue
            refs.append(report)
            vals.append(value)
            rels.append(release)
        return SeriesData(
            ref=ref,
            reference_period=np.array(refs, dtype="datetime64[ns]"),
            value=np.array(vals, dtype=np.float64),
            release_timestamp=np.array(rels, dtype="datetime64[ns]"),
            provenance=self.provenance(ref),
            meta={"n_raw": len(rows), "field": field},
        )

    @staticmethod
    def _field_value(field: str, row: dict) -> float | None:
        cols = _FIELDS[field]
        try:
            nums = [float(row[c]) for c in cols]
        except (KeyError, TypeError, ValueError):
            return None
        if field == "comm_net":
            return nums[0] - nums[1]
        if field == "noncomm_net":
            return nums[0] - nums[1]
        if field == "comm_net_pct_oi":
            long_, short_, oi = nums
            return (long_ - short_) / oi if oi else None
        return None
