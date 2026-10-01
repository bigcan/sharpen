"""CFTC Commitments of Traders connector (spec §4.2 Tier-A positioning) — Crucible P1b.

Large-spec / commercial positioning is a classic uncrowded signal — big money's own footprint.
COT is public data (NO API key). The PIT crux (spec §4.2 gotcha, CR-4): the report describes
**Tuesday** positions but only publishes the following **Friday** — a ~3-day release lag. Joining
Tuesday's report to Tuesday's bar is look-ahead. This connector therefore stamps
``release_timestamp`` at the true public release so :func:`quality_gate.asof_join` binds it to the
first bar on/after the release, never the Tuesday it describes.

Release model (COT-HOLIDAY-LAG-LOOKAHEAD fix, audit S553): the release is the report date rolled
forward by ``release_lag`` **US-federal BUSINESS days** (CFTC follows the federal holiday schedule),
NOT a fixed +3 CALENDAR days. On a normal week Tuesday + 3 business days == Friday (unchanged). On a
week with a federal holiday in the Wed–Fri window (Thanksgiving, year-end, …) CFTC delays the publish
to the next business day, so the business-day roll pushes the stamp to that true Monday+ release —
closing the ~15–19%-of-weeks look-ahead a fixed calendar lag baked in (no downstream causality gate
could catch it: they all take this stamp as ground truth). Business-day rolling is always ≥ the old
calendar stamp, so it never leaks relative to prior behaviour; it only ever delays availability.

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
from functools import lru_cache

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

# Legacy futures-only report (Socrata resource id). publicreporting.cftc.gov is keyless.
_RESOURCE = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_LICENSE = "CFTC Commitments of Traders — U.S. Government public data (no redistribution limit)"
_RELEASE_LAG_DAYS = 3      # Tuesday report_date → Friday public release, in BUSINESS days (spec §4.2)


# crucible-v15.0 — federal CLOSURES that ``USFederalHolidayCalendar`` does not encode: executive-order
# closures (mostly Christmas Eve / the day after Christmas) and national days of mourning. CFTC (and the
# Fed's H.4.1) follow them, so a release window containing one publishes a business day later. Missing
# one leaks a day; including a doubtful one only delays availability by a day — so err on inclusion.
# Evidence for the in-sample ones: CFTC "COT Historical Special Announcements" (2014-12-26 moved the
# release to Tue Dec 30; 2020-12-24 moved it to Mon Dec 28; 2025-01-09 moved it to Mon Jan 13).
_AD_HOC_FEDERAL_CLOSURES: tuple[str, ...] = (
    "1994-04-27",                                            # Nixon — national day of mourning
    "2001-12-24", "2007-12-24", "2008-12-26", "2012-12-24", "2014-12-26", "2015-12-24",
    "2018-12-24", "2019-12-24", "2020-12-24", "2024-12-24", "2025-12-24", "2025-12-26",  # exec. orders
    "2004-06-11",                                            # Reagan — national day of mourning
    "2007-01-02",                                            # Ford — national day of mourning
    "2018-12-05",                                            # G.H.W. Bush — national day of mourning
    "2025-01-09",                                            # Carter — national day of mourning
)

# crucible-v15.0 — ACTUAL publication dates of COT reports whose release DEVIATED from any calendar
# rule: the 2013, 2018-19 and 2025 federal shutdowns and the 2023 ION cyber incident. Keyed by the
# report's as-of date. The connector stamps ``max(business-day model, pinned)``, so a pinned date can
# only DELAY availability. Before this table the 2025 shutdown alone put ~16 weekly reports up to seven
# weeks early — inside every current holdout, on 18 of the 24 us_equity alt-data slots.
# Sources (CFTC press releases / special announcements):
#   2013: PR 6745-13 — Oct 1 data on Oct 25, then a rolling catch-up, current by Nov 8 (the LATER day of
#         each announced week is pinned, never an earlier one).
#   2018-19: PR 7864-19 — Dec 24 data on Feb 1, then two reports a week (sources differ on Mon/Thu vs
#         Tue/Fri: the later day is pinned), current with the Mar 5 report on Mar 8.
#   2023: "Historical Special Announcements" — each delayed ION-incident report's issue date verbatim
#         (the Feb 7 report's issue date is not listed; its successor's is used, which is later).
#   2025: PR 9138-25 and PR 9147-25 give two schedules (the second accelerated); the LATER is pinned.
_PINNED_RELEASES: dict[str, str] = {
    "2013-10-01": "2013-10-25", "2013-10-08": "2013-11-01", "2013-10-15": "2013-11-01",
    "2013-10-22": "2013-11-08", "2013-10-29": "2013-11-08",
    "2018-12-24": "2019-02-01", "2018-12-31": "2019-02-05", "2019-01-08": "2019-02-08",
    "2019-01-15": "2019-02-12", "2019-01-22": "2019-02-15", "2019-01-29": "2019-02-19",
    "2019-02-05": "2019-02-22", "2019-02-12": "2019-02-26", "2019-02-19": "2019-03-01",
    "2019-02-26": "2019-03-05",
    "2023-01-31": "2023-02-24", "2023-02-07": "2023-03-08", "2023-02-14": "2023-03-08",
    "2023-02-21": "2023-03-10", "2023-02-28": "2023-03-14", "2023-03-07": "2023-03-16",
    "2023-03-14": "2023-03-21",
    "2025-09-30": "2025-11-19", "2025-10-07": "2025-11-21", "2025-10-14": "2025-11-25",
    "2025-10-21": "2025-12-02", "2025-10-28": "2025-12-05", "2025-11-04": "2025-12-09",
    "2025-11-10": "2025-12-12", "2025-11-18": "2025-12-16", "2025-11-25": "2025-12-19",
    "2025-12-02": "2025-12-23", "2025-12-09": "2025-12-30", "2025-12-16": "2026-01-06",
    "2025-12-23": "2026-01-09", "2025-12-30": "2026-01-13", "2026-01-06": "2026-01-16",
    "2026-01-13": "2026-01-20",
}


@lru_cache(maxsize=1)
def _us_federal_busdaycal() -> np.busdaycalendar:
    """US federal holiday + closure business-day calendar for rolling a report date to its true release.

    CFTC observes the US federal holiday schedule, so a federal holiday in the release window delays
    the COT publish to the next business day. Built once from pandas' ``USFederalHolidayCalendar``
    (which encodes the observed-date rules and Juneteenth-from-2021) over a wide static span, PLUS the
    ad-hoc closures above (v15.0 — the pandas calendar knows no executive-order or mourning closure), so
    :func:`numpy.busday_offset` can roll releases forward past them. Pandas is a core project
    dependency; a missing holiday only ever under-delays a release, so we fail loud rather than guess.
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar

    hols = USFederalHolidayCalendar().holidays(start="1990-01-01", end="2035-12-31")
    days = np.union1d(hols.values.astype("datetime64[D]"),
                      np.array(_AD_HOC_FEDERAL_CLOSURES, dtype="datetime64[D]"))
    return np.busdaycalendar(holidays=days)

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
        # Scout supplies the contracts of interest (e.g. ('088691','GOLD - COMMODITY EXCHANGE INC.')).
        self._markets = markets
        # release lag counted in US-federal BUSINESS days (holiday-aware — COT-HOLIDAY-LAG fix). The
        # default 3 gives Tuesday→Friday on a normal week and rolls past holidays when present.
        self._release_lag_bdays = int(release_lag_days)

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
            as_of_policy="release-lag-busday",  # report_date rolled +lag US-federal business days (CR-4)
            release_lag_days=self._release_lag_bdays,
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
    def _release_for(self, report: np.datetime64) -> np.datetime64:
        """True public release = ``report`` rolled forward ``release_lag`` US-federal BUSINESS days,
        floored at the PINNED actual publication date for a report inside a disrupted window.

        Tuesday + 3 business days == Friday on a normal week (identical to the old calendar stamp); on
        a holiday or closure week the roll skips the closed day(s) to CFTC's actual next-business-day
        publish, fixing COT-HOLIDAY-LAG-LOOKAHEAD. ``roll='forward'`` also guards a non-business
        report_date. v15.0: ``max(model, _PINNED_RELEASES[report])`` — a shutdown or outage is not a
        calendar event, so no rule can model it; the pinned table can only delay a stamp.
        """
        rd = np.datetime64(report, "D")
        rel = np.busday_offset(
            rd, self._release_lag_bdays, roll="forward", busdaycal=_us_federal_busdaycal())
        pinned = _PINNED_RELEASES.get(str(rd))
        if pinned is not None:
            rel = max(rel, np.datetime64(pinned, "D"))
        return np.datetime64(rel, "ns")

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
            release = self._release_for(report)          # report → true public release (busday, CR-4)
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
