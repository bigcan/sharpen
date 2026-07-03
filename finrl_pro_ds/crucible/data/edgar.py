"""SEC EDGAR connector (spec §4.2 Tier-A fundamentals / filings) — Crucible P5.

Filing timing and reported fundamentals are genuinely capacity-constrained alt-data (spec §4.2). This
connector reads EDGAR's XBRL **company-concept** API (``data.sec.gov/api/xbrl/companyconcept``), which
returns every reported value of one financial-statement concept for one filer, each carrying the two
timestamps PIT correctness lives or dies by: the fiscal-period **end** and the **filed** date.

This is the CLEANEST true-PIT source in the stack: unlike FRED/COT (which model release as
``reference_period + lag``), EDGAR gives the *actual* publication timestamp per observation — the
**filing-acceptance ``filed`` date** — so ``release_timestamp = filed`` directly (CR-4). The spec §4.2
gotcha ("use the filing *acceptance* timestamp, not period-end") is therefore enforced structurally,
not approximated. Amendments and later filings restate an earlier period: EDGAR returns them as extra
observations for the SAME ``end`` with a LATER ``filed``, which :func:`quality_gate.asof_join` replays
as a revision (the newest-released reading in effect wins) — exactly the case the join was built for.

series_id is the source-native ``<cik>:<concept>`` (e.g. ``0000320193:Revenues``). That is not a
DSL-legal terminal (digit-leading, two colons once prefixed), so — like a COT id — the Panel-slot
caller supplies an explicit alias (``edgar:aapl_revenue``) via ``SlotRequest.terminal``; the connector
layer is unconstrained.

Testability mirrors FRED/COT: inject a ``transport`` callable ``(url) -> dict`` and no network / UA is
needed. The live path hits ``data.sec.gov`` over HTTPS (keyless) but SEC policy REQUIRES a descriptive
``User-Agent`` — supply one via ``user_agent=`` or the ``SEC_EDGAR_UA`` env var, else the live pull
fails closed with a clear error (injected transports are exempt).

Frame consistency (``period_days``, default standalone-quarter 80–100d): companyconcept returns a flow
concept under overlapping frames for the SAME period ``end`` — standalone quarter (~90d), YTD cumulative
(~180/270d), full year (~360d). Admitting all of them is not a leak but yields a sawtooth feature; the
default filter keeps only the quarterly span so the bridged series is a clean quarterly flow (:meth:
`_keep_period`). Instantaneous concepts (no ``start``) are unaffected; ``period_days=None`` disables it.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections.abc import Callable

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://data.sec.gov/api/xbrl/companyconcept"
_LICENSE = "SEC EDGAR — U.S. Government public data (no redistribution limit)"
_DEFAULT_TAXONOMY = "us-gaap"
# Preferred unit when a concept reports several (financial-statement concepts are overwhelmingly USD).
_PREFERRED_UNITS = ("USD", "USD/shares", "shares", "pure")

# A small curated fundamentals starter set: (cik, concept, human name). discover() returns these unless
# a custom list is supplied; the Data Scout (P5) proposes additions. CIKs are zero-padded to 10 digits
# on request, so a bare integer string is accepted here.
#
# Revenue uses RevenueFromContractWithCustomerExcludingAssessedTax, NOT the legacy us-gaap:Revenues
# concept: both AAPL and MSFT abandoned `Revenues` when they adopted ASC 606 (FY2018), so that tag is a
# stale stub that dies ~2018 and returns EMPTY in any recent window (the "empty series" reject seen in
# the first live scout). The ASC 606 tag is the concept both file under through 2026 (verified
# 2026-07-03: AAPL 2017-09→2026-03 @113 obs, MSFT 2016-06→2026-03 @131 obs).
_DEFAULT_CONCEPTS: tuple[tuple[str, str, str], ...] = (
    ("0000320193", "RevenueFromContractWithCustomerExcludingAssessedTax", "Apple — Revenue (ASC 606)"),
    ("0000320193", "NetIncomeLoss", "Apple — Net income"),
    ("0000789019", "RevenueFromContractWithCustomerExcludingAssessedTax", "Microsoft — Revenue (ASC 606)"),
    ("0000789019", "NetIncomeLoss", "Microsoft — Net income"),
)


def _default_transport_factory(user_agent: str) -> Callable[[str], dict]:
    def _transport(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=30) as resp:   # noqa: S310 - fixed https host
            return json.loads(resp.read().decode("utf-8"))
    return _transport


class EdgarConnector:
    """DataConnector for SEC EDGAR XBRL company facts. ``asset_class = 'fundamental'``.
    series_id = ``<cik>:<concept>``."""

    source_id = "edgar"
    asset_class = "fundamental"

    def __init__(
        self,
        *,
        transport: Callable[[str], dict] | None = None,
        concepts: tuple[tuple[str, str, str], ...] | None = None,
        taxonomy: str = _DEFAULT_TAXONOMY,
        user_agent: str | None = None,
        period_days: tuple[int, int] | None = (80, 100),
    ) -> None:
        self._transport = transport
        self._concepts = concepts or _DEFAULT_CONCEPTS
        self._taxonomy = taxonomy
        self._user_agent = user_agent if user_agent is not None else os.environ.get("SEC_EDGAR_UA", "")
        # Period-consistency window (see _parse): keep only standalone-quarterly durations for FLOW
        # concepts. None disables the filter (legacy all-frames behavior).
        self._period_days = period_days

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, f"{_cik10(cik)}:{concept}", self.asset_class,
                      title=title, frequency="Q")
            for cik, concept, title in self._concepts
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        cik, concept = self._split(ref.series_id)
        return Provenance(
            source_id=self.source_id,
            url=f"{_BASE}/CIK{cik}/{self._taxonomy}/{concept}.json",
            license=_LICENSE,
            # Not a lag model — the actual filing-acceptance timestamp IS the release (CR-4).
            as_of_policy="filing-acceptance",
            release_lag_days=0,
            revision_policy="revised",         # amendments/later filings restate a period
        )

    def fetch(
        self,
        ref: SeriesRef,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        as_of: np.datetime64 | str | None = None,
    ) -> SeriesData:
        cik, concept = self._split(ref.series_id)
        payload = self._request(cik, concept)
        return self._parse(ref, payload, start, end, as_of=as_of)

    # -- internals -----------------------------------------------------------------
    @staticmethod
    def _split(series_id: str) -> tuple[str, str]:
        cik, _, concept = series_id.partition(":")
        if not cik or not concept:
            raise ValueError(f"EDGAR series_id must be '<cik>:<concept>'; got {series_id!r}")
        return _cik10(cik), concept

    def _request(self, cik: str, concept: str) -> dict:
        url = f"{_BASE}/CIK{cik}/{self._taxonomy}/{concept}.json"
        transport = self._transport or self._live_transport()
        return transport(url)

    def _live_transport(self) -> Callable[[str], dict]:
        if not self._user_agent:
            raise RuntimeError(
                "EdgarConnector live fetch requires a descriptive User-Agent (SEC policy). Set "
                "SEC_EDGAR_UA (e.g. 'FinRL-Pro-DS research you@example.com') or pass user_agent=. "
                "For offline/test use, inject a `transport` callable instead.")
        return _default_transport_factory(self._user_agent)

    def _parse(self, ref: SeriesRef, payload: dict, start, end, *, as_of) -> SeriesData:
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None
        lo, hi = np.datetime64(start, "ns"), np.datetime64(end, "ns")
        units = payload.get("units", {})
        rows = self._pick_unit(units)
        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for row in rows:
            filed = row.get("filed")
            end_p = row.get("end")
            val = row.get("val")
            if filed is None or end_p is None or val is None:
                continue
            try:
                reference = np.datetime64(str(end_p)[:10], "ns")
                release = np.datetime64(str(filed)[:10], "ns")   # filing-acceptance date (CR-4)
                value = float(val)
            except (TypeError, ValueError):
                continue
            if reference < lo or reference > hi:
                continue
            if cutoff is not None and release > cutoff:          # vintage: only what was public by as_of
                continue
            if not self._keep_period(row.get("start"), reference):  # frame-consistency filter
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
            meta={"n_raw": len(rows), "unit": self._unit_key(units)},
        )

    def _keep_period(self, start_p, reference: np.datetime64) -> bool:
        """Frame-consistency gate. EDGAR's companyconcept endpoint returns a FLOW concept (revenue,
        net income) under MULTIPLE overlapping frames for the same fiscal-period ``end``: the
        standalone quarter (~90d), the year-to-date cumulatives (~180/270d) and the full year (~360d).
        Admitting all of them builds a sawtooth feature (a $95B quarter joined next to a $416B year) —
        not a leak (every value is correctly release-stamped) but a semantically incoherent series an
        overlay would score as an artifact. So for a DURATION observation (one carrying a ``start``) we
        keep only the standalone-quarter span, yielding a clean, comparable quarterly series. An
        INSTANTANEOUS observation (balance-sheet item; no ``start``) has no duration and is always kept.
        ``period_days=None`` disables the filter entirely (legacy all-frames behavior)."""
        if self._period_days is None or start_p is None:
            return True
        try:
            span = int((reference - np.datetime64(str(start_p)[:10], "ns")) / np.timedelta64(1, "D"))
        except (TypeError, ValueError):
            return True                                          # unparseable start -> don't drop
        lo_d, hi_d = self._period_days
        return lo_d <= span <= hi_d

    @staticmethod
    def _unit_key(units: dict) -> str | None:
        for u in _PREFERRED_UNITS:
            if u in units:
                return u
        return next(iter(units), None)

    def _pick_unit(self, units: dict) -> list[dict]:
        key = self._unit_key(units)
        return list(units.get(key, [])) if key is not None else []


def _cik10(cik: str | int) -> str:
    """Zero-pad a CIK to the 10-digit form the EDGAR API expects (``320193`` → ``0000320193``)."""
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        raise ValueError(f"invalid CIK {cik!r}")
    return digits.zfill(10)
