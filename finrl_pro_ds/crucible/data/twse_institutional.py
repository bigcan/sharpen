"""TWSE three-institutional-investors connector (Taiwan positioning, T86) — Crucible v2.8.

Taiwan's analog of CFTC COT: daily net buy/sell by foreign, investment-trust, and dealer investor
categories, per listed security. Unlike COT (a business-day-lagged, range-queryable Socrata feed),
TWSE's legacy report gateway (``www.twse.com.tw/rwd/zh/fund/T86``) is a per-CALENDAR-DAY snapshot
covering EVERY listed security in one response (``selectType=ALL``) — there is no server-side date
RANGE query, so ``fetch`` iterates days itself and caches each day's raw payload once per connector
instance (a naive per-(ticker, field) fetch would otherwise re-download the same day's full universe
up to 40x for this connector's default 10-ticker x 4-field ``discover()`` set).

Live-verified 2026-07-04 (session S553-cont-115, see
``.agent/artifacts/crucible_taiwan_breadth_and_scheduling_architecture.md``):
  - Response shape: ``{"stat":"OK","date":"YYYYMMDD","title":"...(ROC year, e.g. 115年)...",
    "fields":[19 Chinese column names],"data":[[19 string values] per security]}``. The ``date``
    field itself is Gregorian (matches the request param) — ROC only ever appears inside the
    free-text ``title``, which this parser never reads. There is NO per-row date; every row in one
    response describes the SAME queried day.
  - Numbers are comma-formatted strings (e.g. ``"52,683,779"``, ``"-445,000"``) — must strip commas
    before ``float()`` or every value silently raises.
  - Column lookup is BY NAME against the response's own ``fields`` array (robust to any future TWSE
    reordering), not by hardcoded position.

Release-timestamp policy (CR-4, the anti-COT-holiday-lag-bug design — architecture doc ADR-A2): T86
is a same-trading-day-evening publication, so ``release_timestamp = reference_period + 1 CALENDAR
day`` (never a business-day count). Because :func:`quality_gate.asof_join` binds a bar to "most
recent release <= that bar's real trading date", a flat calendar-day lag is automatically carried
forward to whichever real trading day comes next — correct regardless of weekends or Taiwan holidays,
with no Taiwan holiday calendar required (one fewer assumption than the COT connector needed).

Testability mirrors every other connector: inject a ``transport`` callable ``(url) -> dict`` and no
network is needed. The live path needs no key (T86 is a public, keyless legacy report endpoint) but
DOES send a browser-like User-Agent (the live probe that verified this connector's shape used one).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from collections.abc import Callable

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://www.twse.com.tw/rwd/zh/fund/T86"
_LICENSE = "TWSE (Taiwan Stock Exchange) — public data, keyless"
_USER_AGENT = "Mozilla/5.0 (FinRL-Pro-DS research; contact via project owner)"

# The 10-ETF panel universe (configs/taiwan_cross_asset.yaml) — the default discover() scope. A
# caller may pass a different `tickers` tuple to widen/narrow it; the quality gate + DataScout
# accept/reject filter handles any ticker T86 doesn't cover (e.g. a thin bond ETF) gracefully — no
# code path here assumes universal coverage (live-verified only that SOME 00-prefixed ETF tickers
# appear in a T86 pull, not exhaustively all 10 of these specific ones).
_DEFAULT_TICKERS: tuple[str, ...] = (
    "0050", "006208", "0056", "0055", "00878", "00891", "00679B", "00751B", "00635U", "00642U",
)

# field key -> the EXACT TWSE column name to look up in the response's own `fields` array (live-
# verified 2026-07-04, see module docstring). Looking up BY NAME (not position) survives any future
# TWSE column reordering. `dealer_net` is already the aggregate self+hedge dealer net (index 11 in
# the live sample), not something this connector needs to sum from the self/hedge sub-columns.
_FIELD_COLUMN: dict[str, str] = {
    "foreign_net": "外陸資買賣超股數(不含外資自營商)",
    "trust_net": "投信買賣超股數",
    "dealer_net": "自營商買賣超股數",
    "total_net": "三大法人買賣超股數",
}
_TICKER_COLUMN = "證券代號"

# T86 has no server-side date-RANGE query (one HTTP call = one calendar day, ALL securities at
# once) — a naive multi-year `start` would issue thousands of sequential calls per connector
# instantiation. Cap the actual lookback so a real-mode tick stays bounded; this feature slot is
# meant to widen BREADTH of information type, not replace the OHLCV panel's own 2010+ depth. A
# persistent on-disk cache (mirroring taiwan_panel_loader's) would let real history accumulate across
# repeated scheduled runs without re-fetching — a natural follow-up, not built here (disclosed in the
# architecture doc rather than silently pretended away). 400d mirrors quality_gate's own
# `max_gap_days` default — a deliberate, not arbitrary, choice of "recent regime" depth.
_DEFAULT_MAX_LOOKBACK_DAYS = 400


def _parse_num(s: str | int | float) -> float:
    """TWSE legacy numbers are USUALLY comma-formatted strings (e.g. "52,683,779", "-445,000"),
    but the endpoint occasionally emits a bare JSON number instead of a string for a security with
    a round value in a category (live-observed 2026-07-06: 00635U's `投信買賣超股數` came back as the
    integer 0, not "0"). A bare number is a REAL zero-flow observation, not missing data — coerce it
    rather than letting `int.replace` raise AttributeError (which, not being TypeError/ValueError,
    escaped `_extract`'s guard and dropped the whole series at the altdata bridge)."""
    if isinstance(s, (int, float)):
        return float(s)
    return float(s.replace(",", ""))


def _default_transport(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:      # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


class TwseInstitutionalConnector:
    """DataConnector for TWSE T86 (three-institutional-investors daily net flow).
    ``asset_class = 'positioning'``. series_id = ``<ticker>:<field>``."""

    source_id = "twse_inst"
    asset_class = "positioning"

    def __init__(
        self,
        *,
        transport: Callable[[str], dict] | None = None,
        tickers: tuple[str, ...] | None = None,
        release_lag_days: int = 1,
        max_lookback_days: int = _DEFAULT_MAX_LOOKBACK_DAYS,
        sleep_seconds: float = 0.3,
    ) -> None:
        self._transport = transport
        self._tickers = tickers or _DEFAULT_TICKERS
        self._release_lag = np.timedelta64(int(release_lag_days), "D")
        self._max_lookback_days = int(max_lookback_days)
        # Politeness delay between LIVE per-day HTTP calls only (mirrors taiwan_panel_loader's own
        # `sleep=` convention) — never applied when a `transport` is injected (offline tests).
        self._sleep_seconds = float(sleep_seconds)
        # Per-instance cache of already-fetched days (raw parsed JSON), so N (ticker, field) fetches
        # against the same day cost exactly ONE HTTP call, not N. Keyed by "YYYYMMDD".
        self._day_cache: dict[str, dict | None] = {}

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, f"{ticker}:{field}", self.asset_class,
                      title=f"{ticker} — {field}", frequency="D")
            for ticker in self._tickers
            for field in _FIELD_COLUMN
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        return Provenance(
            source_id=self.source_id,
            url=_BASE,
            license=_LICENSE,
            as_of_policy="release-lag",           # reference_period + release_lag_days (CR-4)
            release_lag_days=int(self._release_lag / np.timedelta64(1, "D")),
            revision_policy="final",              # T86 is a settled daily report, no later revision
        )

    def fetch(
        self,
        ref: SeriesRef,
        start: np.datetime64 | str,
        end: np.datetime64 | str,
        *,
        as_of: np.datetime64 | str | None = None,
    ) -> SeriesData:
        ticker, field = self._split(ref.series_id)
        column = _FIELD_COLUMN[field]
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None

        lo = np.datetime64(start, "D")
        hi = np.datetime64(end, "D")
        # Cap the lookback (module docstring) — never crawl further back than max_lookback_days
        # regardless of what the caller's `start` requests.
        floor = hi - np.timedelta64(self._max_lookback_days, "D")
        if lo < floor:
            lo = floor

        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for day in np.arange(lo, hi + np.timedelta64(1, "D"), np.timedelta64(1, "D")):
            payload = self._fetch_day(str(day))
            if payload is None:
                continue                          # non-trading day / no data — not an error
            value = self._extract(payload, ticker, column)
            if value is None:
                continue
            reference = np.datetime64(day, "ns")
            release = reference + self._release_lag
            if cutoff is not None and release > cutoff:
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
            meta={"n_days_queried": int((hi - lo) / np.timedelta64(1, "D")) + 1},
        )

    # -- internals -----------------------------------------------------------------
    @staticmethod
    def _split(series_id: str) -> tuple[str, str]:
        ticker, _, field = series_id.partition(":")
        if field not in _FIELD_COLUMN:
            raise ValueError(f"unknown TWSE institutional field {field!r}; "
                             f"supported: {sorted(_FIELD_COLUMN)}")
        return ticker, field

    def _fetch_day(self, day: str) -> dict | None:
        """One HTTP call per calendar day, cached for the life of this connector instance — shared
        across every (ticker, field) SeriesRef so a 10-ticker x 4-field discover() set costs exactly
        one request per day, not 40. Returns None for a non-trading day / malformed response (treated
        as "no observation", never an error — T86 simply has nothing to say on a closed day)."""
        date_str = day.replace("-", "")
        if date_str in self._day_cache:
            return self._day_cache[date_str]
        url = f"{_BASE}?response=json&date={date_str}&selectType=ALL"
        live = self._transport is None
        transport = self._transport or _default_transport
        try:
            payload = transport(url)
        except Exception as exc:                  # noqa: BLE001 - transport failed closed, not fatal
            logger.info("twse_inst: fetch failed for %s (%r) — treated as no data", date_str, exc)
            payload = None
        if not isinstance(payload, dict) or payload.get("stat") != "OK" or "fields" not in payload:
            payload = None
        self._day_cache[date_str] = payload
        if live and self._sleep_seconds > 0:
            time.sleep(self._sleep_seconds)
        return payload

    @staticmethod
    def _extract(payload: dict, ticker: str, column: str) -> float | None:
        fields = payload.get("fields", [])
        try:
            ticker_idx = fields.index(_TICKER_COLUMN)
            col_idx = fields.index(column)
        except ValueError:
            return None
        for row in payload.get("data", []):
            if len(row) <= max(ticker_idx, col_idx):
                continue
            if row[ticker_idx].strip() == ticker:
                try:
                    return _parse_num(row[col_idx])
                except (TypeError, ValueError, AttributeError):
                    # This guard's job is to turn an unparseable value cell into "no observation for
                    # this row" rather than a hard error. AttributeError belongs here too: if the cell
                    # is neither a number nor a comma-string but some other JSON type (e.g. a bare
                    # `null` -> None, on which `.replace` raises AttributeError), drop THIS row — never
                    # let one junk cell escape and kill the whole series at the altdata bridge (the
                    # bare-int failure mode `_parse_num` now handles for real numbers). A bare number
                    # is kept by `_parse_num`; only genuinely non-numeric cells reach here.
                    return None
        return None
