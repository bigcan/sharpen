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
import os
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://www.twse.com.tw/rwd/zh/fund/T86"
_LICENSE = "TWSE (Taiwan Stock Exchange) — public data, keyless"
_USER_AGENT = "Mozilla/5.0 (Sharpen research; contact via project owner)"

# The 10-ETF panel universe (configs/taiwan_cross_asset.yaml) — the default discover() scope. A
# caller may pass a different `tickers` tuple to widen/narrow it; the quality gate + DataScout
# accept/reject filter handles any ticker T86 doesn't cover (e.g. a thin bond ETF) gracefully — no
# code path here assumes universal coverage (live-verified only that SOME 00-prefixed ETF tickers
# appear in a T86 pull, not exhaustively all 10 of these specific ones).
_DEFAULT_TICKERS: tuple[str, ...] = (
    "0050", "006208", "0056", "0055", "00878", "00891", "00679B", "00751B", "00635U", "00642U",
)

# field key -> an ORDERED tuple of candidate TWSE column names; `_extract_day` uses the FIRST one
# present in the response's own `fields` array. Lookup is BY NAME (not position) to survive TWSE
# column reordering. Most fields have ONE stable name across T86's whole history, but the T86 SCHEMA
# CHANGED ~2018 (live-verified 2026-07-06 across 2015/2018/2020/2026 payloads): the pre-2018 report
# had 16 columns with a SINGLE foreign column `外資買賣超股數`; the modern 19-column report SPLIT
# foreign into `外陸資買賣超股數(不含外資自營商)` (foreign + mainland capital, EXCLUDING foreign
# proprietary dealers) plus a separate foreign-dealer column. So `foreign_net` carries BOTH names,
# modern-first with the pre-2018 name as fallback — stitching one continuous "foreign investor net"
# overlay across a KNOWN DEFINITIONAL SEAM at ~2018 (the closest continuous series TWSE's own schema
# change allows; the added-mainland / excluded-foreign-dealer delta is minor for this large-cap ETF
# universe, but the seam is real and disclosed, not silently pretended away). trust/dealer/total
# names are stable across both eras, so they carry a single name. `dealer_net` is the AGGREGATE
# self+hedge dealer net, present in both schemas — not summed from the self/hedge sub-columns.
_FIELD_COLUMNS: dict[str, tuple[str, ...]] = {
    "foreign_net": ("外陸資買賣超股數(不含外資自營商)", "外資買賣超股數"),
    "trust_net": ("投信買賣超股數",),
    "dealer_net": ("自營商買賣超股數",),
    "total_net": ("三大法人買賣超股數",),
}
_TICKER_COLUMN = "證券代號"

# --- Persistent accumulation store (S553-cont-117, P0) -------------------------------------------
# T86 has no server-side date-RANGE query (one HTTP call = one calendar day, ALL securities at once)
# but it IS range-queryable day-by-day arbitrarily far back (live-verified: full snapshots returned for
# 2014-01-03 and 2018-01-05). So this connector persists each fetched day to an on-disk store — the
# "natural follow-up" the old 400d-cap comment disclosed but did not build. The store is BOTH a cache
# (a nightly tick re-fetches only genuinely new days, not the whole history) AND one-shot backfillable
# (scripts/research/backfill_twse_t86.py fills 2015->today once, lifting overlay coverage from ~267 to
# ~2750 bars). It mirrors TaifexPositioningConnector's _AccumulationStore, but is keyed by the FETCH
# UNIT — a full calendar DAY — value = {ticker: {field: value}}; an EMPTY dict records a POLLED
# non-trading day, and a day-key ABSENT from the store means "never polled" (so weekends/holidays and
# T86-absent bond ETFs are not re-fetched every tick). The store is causally inert: it caches raw
# reference-day values only; the +1-calendar-day release stamp and the as_of cutoff are applied at READ
# time in fetch(), identically to before — a backfilled 2015 day is stamped release=2015..+1d and can
# never be seen before then (LEAK-2 preserved; parity-tested against the direct-transport path).
ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STORE_PATH = ROOT / "data" / "raw" / "taiwan_t86_institutional" / "t86_institutional.json"

# The store makes a deep read cheap, so the cap on the READ RANGE is raised to cover the panel's full
# 2015+ history (was 400d — the ~9.5%-coverage cap that left every overlay power-starved at ~1yr). A
# SEPARATE, modest per-instance LIVE-fetch budget bounds how many un-polled days a single connector
# instance will pull over the network, so a COLD-store orchestrator tick fills the recent tail
# (descending) in ~seconds rather than hanging on a ~4000-day inline crawl; the backfill script raises
# that budget to fill everything in one run.
_DEFAULT_MAX_LOOKBACK_DAYS = 4400        # ~12y read range (panel starts 2015-01-05); store-backed
_DEFAULT_MAX_LIVE_DAYS_PER_FETCH = 90    # per-instance network budget; backfill overrides to fill all


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


class _AccumulationStore:
    """A tiny, idempotent JSON store — ``"YYYYMMDD" -> {ticker: {field: value}}`` — mirroring
    :class:`~.taifex_positioning._AccumulationStore` but keyed by the T86 FETCH UNIT (a full calendar
    day) rather than ``(contract, date)``. A day KEY present means "polled"; an EMPTY value dict means
    "polled, non-trading / no configured ticker present"; a ticker absent from a present day's dict
    means that ticker had no T86 row that day. Scoped to exactly this connector's needs, not a
    general store (same spirit as the TAIFEX one)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, dict[str, float]]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("twse_inst: store at %s unreadable — treating as empty", self.path)
            return {}

    def save(self, data: dict[str, dict[str, dict[str, float]]]) -> None:
        # Atomic write (tmp + os.replace on the same dir/volume): a checkpoint interrupted mid-write
        # during a long backfill leaves the PREVIOUS complete store intact rather than a torn file that
        # load() would treat as empty (losing all accumulated progress on resume).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


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
        max_live_days_per_fetch: int = _DEFAULT_MAX_LIVE_DAYS_PER_FETCH,
        sleep_seconds: float = 0.3,
        store_path: str | Path | None = None,
        save_every_days: int = 200,
    ) -> None:
        self._transport = transport
        self._tickers = tickers or _DEFAULT_TICKERS
        self._release_lag = np.timedelta64(int(release_lag_days), "D")
        self._max_lookback_days = int(max_lookback_days)
        # Per-instance network budget: at most this many un-polled days are fetched LIVE across the
        # whole life of this connector instance (shared by every SeriesRef's fetch, like the old
        # day-cache). Bounds a cold-store tick; the backfill script passes a huge value to fill all.
        self._live_budget = int(max_live_days_per_fetch)
        # Politeness delay between LIVE per-day HTTP calls only (mirrors taiwan_panel_loader's own
        # `sleep=` convention) — never applied when a `transport` is injected (offline tests).
        self._sleep_seconds = float(sleep_seconds)
        # This connector's SECOND injectable seam (like TAIFEX's): the persistent day store. Tests
        # SHOULD inject `store_path=` (a tmp file) so they never write the repo's default store.
        self._store = _AccumulationStore(store_path if store_path is not None else _DEFAULT_STORE_PATH)
        # In-memory copy of the store, loaded once per instance so 40 (ticker, field) fetch calls
        # share one disk read; written back to disk at the end of a polling pass that changed it.
        self._loaded: dict[str, dict[str, dict[str, float]]] | None = None
        # Checkpoint cadence: save the store every N newly-fetched days so a long backfill survives
        # interruption and is resumable (a re-run skips already-stored days). The orchestrator fetches
        # far fewer than this per tick, so it still saves exactly once (at the end) — no behavior change.
        self._save_every = int(save_every_days)

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, f"{ticker}:{field}", self.asset_class,
                      title=f"{ticker} — {field}", frequency="D")
            for ticker in self._tickers
            for field in _FIELD_COLUMNS
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
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None

        lo = np.datetime64(start, "D")
        hi = np.datetime64(end, "D")
        # Cap the READ RANGE (module docstring) — never read/crawl further back than max_lookback_days
        # regardless of what the caller's `start` requests. (Store-backed, so a deep range is cheap.)
        floor = hi - np.timedelta64(self._max_lookback_days, "D")
        if lo < floor:
            lo = floor

        store = self._ensure_polled(lo, hi)       # fill un-polled days live (budget-bounded), persist

        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for day in np.arange(lo, hi + np.timedelta64(1, "D"), np.timedelta64(1, "D")):
            day_rows = store.get(str(day).replace("-", ""))
            if not day_rows:                      # never-polled OR polled non-trading day → no obs
                continue
            row = day_rows.get(ticker)
            if row is None or field not in row:   # ticker absent from T86 that day / field unparseable
                continue
            reference = np.datetime64(day, "ns")
            release = reference + self._release_lag
            if cutoff is not None and release > cutoff:
                continue                          # PIT: not yet released at the as_of point (LEAK-2)
            refs.append(reference)
            vals.append(float(row[field]))
            rels.append(release)

        return SeriesData(
            ref=ref,
            reference_period=np.array(refs, dtype="datetime64[ns]"),
            value=np.array(vals, dtype=np.float64),
            release_timestamp=np.array(rels, dtype="datetime64[ns]"),
            provenance=self.provenance(ref),
            meta={"n_days_queried": int((hi - lo) / np.timedelta64(1, "D")) + 1,
                  "n_store_days_total": len(store),
                  "live_budget_remaining": self._live_budget},
        )

    # -- internals -----------------------------------------------------------------
    @staticmethod
    def _split(series_id: str) -> tuple[str, str]:
        ticker, _, field = series_id.partition(":")
        if field not in _FIELD_COLUMNS:
            raise ValueError(f"unknown TWSE institutional field {field!r}; "
                             f"supported: {sorted(_FIELD_COLUMNS)}")
        return ticker, field

    def _ensure_polled(self, lo: np.datetime64, hi: np.datetime64
                       ) -> dict[str, dict[str, dict[str, float]]]:
        """Fill any un-polled calendar day in ``[lo, hi]`` by fetching it LIVE (one T86 call per day,
        every configured ticker extracted at once), up to this instance's remaining live-fetch budget,
        then persist ONCE. Days already in the store are never re-fetched (idempotent). Iterates
        most-recent-first so a budget-limited cold tick fills the useful tail first. Shared by all 40
        (ticker, field) fetch calls via the in-memory ``self._loaded`` (one disk read per instance)."""
        if self._loaded is None:
            self._loaded = self._store.load()
        store = self._loaded
        fetched = 0
        for day in np.arange(lo, hi + np.timedelta64(1, "D"), np.timedelta64(1, "D"))[::-1]:
            if self._live_budget <= 0:
                break                             # network budget spent — remaining gaps fill next run
            date_str = str(day).replace("-", "")
            if date_str in store:
                continue                          # already polled (idempotent) — costs no budget
            result = self._fetch_day_live(date_str)   # OK-payload | {} (no data) | None (transport fail)
            self._live_budget -= 1                    # a network call was made either way
            if result is None:                        # C1-06: a transient TRANSPORT failure must NEVER be
                continue                              # recorded as a holiday — leave it unpolled to retry
            store[date_str] = self._extract_day(result) if result else {}
            fetched += 1
            if self._save_every > 0 and fetched % self._save_every == 0:
                self._store.save(store)           # checkpoint: a long backfill survives interruption
        if fetched:
            self._store.save(store)
        return store

    def _fetch_day_live(self, date_str: str) -> dict | None:
        """One live T86 HTTP call for one calendar day (all securities). THREE-state result (C1-06):
          * the parsed OK payload (``dict`` with ``stat == 'OK'``) — a trading day WITH data;
          * ``{}`` — a DEFINITIVE no-data reply (valid JSON dict, ``stat != 'OK'``): TWSE answered and
            there is nothing for this day (a genuine non-trading day); the caller persists it as polled;
          * ``None`` — a TRANSPORT failure (exception or non-dict/unparseable reply): the caller leaves
            the day UNPOLLED so it is retried next run, instead of permanently recording a transient
            rate-limit / network blip as a market holiday (the C1-06 silent-trading-day-deletion bug)."""
        url = f"{_BASE}?response=json&date={date_str}&selectType=ALL"
        live = self._transport is None
        transport = self._transport or _default_transport
        try:
            payload = transport(url)
        except Exception as exc:                  # noqa: BLE001 - transport failed: RETRY, not "no data"
            logger.info("twse_inst: fetch FAILED for %s (%r) — leaving unpolled, will retry", date_str, exc)
            if live and self._sleep_seconds > 0:
                time.sleep(self._sleep_seconds)
            return None
        if live and self._sleep_seconds > 0:
            time.sleep(self._sleep_seconds)
        if not isinstance(payload, dict):         # unparseable/None reply ⇒ transport-level failure
            logger.info("twse_inst: non-dict reply for %s — leaving unpolled, will retry", date_str)
            return None
        if payload.get("stat") != "OK" or "fields" not in payload:
            return {}                             # valid reply, no data ⇒ a genuine non-trading day
        return payload

    def _extract_day(self, payload: dict) -> dict[str, dict[str, float]]:
        """Extract EVERY configured ticker's fields from one day's payload in a SINGLE pass (vs the old
        per-(ticker, field) scan of ~8800 rows). Column lookup is BY NAME against the response's own
        ``fields`` array (survives TWSE reordering). A ticker not present that day is simply absent from
        the result; a value cell that will not parse is omitted for that FIELD ONLY (the widened guard —
        bare numbers are kept by _parse_num, genuine junk like JSON null is dropped), and a non-string
        ticker cell is skipped rather than crashing on ``.strip()``."""
        fields = payload.get("fields", [])
        try:
            ticker_idx = fields.index(_TICKER_COLUMN)
        except ValueError:
            return {}
        # Resolve each field to the FIRST of its candidate column names present in THIS payload's
        # schema (T86 changed its columns ~2018 — see _FIELD_COLUMNS). A field whose column is absent
        # in this era is simply omitted for these days, never a crash.
        col_idx: dict[str, int] = {}
        for f, candidates in _FIELD_COLUMNS.items():
            for col in candidates:
                if col in fields:
                    col_idx[f] = fields.index(col)
                    break
        if not col_idx:
            return {}
        wanted = set(self._tickers)
        max_idx = max(ticker_idx, max(col_idx.values()))
        out: dict[str, dict[str, float]] = {}
        for row in payload.get("data", []):
            if len(row) <= max_idx:
                continue
            tk = row[ticker_idx]
            if not isinstance(tk, str) or tk.strip() not in wanted:
                continue
            vals: dict[str, float] = {}
            for f, ci in col_idx.items():
                try:
                    vals[f] = _parse_num(row[ci])
                except (TypeError, ValueError, AttributeError):
                    continue                      # unparseable cell → omit this field only, never crash
            if vals:
                out[tk.strip()] = vals
        return out

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
