"""TAIFEX large-trader open-interest connector (Taiwan positioning, concentration) — Crucible v2.8.

Targets the SAME TX/TE/TF contracts already in the Taiwan base book (``taiwan_base_sleeves``) — a
tighter-targeted analog of CFTC COT than COT's own orthogonal-market breadth choice: this asks "is
the crowd on one side of the book we're already trading," on the exact underlyings, not an
economically-adjacent one.

**Architecturally different from every other connector in this package (POLL-ONLY, not
range-queryable) — live-verified 2026-07-04** (session S553-cont-115, see
``.agent/artifacts/crucible_taiwan_breadth_and_scheduling_architecture.md``):
``openapi.taifex.com.tw/v1/OpenInterestOfLargeTradersFutures`` silently IGNORES a ``date``/``Date``
query parameter and always returns only the latest available trading day's snapshot (confirmed: a
full pull's rows carry exactly one distinct ``Date`` regardless of query params). There is no
historical range to request. This connector is therefore **poll-and-accumulate**: each time
``fetch`` actually runs (i.e., each real scheduled tick), it polls the current snapshot and appends
any not-yet-seen ``(contract, date)`` rows to a small persistent JSON store, then answers the
requested ``[start, end]`` window from that accumulated store — mirroring
:mod:`finrl_pro_ds.data.taiwan_panel_loader`'s own cache/manifest pattern, not a new pattern invented
here.

**Honest cold start**: on first deployment the store has ~1 day; it only becomes a useful feature
slot after weeks of the scheduled job actually running, and it does NOT retroactively enrich the
panel's 2010+ OHLCV history — it only grows forward from whenever it is first polled. This is a
disclosed, real limitation of a snapshot-only public endpoint, not a bug.

Field semantics (live-verified): each row carries ``Contract`` (exactly ``"TX"``/``"TE"``/``"TF"`` —
matches ``configs/taiwan_cross_asset.yaml`` with zero code-level mapping needed), ``SettlementMonth``
(an all-months-combined sentinel ``"999912"`` alongside per-month rows — this connector uses ONLY the
sentinel row, the standard "whole book" reading), and ``TypeOfTraders`` (``"0"``=all traders,
``"1"``=a specific institutional sub-category, confirmed live to be a numeric subset of ``"0"`` — this
connector uses ONLY ``"0"``). Numbers are plain digit strings (no thousands-separator, unlike TWSE's
legacy gateway) but are parsed through the same comma-tolerant helper for consistency/safety.

Release-timestamp policy: same reasoning as :mod:`.twse_institutional` (ADR-A2) — a flat
``+1 CALENDAR day`` lag on top of whatever date is in the store, never a business-day count.

**Math-audit note (S553-cont-115):** ``top5_net_pct_oi``/``top10_net_pct_oi`` difference TWO
DIFFERENT, generally disjoint trader groups (the biggest-LONG holders vs the biggest-SHORT holders —
TAIFEX's own methodology reports each side's top N separately), unlike CFTC COT's ``comm_net_pct_oi``
(one cohort's own long minus short). It is a standard, valid large-trader concentration/skew
indicator with the same sign convention as COT (positive = book skewed long) — but economically it is
a "biggest-longs-vs-biggest-shorts skew," not "the top N traders' net position."
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .connector import Provenance, SeriesData, SeriesRef

logger = logging.getLogger(__name__)

_BASE = "https://openapi.taifex.com.tw/v1/OpenInterestOfLargeTradersFutures"
_LICENSE = "TAIFEX (Taiwan Futures Exchange) OpenAPI — public data, keyless"
_ALL_MONTHS_SENTINEL = "999912"     # live-verified all-settlement-months-combined row
_ALL_TRADERS = "0"                  # live-verified: "0"=all traders, "1" subset = specific institutions

ROOT = Path(__file__).resolve().parents[3]
# Audit note (S553-cont-115): `/data/` is repo-gitignored, so a test that injects `transport=` but
# forgets `store_path=` (this connector's SECOND injectable seam, unlike every other stateless
# connector) writes a stray local file here rather than risking an accidental commit -- contained,
# not zero-cost (still pollutes a real path with test data), so tests should inject `store_path=`.
_DEFAULT_STORE_PATH = ROOT / "data" / "raw" / "taifex_large_trader_oi" / "large_trader_oi.json"

_FIELDS: tuple[str, ...] = ("top5_net_pct_oi", "top10_net_pct_oi")
# Math-audit note (S553-cont-115): unlike CFTC COT's `comm_net_pct_oi` (one cohort's own long minus
# short), these fields difference TWO DIFFERENT, generally disjoint trader groups — the biggest-LONG
# holders vs the biggest-SHORT holders (TAIFEX's own large-trader methodology reports each side's top
# N separately). It is a standard, valid concentration/skew indicator (positive = the large-trader
# book is skewed long, same sign convention as COT) — but it is a "biggest-longs-vs-biggest-shorts
# skew," NOT "the top N traders' net position" the way the field name alone might suggest.


def _parse_num(s: str) -> float:
    """Defensive: plain digit strings on this endpoint, but tolerate commas for safety/consistency
    with the TWSE legacy-gateway connector's parsing convention."""
    return float(str(s).replace(",", ""))


def _parse_yyyymmdd(s: str) -> np.datetime64:
    if len(s) == 8 and s.isdigit():
        return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:]}", "ns")
    return np.datetime64(s, "ns")


def _default_transport(url: str) -> list[dict]:
    import urllib.request
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:      # noqa: S310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


class _AccumulationStore:
    """A tiny, idempotent JSON key-value store — ``"<contract>|<date>" -> {field: value}``. Not a
    general-purpose store; scoped to exactly what this connector needs (a handful of contracts, one
    row/day)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, float | None]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("taifex_oi: store at %s unreadable — treating as empty", self.path)
            return {}

    def save(self, data: dict[str, dict[str, float | None]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class TaifexPositioningConnector:
    """DataConnector for TAIFEX large-trader OI concentration. ``asset_class = 'positioning'``.
    series_id = ``<contract>:<field>``, field in ``{top5_net_pct_oi, top10_net_pct_oi}``.

    **Poll-and-accumulate, not stateless range-query** (module docstring) — ``fetch`` has the side
    effect of polling the live endpoint once per connector instance (cached thereafter) and merging
    any new day into ``store_path``, which persists across process runs."""

    source_id = "taifex_oi"
    asset_class = "positioning"

    def __init__(
        self,
        *,
        transport: Callable[[str], list[dict]] | None = None,
        contracts: tuple[str, ...] = ("TX", "TE", "TF"),
        release_lag_days: int = 1,
        store_path: str | Path | None = None,
    ) -> None:
        self._transport = transport
        self._contracts = contracts
        self._release_lag = np.timedelta64(int(release_lag_days), "D")
        self._store = _AccumulationStore(store_path if store_path is not None
                                         else _DEFAULT_STORE_PATH)
        self._polled = False        # per-instance: poll the live endpoint at most once

    # -- interface -----------------------------------------------------------------
    def discover(self) -> list[SeriesRef]:
        return [
            SeriesRef(self.source_id, f"{contract}:{field}", self.asset_class,
                      title=f"{contract} large-trader {field}", frequency="D")
            for contract in self._contracts
            for field in _FIELDS
        ]

    def provenance(self, ref: SeriesRef) -> Provenance:
        return Provenance(
            source_id=self.source_id,
            url=_BASE,
            license=_LICENSE,
            # Not a pure lag model — see module docstring: poll-and-accumulate, THEN a release lag on
            # top of whatever calendar day the store has recorded (CR-4).
            as_of_policy="poll-accumulate-release-lag",
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
        contract, field = self._split(ref.series_id)
        self._poll_and_merge_once()
        store = self._store.load()
        cutoff = np.datetime64(as_of, "ns") if as_of is not None else None
        lo = np.datetime64(np.datetime64(start, "D"), "ns")
        hi = np.datetime64(np.datetime64(end, "D"), "ns")

        refs: list[np.datetime64] = []
        vals: list[float] = []
        rels: list[np.datetime64] = []
        for key in sorted(store):
            c, _, date_str = key.partition("|")
            if c != contract:
                continue
            value = store[key].get(field)
            if value is None:
                continue
            reference = _parse_yyyymmdd(date_str)          # always a midnight-ns timestamp
            if reference < lo or reference > hi:            # both sides midnight-ns -> plain compare
                continue
            release = reference + self._release_lag
            if cutoff is not None and release > cutoff:
                continue
            refs.append(reference)
            vals.append(float(value))
            rels.append(release)

        return SeriesData(
            ref=ref,
            reference_period=np.array(refs, dtype="datetime64[ns]"),
            value=np.array(vals, dtype=np.float64),
            release_timestamp=np.array(rels, dtype="datetime64[ns]"),
            provenance=self.provenance(ref),
            meta={"n_accumulated_days_this_contract": sum(
                1 for k in store if k.partition("|")[0] == contract)},
        )

    # -- internals -----------------------------------------------------------------
    @staticmethod
    def _split(series_id: str) -> tuple[str, str]:
        contract, _, field = series_id.partition(":")
        if field not in _FIELDS:
            raise ValueError(f"unknown TAIFEX positioning field {field!r}; supported: {_FIELDS}")
        return contract, field

    def _poll_and_merge_once(self) -> None:
        """Poll the live snapshot at most once per connector instance (every SeriesRef this
        instance's ``discover()`` yields shares one HTTP call, mirroring the T86 connector's
        per-instance day-cache), and merge any not-yet-seen ``(contract, date)`` rows into the
        persistent store. A transport failure degrades to "use the store as-is" — never fatal,
        matching every other connector's fail-closed-but-not-crash policy."""
        if self._polled:
            return
        self._polled = True
        transport = self._transport or _default_transport
        try:
            rows = transport(_BASE)
        except Exception as exc:                  # noqa: BLE001 - transport failed closed
            logger.warning("taifex_oi: live poll failed (%r) — answering from the store as-is", exc)
            return
        if not isinstance(rows, list):
            logger.warning("taifex_oi: unexpected payload shape %r — ignored", type(rows))
            return
        self._merge_into_store(rows)

    def _merge_into_store(self, rows: list[dict]) -> None:
        store = self._store.load()
        changed = False
        for row in rows:
            if (row.get("SettlementMonth") != _ALL_MONTHS_SENTINEL
                    or row.get("TypeOfTraders") != _ALL_TRADERS):
                continue
            contract = row.get("Contract")
            date_str = row.get("Date")
            if contract not in self._contracts or not date_str:
                continue
            key = f"{contract}|{date_str}"
            if key in store:
                continue                           # idempotent — this day is already recorded
            try:
                top5b, top5s = _parse_num(row["Top5Buy"]), _parse_num(row["Top5Sell"])
                top10b, top10s = _parse_num(row["Top10Buy"]), _parse_num(row["Top10Sell"])
                oi = _parse_num(row["OIOfMarket"])
            except (KeyError, TypeError, ValueError):
                continue
            store[key] = {
                "top5_net_pct_oi": (top5b - top5s) / oi if oi else None,
                "top10_net_pct_oi": (top10b - top10s) / oi if oi else None,
            }
            changed = True
        if changed:
            self._store.save(store)
