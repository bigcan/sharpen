"""Bridge accepted alt-data connector series into Panel feature slots (real-mode substrate).

Closes the gap that made the real-mode orchestrator sterile: it loaded a *price-only* panel, so the
DataScout's macro/positioning/fundamental series never became DSL terminals, and the
``LibrarySeedProposer`` OVERLAY path (family ``altdata``) emitted nothing — every real tick mined only
the fixed cross-sectional WQ101 bank. This module turns accepted connector series into
``{terminal: (T,) array}`` feature slots ready to drop into ``Panel(feature_slots=...)``, so
``available_terminals`` exposes them and the overlay proposer finally scores them against the book.

Pipeline (all sanctioned primitives — no new PIT logic):

  1. :class:`DataScout` ``survey`` → the ACCEPT FILTER: data-quality gate + the exhaustive
     as-of-join reconstruction tripwire (CR-4/LEAK-2). Only ACCEPTED series are bridged — a source
     the data crucible rejects (outliers, PIT leak, empty) never reaches the funnel.
  2. ``register_accepted`` → Stage-1 ACQUIRE: the accepted series land in the catalog, which is the
     substrate's ``snapshot_hash`` (the "dirty" signal the orchestrator gates mining on).
  3. :func:`panel_bridge.build_feature_slots` per accepted series → release-time ``asof_join`` +
     ``assert_feature_causal`` (a HARD PIT gate). A look-ahead **raises** — it is never a skip.

Failure policy (nightly-safe): a *transport* failure on one series (bot-wall, rate-limit, missing
``FRED_API_KEY`` / ``SEC_EDGAR_UA``) degrades to a skip+warn so one dead source can't abort the tick.
A *PIT leak* (``AssertionError`` from the causal gate) is NOT swallowed — it propagates.

CR-1/CR-6 boundary: this layer only widens the terminal set; it never reads a gate value, a verdict,
or the trial ledger.
"""
from __future__ import annotations

import logging

import numpy as np

from ..agentic.scout import DataScout
from ..catalog import DataCatalog
from .cftc_cot import CftcCotConnector
from .connector import DataConnector, SeriesRef
from .edgar import EdgarConnector
from .fred import FredConnector
from .panel_bridge import SlotRequest, build_feature_slots

log = logging.getLogger("crucible.altdata_bridge")

# COT markets to pull, spanning ECONOMICALLY-ORTHOGONAL underlyings (Doc 3 Part C — breadth is the
# strongest driver of return-stream de-correlation; the old single-market default gave 3 mutually
# correlated views of ONE asset). One deep-open-interest contract per asset class. All codes VERIFIED
# LIVE against publicreporting.cftc.gov (report 2026-06-23) — the previous default `067651` was
# MISLABELED "GOLD": it is WTI crude (WTI-PHYSICAL, NYMEX). Real COMEX gold is `088691`.
# (code, short_name, asset_class, official CFTC market_and_exchange_names).
_COT_MARKET_SPEC: tuple[tuple[str, str, str, str], ...] = (
    ("088691", "gold",   "metal",  "GOLD - COMMODITY EXCHANGE INC."),
    ("067651", "wti",    "energy", "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE"),
    ("099741", "eurofx", "fx",     "EURO FX - CHICAGO MERCANTILE EXCHANGE"),
    ("002602", "corn",   "ag",     "CORN - CHICAGO BOARD OF TRADE"),
    ("043602", "ust10y", "rate",   "UST 10Y NOTE - CHICAGO BOARD OF TRADE"),
    ("13874A", "spx",    "equity", "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"),
)
DEFAULT_COT_MARKETS: tuple[tuple[str, str], ...] = tuple(
    (code, name) for code, _short, _cls, name in _COT_MARKET_SPEC)

# COT field (cftc_cot._FIELDS key) → DSL-legal terminal suffix. Mirrors cftc_cot._FIELDS; kept here
# because that map is private to the connector.
_COT_FIELD_SUFFIX: dict[str, str] = {
    "comm_net": "comm_net", "noncomm_net": "noncomm_net", "comm_net_pct_oi": "comm_pct_oi"}

# DSL-legal terminal aliases for series whose native ``source:series_id`` is NOT a legal terminal
# (digit-leading and/or two colons — e.g. ``cot:067651:comm_net``). FRED ids are already legal
# (``fred:T10Y2Y``), so they need no alias and fall through to ``ref.terminal``. Grammar:
# ``[A-Za-z_]\w*(?::[A-Za-z_]\w*)?`` (panel_bridge._TERMINAL_RE). COT aliases are generated across the
# orthogonal market set × fields; EDGAR aliases are explicit (two filers × two concepts).
_COT_ALIASES: dict[tuple[str, str], str] = {
    ("cot", f"{code}:{field}"): f"cot:{short}_{suffix}"
    for code, short, _cls, _name in _COT_MARKET_SPEC
    for field, suffix in _COT_FIELD_SUFFIX.items()
}
ALTDATA_ALIASES: dict[tuple[str, str], str] = {
    **_COT_ALIASES,
    # EDGAR revenue is the ASC 606 tag, not legacy us-gaap:Revenues (which is empty post-2018 for both
    # filers — see edgar._DEFAULT_CONCEPTS). NetIncomeLoss is the same across eras.
    ("edgar", "0000320193:RevenueFromContractWithCustomerExcludingAssessedTax"): "edgar:aapl_revenue",
    ("edgar", "0000320193:NetIncomeLoss"): "edgar:aapl_netincome",
    ("edgar", "0000789019:RevenueFromContractWithCustomerExcludingAssessedTax"): "edgar:msft_revenue",
    ("edgar", "0000789019:NetIncomeLoss"): "edgar:msft_netincome",
}

# Per-terminal asset class (for the pool-diversity report + future source-aware diversity, Doc 3
# Part D.1). FRED = macro, EDGAR = fundamental; COT carries the underlying's class, not just
# "positioning", so orthogonality logic sees gold-metal vs corn-ag vs ust10y-rate.
COT_TERMINAL_ASSET_CLASS: dict[str, str] = {
    f"cot:{short}_{suffix}": cls
    for code, short, cls, _name in _COT_MARKET_SPEC
    for field, suffix in _COT_FIELD_SUFFIX.items()
}


def default_connectors() -> list[DataConnector]:
    """The live free-data connectors bridged by default. FRED reads ``FRED_API_KEY`` and EDGAR reads
    ``SEC_EDGAR_UA`` from the environment; a missing credential makes that source fail its fetch, so
    the scout rejects it and it is simply skipped (macro/positioning still bridge)."""
    return [
        FredConnector(),
        CftcCotConnector(markets=DEFAULT_COT_MARKETS),
        EdgarConnector(),
    ]


def resolve_terminal(ref: SeriesRef, aliases: dict[tuple[str, str], str] | None = None) -> str:
    """The DSL-legal feature-slot key for ``ref`` — its alias if one is registered, else the native
    ``source:series`` terminal (correct for FRED)."""
    aliases = ALTDATA_ALIASES if aliases is None else aliases
    return aliases.get((ref.source_id, ref.series_id), ref.terminal)


def bridge_altdata_feature_slots(
    *,
    bar_dates: np.ndarray,
    start: np.datetime64 | str,
    end: np.datetime64 | str,
    catalog: DataCatalog | None = None,
    connectors: list[DataConnector] | None = None,
    aliases: dict[tuple[str, str], str] | None = None,
    max_slots_per_source: int | None = None,
) -> dict[str, np.ndarray]:
    """Survey ``connectors``, register accepted series into ``catalog``, and return the PIT-safe
    ``{terminal: (T,) array}`` feature slots for the ACCEPTED series only, joined onto ``bar_dates``.

    A transport failure on a single accepted series (rare — it fetched during the survey) is skipped
    with a warning; a PIT leak raises. Returns ``{}`` when nothing is accepted (e.g. all sources
    down / no credentials) — the caller then runs on the price-only panel unchanged.

    ``max_slots_per_source`` (Doc 3 Part C point 3) caps how many feature slots any ONE source
    contributes, so a prolific connector (e.g. dozens of FRED series or many COT market×field combos)
    cannot dominate the pool and re-create the concentration problem breadth is meant to fix. Slots
    are kept in each connector's ``discover()`` order until the cap; the rest are logged and skipped.
    ``None`` (default) = no cap (back-compat).
    """
    connectors = default_connectors() if connectors is None else connectors
    aliases = ALTDATA_ALIASES if aliases is None else aliases

    # 1) survey = the accept filter (quality gate + as-of-join tripwire) + 2) register accepted.
    scout = DataScout(connectors)
    report = scout.survey(start, end, bar_dates)
    if catalog is not None:
        scout.register_accepted(report, catalog)
    accepted = {(f.source_id, f.series_id) for f in report.accepted()}
    rejected = [(f.source_id, f.series_id, "; ".join(f.reasons)) for f in report.rejected()]
    log.info("altdata bridge: %d/%d series accepted", len(accepted), len(report.findings))
    for src, sid, why in rejected:
        log.info("altdata bridge: rejected %s:%s (%s)", src, sid, why)

    # 3) build PIT-gated feature slots for the accepted series only (build_feature_slots enforces
    # assert_feature_causal — a look-ahead HARD-fails). One request at a time so a lone re-fetch
    # failure skips that series instead of aborting the whole batch.
    slots: dict[str, np.ndarray] = {}
    for connector in connectors:
        n_from_source = 0
        for ref in connector.discover():
            if (ref.source_id, ref.series_id) not in accepted:
                continue
            if max_slots_per_source is not None and n_from_source >= max_slots_per_source:
                log.info("altdata bridge: %s hit per-source cap %d — remaining slots skipped",
                         connector.source_id, max_slots_per_source)
                break
            terminal = resolve_terminal(ref, aliases)
            req = SlotRequest(connector=connector, ref=ref, terminal=terminal)
            try:
                built = build_feature_slots([req], bar_dates, start=start, end=end, catalog=None)
            except AssertionError:
                raise  # PIT leak — never silently skipped
            except Exception as exc:  # noqa: BLE001 - transport failed closed; degrade to skip
                log.warning("altdata bridge: %s skipped on re-fetch (%r)", terminal, exc)
                continue
            slots.update(built)
            n_from_source += len(built)

    log.info("altdata bridge: %d feature slots bridged -> %s", len(slots), sorted(slots))
    return slots
