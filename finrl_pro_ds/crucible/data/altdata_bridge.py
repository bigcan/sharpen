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

# COT markets to pull (CFTC contract code, human name). The connector defaults to none — the operator
# supplies contracts of interest. Gold (067651) is the positioning series with the deepest history.
DEFAULT_COT_MARKETS: tuple[tuple[str, str], ...] = (("067651", "GOLD - COMMODITY EXCHANGE"),)

# DSL-legal terminal aliases for series whose native ``source:series_id`` is NOT a legal terminal
# (digit-leading and/or two colons — e.g. ``cot:067651:comm_net``). FRED ids are already legal
# (``fred:T10Y2Y``), so they need no alias and fall through to ``ref.terminal``. Grammar:
# ``[A-Za-z_]\w*(?::[A-Za-z_]\w*)?`` (panel_bridge._TERMINAL_RE).
ALTDATA_ALIASES: dict[tuple[str, str], str] = {
    ("cot", "067651:comm_net"): "cot:gold_comm_net",
    ("cot", "067651:noncomm_net"): "cot:gold_noncomm_net",
    ("cot", "067651:comm_net_pct_oi"): "cot:gold_comm_pct_oi",
    ("edgar", "0000320193:Revenues"): "edgar:aapl_revenue",
    ("edgar", "0000320193:NetIncomeLoss"): "edgar:aapl_netincome",
    ("edgar", "0000789019:Revenues"): "edgar:msft_revenue",
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
) -> dict[str, np.ndarray]:
    """Survey ``connectors``, register accepted series into ``catalog``, and return the PIT-safe
    ``{terminal: (T,) array}`` feature slots for the ACCEPTED series only, joined onto ``bar_dates``.

    A transport failure on a single accepted series (rare — it fetched during the survey) is skipped
    with a warning; a PIT leak raises. Returns ``{}`` when nothing is accepted (e.g. all sources
    down / no credentials) — the caller then runs on the price-only panel unchanged.
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
        for ref in connector.discover():
            if (ref.source_id, ref.series_id) not in accepted:
                continue
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

    log.info("altdata bridge: %d feature slots bridged -> %s", len(slots), sorted(slots))
    return slots
