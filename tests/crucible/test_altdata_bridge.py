"""Crucible alt-data bridge — accepted connector series become real-mode Panel feature slots.

Proves the sterile real-mode path is fixed: the bridge surveys connectors (reusing the DataScout
accept filter), registers accepted series into the catalog, and returns PIT-safe {terminal: (T,)}
feature slots keyed by their DSL-legal alias — the substrate the overlay proposer (family 'altdata')
scores. A dead source is skipped (never aborts the tick); only accepted series are bridged.
"""
from __future__ import annotations

import numpy as np

from finrl_pro_ds.crucible import DataCatalog
from finrl_pro_ds.crucible.agentic.proposer import LibrarySeedProposer, ProposalContext
from finrl_pro_ds.crucible.data import EdgarConnector, FredConnector, SeriesRef
from finrl_pro_ds.crucible.data.altdata_bridge import (
    ALTDATA_ALIASES,
    bridge_altdata_feature_slots,
    default_connectors,
    resolve_terminal,
)
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.grammar import available_terminals

_BARS = np.arange(np.datetime64("2019-06-01"), np.datetime64("2020-06-01"),
                  np.timedelta64(1, "D")).astype("datetime64[ns]")


def _clean_fred() -> FredConnector:
    """A clean daily macro series over the bar span (passes the scout's quality gate + tripwire)."""
    days = np.arange(np.datetime64("2019-05-01"), np.datetime64("2020-06-01"), np.timedelta64(1, "D"))
    obs = [{"date": str(d), "value": f"{4.0 + 0.001 * i:.4f}"} for i, d in enumerate(days)]
    return FredConnector(transport=lambda url: {"observations": obs},
                         series=(("DGS10", "10Y yield", "D"),), release_lag_days=1)


def _clean_edgar() -> EdgarConnector:
    """A clean fundamentals series whose native id (the ASC 606 revenue tag) is NOT DSL-legal → aliased."""
    eunits = {"units": {"USD": [{"end": "2019-12-31", "val": 200.0, "filed": "2020-02-15",
                                 "form": "10-K"}]}}
    return EdgarConnector(
        transport=lambda u: eunits,
        concepts=(("320193", "RevenueFromContractWithCustomerExcludingAssessedTax", "AAPL rev"),))


class _BrokenConnector:
    """A source whose fetch always fails (bot-wall / rate-limit / missing key) — must be skipped."""

    source_id = "stooq"
    asset_class = "market"

    def discover(self) -> list[SeriesRef]:
        return [SeriesRef("stooq", "spx", "market")]

    def fetch(self, ref, start, end, *, as_of=None):
        raise RuntimeError("HTTP 429 Too Many Requests")

    def provenance(self, ref):
        from finrl_pro_ds.crucible.data import Provenance
        return Provenance("stooq", "u", "lic", "release-lag")


class _LeakyConnector:
    """A PIT-violating source: observations released BEFORE their reference period. The scout's accept
    filter must REJECT it so it never becomes a feature slot (LEAK-2 tripwire at the bridge boundary)."""

    source_id = "leaky"
    asset_class = "macro"

    def discover(self) -> list[SeriesRef]:
        return [SeriesRef("leaky", "bad", "macro")]

    def fetch(self, ref, start, end, *, as_of=None):
        from finrl_pro_ds.crucible.data import SeriesData
        rp = np.array(["2020-02-01"], dtype="datetime64[ns]")
        return SeriesData(ref, rp, np.array([1.0]), rp - np.timedelta64(5, "D"))  # release 5d BEFORE ref

    def provenance(self, ref):
        from finrl_pro_ds.crucible.data import Provenance
        return Provenance("leaky", "u", "lic", "none-rejected")


def test_resolve_terminal_aliases_nonlegal_and_passes_legal() -> None:
    # FRED ids are already DSL-legal → passthrough to source:series.
    assert resolve_terminal(SeriesRef("fred", "T10Y2Y", "macro")) == "fred:T10Y2Y"
    # COT / EDGAR native ids are not legal → mapped to the registered alias. 067651 is WTI crude
    # (was MISLABELED "gold"); real COMEX gold is 088691 (codes verified live 2026-06-23).
    assert resolve_terminal(SeriesRef("cot", "067651:comm_net", "positioning")) == "cot:wti_comm_net"
    assert resolve_terminal(SeriesRef("cot", "088691:comm_net", "positioning")) == "cot:gold_comm_net"
    assert resolve_terminal(SeriesRef(
        "edgar", "0000320193:RevenueFromContractWithCustomerExcludingAssessedTax",
        "fundamental")) == "edgar:aapl_revenue"
    # every alias value is a distinct, DSL-legal terminal.
    from finrl_pro_ds.crucible.data import is_valid_terminal
    assert all(is_valid_terminal(v) for v in ALTDATA_ALIASES.values())
    assert len(set(ALTDATA_ALIASES.values())) == len(ALTDATA_ALIASES)


def test_default_cot_markets_are_orthogonal_and_correctly_labelled() -> None:
    """Doc 3 Part C: the default COT set spans economically-orthogonal underlyings (not one asset's
    views), and 067651 is labelled WTI (not the old 'gold' mislabel)."""
    from finrl_pro_ds.crucible.data.altdata_bridge import (
        COT_TERMINAL_ASSET_CLASS,
        DEFAULT_COT_MARKETS,
        _COT_MARKET_SPEC,
    )
    codes = {c for c, _ in DEFAULT_COT_MARKETS}
    assert {"088691", "067651", "099741", "002602", "043602", "13874A"} <= codes
    # 067651 is WTI energy, 088691 is metal gold — the mislabel is gone.
    assert COT_TERMINAL_ASSET_CLASS["cot:wti_comm_net"] == "energy"
    assert COT_TERMINAL_ASSET_CLASS["cot:gold_comm_net"] == "metal"
    # at least 5 distinct asset classes ⇒ genuinely orthogonal breadth.
    assert len({cls for _c, _s, cls, _n in _COT_MARKET_SPEC}) >= 5


def test_per_source_cap_limits_slots(tmp_path) -> None:
    """max_slots_per_source caps how many slots one source contributes (Doc 3 Part C point 3)."""
    days = np.arange(np.datetime64("2019-05-01"), np.datetime64("2020-06-01"), np.timedelta64(1, "D"))
    obs = [{"date": str(d), "value": f"{4.0 + 0.001 * i:.4f}"} for i, d in enumerate(days)]
    multi = FredConnector(
        transport=lambda url: {"observations": obs}, release_lag_days=1,
        series=(("DGS10", "a", "D"), ("T10Y2Y", "b", "D"), ("VIXCLS", "c", "D")))
    catalog = DataCatalog(tmp_path / "catalog.db")
    slots = bridge_altdata_feature_slots(
        bar_dates=_BARS, start="2019-01-01", end="2020-12-31", catalog=catalog,
        connectors=[multi], max_slots_per_source=2)
    assert len(slots) == 2                       # capped from 3
    catalog.close()


def test_default_connectors_are_the_three_live_sources() -> None:
    conns = default_connectors()
    assert {c.source_id for c in conns} == {"fred", "cot", "edgar"}
    assert {c.asset_class for c in conns} == {"macro", "positioning", "fundamental"}


def test_bridge_builds_accepted_slots_and_registers(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.db")
    slots = bridge_altdata_feature_slots(
        bar_dates=_BARS, start="2019-01-01", end="2020-12-31", catalog=catalog,
        connectors=[_clean_fred(), _clean_edgar()])
    # Both accepted series bridged, keyed by DSL-legal terminal (edgar via alias, fred passthrough).
    assert set(slots) == {"fred:DGS10", "edgar:aapl_revenue"}
    for arr in slots.values():
        assert arr.shape == (_BARS.shape[0],)
    # Stage-1 ACQUIRE: accepted series were registered into the catalog (the 'dirty' signal source).
    registered = {(r["source_id"], r["series"]) for r in catalog.list_series()}
    assert ("fred", "DGS10") in registered
    assert ("edgar", "0000320193:RevenueFromContractWithCustomerExcludingAssessedTax") in registered
    catalog.close()


def test_bridge_skips_failing_source_without_aborting(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.db")
    # A broken source alongside a clean one: the broken one is rejected by the survey and skipped;
    # the clean one still bridges. No exception escapes.
    slots = bridge_altdata_feature_slots(
        bar_dates=_BARS, start="2019-01-01", end="2020-12-31", catalog=catalog,
        connectors=[_BrokenConnector(), _clean_fred()])
    assert set(slots) == {"fred:DGS10"}
    catalog.close()


def test_bridge_rejects_pit_leaking_source(tmp_path) -> None:
    """LEAK-2 tripwire at the bridge boundary: a source whose data is released before its reference
    period is rejected by the accept filter and NEVER becomes a feature slot; the clean source still
    bridges and no exception escapes. Reintroducing a leak here can only be silently bridged if this
    filter is removed AND build_feature_slots' assert_asof_join_causal gate is bypassed."""
    catalog = DataCatalog(tmp_path / "catalog.db")
    slots = bridge_altdata_feature_slots(
        bar_dates=_BARS, start="2019-01-01", end="2020-12-31", catalog=catalog,
        connectors=[_LeakyConnector(), _clean_fred()])
    assert "leaky:bad" not in slots
    assert set(slots) == {"fred:DGS10"}
    # the leaking series was never registered into the catalog either.
    assert ("leaky", "bad") not in {(r["source_id"], r["series"]) for r in catalog.list_series()}
    catalog.close()


def test_bridge_returns_empty_when_nothing_accepted(tmp_path) -> None:
    catalog = DataCatalog(tmp_path / "catalog.db")
    slots = bridge_altdata_feature_slots(
        bar_dates=_BARS, start="2019-01-01", end="2020-12-31", catalog=catalog,
        connectors=[_BrokenConnector()])
    assert slots == {}
    catalog.close()


def test_bridged_slots_drive_altdata_overlay_proposals(tmp_path) -> None:
    """The payoff: bridged terminals reach available_terminals and the proposer emits 'altdata'
    overlays for them — the candidates a price-only real run never produced."""
    catalog = DataCatalog(tmp_path / "catalog.db")
    slots = bridge_altdata_feature_slots(
        bar_dates=_BARS, start="2019-01-01", end="2020-12-31", catalog=catalog,
        connectors=[_clean_fred(), _clean_edgar()])
    t, n = _BARS.shape[0], 3
    rng = np.random.default_rng(0)
    px = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0) + 3.0)
    panel = Panel(_BARS, tuple(f"A{i}" for i in range(n)), px, px * 1.001, px * 0.999, px,
                  rng.uniform(1e6, 1e8, (t, n)), np.ones((t, n), bool), px * 1e6,
                  rng.integers(0, 3, size=n), {"survivorship_free": True}, feature_slots=slots)

    terminals = available_terminals(panel)
    assert "fred:DGS10" in terminals and "edgar:aapl_revenue" in terminals
    ctx = ProposalContext(available_terminals=terminals, max_proposals=64)
    props = LibrarySeedProposer().propose(ctx)
    overlays = [p for p in props if p.candidate_type == "overlay"]
    assert overlays, "bridged feature slots should yield overlay proposals"
    overlay_terms = {t for p in overlays for t in ("fred:DGS10", "edgar:aapl_revenue") if t in p.formula}
    assert overlay_terms == {"fred:DGS10", "edgar:aapl_revenue"}
    catalog.close()
