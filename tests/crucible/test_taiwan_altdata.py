"""Taiwan alt-data connector registry — Crucible v2.8."""
from __future__ import annotations

from finrl_pro_ds.crucible.data.panel_bridge import is_valid_terminal
from finrl_pro_ds.crucible.data.taifex_positioning import TaifexPositioningConnector
from finrl_pro_ds.crucible.data.taiwan_altdata import (
    TAIWAN_ALTDATA_ALIASES,
    TAIWAN_TERMINAL_ASSET_CLASS,
    taiwan_connectors,
)
from finrl_pro_ds.crucible.data.twse_institutional import TwseInstitutionalConnector


def test_every_alias_value_is_dsl_legal() -> None:
    for terminal in TAIWAN_ALTDATA_ALIASES.values():
        assert is_valid_terminal(terminal), f"{terminal!r} is not a DSL-legal terminal"


def test_no_alias_value_collisions() -> None:
    values = list(TAIWAN_ALTDATA_ALIASES.values())
    assert len(values) == len(set(values)), "two different native ids alias to the same terminal"


def test_alias_map_covers_every_series_every_connector_discovers() -> None:
    """Completeness: a newly-added ticker/contract/field that forgets its alias would otherwise only
    fail deep inside a real run's `build_feature_slots` call (ValueError) -- catch it here instead."""
    for connector in taiwan_connectors():
        for ref in connector.discover():
            key = (ref.source_id, ref.series_id)
            assert key in TAIWAN_ALTDATA_ALIASES, f"{key} has no Taiwan alias registered"


def test_terminal_asset_class_map_covers_every_terminal_exactly() -> None:
    """The map exists so pool-diversity reporting can look up any terminal's asset class, so its
    contract is "every terminal the Taiwan branch can emit, and nothing else".

    WIDENED for U3a (2026-07-30): the branch now emits two KINDS of terminal — the aliased BROADCAST
    ones (`twse_inst:tw50_foreign_net`, one per ticker x field) and the ASSEMBLED per-name `(T,N)` ones
    (`twse_inst:foreign_net`, one per field, columns = Panel.tickers). The per-name terminals are not
    alias values — nothing aliases to them, they are built by `taiwan_per_name_slots` — so the old
    equality against the alias values alone could not hold once they existed. Kept as an EXACT equality
    against the union so a newly-added terminal of either kind still cannot ship unclassified."""
    from finrl_pro_ds.crucible.data.taiwan_altdata import _PER_NAME_TERMINALS

    expected = set(TAIWAN_ALTDATA_ALIASES.values()) | set(_PER_NAME_TERMINALS)
    assert set(TAIWAN_TERMINAL_ASSET_CLASS) == expected
    assert set(TAIWAN_TERMINAL_ASSET_CLASS.values()) == {"positioning"}
    # the two kinds must stay disjoint: one terminal cannot be both a broadcast series and a matrix
    assert not (set(TAIWAN_ALTDATA_ALIASES.values()) & set(_PER_NAME_TERMINALS))


def test_taiwan_connectors_factory_returns_expected_types() -> None:
    connectors = taiwan_connectors()
    kinds = {type(c) for c in connectors}
    assert kinds == {TwseInstitutionalConnector, TaifexPositioningConnector}
