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


def test_terminal_asset_class_map_matches_aliases() -> None:
    assert set(TAIWAN_TERMINAL_ASSET_CLASS) == set(TAIWAN_ALTDATA_ALIASES.values())
    assert set(TAIWAN_TERMINAL_ASSET_CLASS.values()) == {"positioning"}


def test_taiwan_connectors_factory_returns_expected_types() -> None:
    connectors = taiwan_connectors()
    kinds = {type(c) for c in connectors}
    assert kinds == {TwseInstitutionalConnector, TaifexPositioningConnector}
