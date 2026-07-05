"""Taiwan alt-data connector registry — Crucible v2.8.

Mirrors :mod:`.altdata_bridge`'s US/cross-asset registry section (``_COT_MARKET_SPEC`` /
``ALTDATA_ALIASES`` / ``default_connectors()``) for the Taiwan substrate: the connector list + the
DSL-terminal alias map the ``taiwan`` branch of ``_build_substrate`` passes into the already-generic
:func:`altdata_bridge.bridge_altdata_feature_slots`. **No new bridge function is needed** — that
function already accepts injectable ``connectors``/``aliases`` params; this module only supplies
Taiwan-specific values for them.
"""
from __future__ import annotations

from .connector import DataConnector
from .taifex_positioning import TaifexPositioningConnector
from .twse_institutional import TwseInstitutionalConnector

# ticker -> short DSL-safe name, taken from configs/taiwan_cross_asset.yaml's own inline comments
# (TW equity broad / high-dividend / financials / ESG high-div / semiconductors / bond / commodity).
_TICKER_SHORT: tuple[tuple[str, str], ...] = (
    ("0050", "tw50"), ("006208", "ftse_tw"), ("0056", "hidiv"), ("0055", "fin"),
    ("00878", "esg_hidiv"), ("00891", "semi"), ("00679B", "ust20y"),
    ("00751B", "corp_bond"), ("00635U", "gold_etf"), ("00642U", "wti_etf"),
)
_TWSE_INST_FIELDS: tuple[str, ...] = ("foreign_net", "trust_net", "dealer_net", "total_net")
_TAIFEX_CONTRACTS: tuple[str, ...] = ("TX", "TE", "TF")
_TAIFEX_FIELDS: tuple[str, ...] = ("top5_net_pct_oi", "top10_net_pct_oi")

# Every native (source_id, series_id) this registry's connectors can emit is two-colon and/or
# digit-leading (`twse_inst:0050:foreign_net`, `taifex_oi:TX:top5_net_pct_oi`) -- fails
# `panel_bridge.is_valid_terminal`'s single-`:segment` grammar exactly like COT/EDGAR ids do on the
# US side. Aliasing is therefore mandatory here, not optional (test_taiwan_altdata.py enforces both
# legality and completeness so a newly-added field/ticker can't silently ship unaliased).
TAIWAN_ALTDATA_ALIASES: dict[tuple[str, str], str] = {
    **{
        ("twse_inst", f"{ticker}:{field}"): f"twse_inst:{short}_{field}"
        for ticker, short in _TICKER_SHORT
        for field in _TWSE_INST_FIELDS
    },
    **{
        ("taifex_oi", f"{contract}:{field}"): f"taifex_oi:{contract.lower()}_{field}"
        for contract in _TAIFEX_CONTRACTS
        for field in _TAIFEX_FIELDS
    },
}

# Both connectors are `asset_class = "positioning"` (catalog.ASSET_CLASSES) -- this map exists for
# parity with altdata_bridge.COT_TERMINAL_ASSET_CLASS (per-terminal class lookup for pool-diversity
# reporting), even though here every value is the same class.
TAIWAN_TERMINAL_ASSET_CLASS: dict[str, str] = {
    v: "positioning" for v in TAIWAN_ALTDATA_ALIASES.values()
}


def taiwan_connectors() -> list[DataConnector]:
    """The live Taiwan alt-data connectors bridged by default. Unlike FRED/EDGAR on the US side,
    neither needs a credential/API key — T86 and the TAIFEX large-trader OI endpoint are both
    public and keyless."""
    return [TwseInstitutionalConnector(), TaifexPositioningConnector()]
