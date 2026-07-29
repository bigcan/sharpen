"""Taiwan alt-data connector registry — Crucible v2.8.

Mirrors :mod:`.altdata_bridge`'s US/cross-asset registry section (``_COT_MARKET_SPEC`` /
``ALTDATA_ALIASES`` / ``default_connectors()``) for the Taiwan substrate: the connector list + the
DSL-terminal alias map the ``taiwan`` branch of ``_build_substrate`` passes into the already-generic
:func:`altdata_bridge.bridge_altdata_feature_slots`. **No new bridge function is needed** — that
function already accepts injectable ``connectors``/``aliases`` params; this module only supplies
Taiwan-specific values for them.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..catalog import DataCatalog
from .connector import DataConnector
from .panel_bridge import SlotRequest, build_panel_feature_slot
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


# ------------------------------------------------------- PER-NAME (T,N) slots (audit U3a) ----
# T86 is already per-stock: `TwseInstitutionalConnector.series_id` is "<ticker>:<field>". But the
# alias map above flattens each (ticker, field) pair into its OWN BROADCAST terminal
# (`twse_inst:tw50_foreign_net`, `twse_inst:hidiv_foreign_net`, ... 10 tickers x 4 fields = 40
# constant-across-the-cross-section series). That is the shape the overlay path wants, and for a long
# time it was the only shape the bridge could produce — which is why genuinely per-name data could
# only ever act as a market-timing overlay (audit RC-4).
#
# This assembles the SAME observations the other way up: one `(T, N)` matrix per FIELD, columns
# aligned to `Panel.tickers`, so `rank(twse_inst:foreign_net)` means "rank names by today's foreign
# institutional net flow" — a cross-sectional characteristic. `grammar.cross_sectional_terminals`
# admits it (crucible-v7.1); the broadcast terminals stay excluded there, as they must.
#
# ADDITIVE, not a replacement: the 40 broadcast terminals are untouched, so the overlay search is
# unchanged. The same underlying observations therefore appear in BOTH shapes, which is a deliberate
# duplication with one consequence worth naming — the overlay terminal pool grows by 4, and those 4
# collapse under `nanmean` to the EQUAL-WEIGHTED cross-sectional average flow, which is a different
# series from any single-ticker broadcast slot rather than a copy of one.
_PER_NAME_TERMINALS: tuple[str, ...] = tuple(f"twse_inst:{f}" for f in _TWSE_INST_FIELDS)

TAIWAN_TERMINAL_ASSET_CLASS.update({t: "positioning" for t in _PER_NAME_TERMINALS})


def taiwan_per_name_slots(
    tickers: Sequence[str],
    bar_dates: np.ndarray,
    *,
    start: np.datetime64 | str,
    end: np.datetime64 | str,
    connector: TwseInstitutionalConnector | None = None,
    catalog: DataCatalog | None = None,
    fields: Sequence[str] = _TWSE_INST_FIELDS,
) -> dict[str, np.ndarray]:
    """Assemble TWSE T86 into PER-NAME ``(T, N)`` feature slots — one per field (audit U3a).

    ``tickers`` MUST be ``Panel.tickers`` (same object/order): ``build_panel_feature_slot`` aligns
    columns positionally against it, and a name with no T86 series becomes an ALL-NaN column rather
    than a zero. The Taiwan panel's tickers are the raw listing codes (``"0050"``, ``"006208"``, …),
    which is exactly the key ``TwseInstitutionalConnector`` uses, so alignment is by construction —
    but it is a silent-failure surface (a mismatch yields an all-NaN matrix, not an error), so
    :func:`taiwan_per_name_coverage` is provided to assert it and the caller logs the result.

    ``connector`` should be the SAME instance the broadcast bridge uses: the connector caches its
    per-day T86 payloads for the life of the instance, so sharing it makes this assembly nearly free
    instead of re-polling every trading day a second time.

    ``catalog`` defaults to None on purpose. The broadcast path already registers every one of these
    series, so registering them again here would double-count them in the catalog that feeds
    pool-diversity reporting and the data snapshot hash."""
    conn = connector if connector is not None else TwseInstitutionalConnector()
    refs = {r.series_id: r for r in conn.discover()}
    out: dict[str, np.ndarray] = {}
    for field in fields:
        per_ticker = {t: SlotRequest(conn, refs[f"{t}:{field}"])
                      for t in tickers if f"{t}:{field}" in refs}
        if not per_ticker:
            continue
        out.update(build_panel_feature_slot(
            per_ticker, list(tickers), bar_dates, terminal=f"twse_inst:{field}",
            start=start, end=end, catalog=catalog))
    return out


def taiwan_per_name_coverage(tickers: Sequence[str],
                            connector: TwseInstitutionalConnector | None = None,
                            fields: Sequence[str] = _TWSE_INST_FIELDS) -> dict[str, int]:
    """``{field: n_tickers_with_a_T86_series}`` for ``tickers`` — the pre-flight alignment check.

    Column alignment fails SILENTLY (an unmatched ticker is an all-NaN column, by design, because
    absent data must not read as 0.0), so a ticker-naming mismatch would produce a perfectly
    well-formed, entirely empty `(T,N)` matrix. Call this and look at the numbers before trusting a
    per-name slot."""
    conn = connector if connector is not None else TwseInstitutionalConnector()
    ids = {r.series_id for r in conn.discover()}
    return {f: sum(1 for t in tickers if f"{t}:{f}" in ids) for f in fields}
