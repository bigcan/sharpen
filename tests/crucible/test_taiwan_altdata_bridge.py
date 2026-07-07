"""Integration: Taiwan connectors -> bridge_altdata_feature_slots -> DSL-addressable Panel terminals.

This is the exact call wired into `scripts/research/crucible_orchestrator.py::_build_substrate`'s
`taiwan` branch (`bridge_altdata_feature_slots(connectors=taiwan_connectors(), aliases=
TAIWAN_ALTDATA_ALIASES, ...)`) — verified here against a synthetic Panel + injected transports so it
doesn't require the real FinMind-backed `load_taiwan_panel` or live network access. `_build_substrate`
itself is not unit-tested directly (it would need a fake argparse.Namespace AND either real network or
deep mocking of `load_taiwan_panel`/`taiwan_base_sleeves` for low marginal value over this test, which
exercises the actual new logic: the bridge + registry wiring).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from finrl_pro_ds.crucible.data.altdata_bridge import bridge_altdata_feature_slots
from finrl_pro_ds.crucible.data.taifex_positioning import TaifexPositioningConnector
from finrl_pro_ds.crucible.data.taiwan_altdata import TAIWAN_ALTDATA_ALIASES
from finrl_pro_ds.crucible.data.twse_institutional import TwseInstitutionalConnector
from finrl_pro_ds.signals.features import make_synthetic_panel
from finrl_pro_ds.signals.generation.grammar import available_terminals


def _t86_transport(date: str, ticker: str):
    """Only responds for the exact requested `date` (every other day looks like a non-trading day) —
    catches a bug class where the connector's per-day loop ignores which day it's actually fetching."""
    fields = ["證券代號", "外陸資買賣超股數(不含外資自營商)", "投信買賣超股數",
             "自營商買賣超股數", "三大法人買賣超股數"]

    def transport(url: str) -> dict:
        requested = url.split("date=")[1].split("&")[0]
        if requested != date:
            return {"stat": "很抱歉，沒有符合條件的資料!"}
        return {"stat": "OK", "date": date, "title": "x", "fields": fields,
                "data": [[ticker, "1000", "500", "-200", "1300"]]}
    return transport


def _taifex_transport(date: str):
    def transport(url: str) -> list[dict]:
        return [{"Date": date, "Contract": "TX", "ContractName": "x", "SettlementMonth": "999912",
                 "TypeOfTraders": "0", "Top5Buy": "100", "Top5Sell": "80", "Top10Buy": "150",
                 "Top10Sell": "140", "OIOfMarket": "1000"}]
    return transport


def test_taiwan_connectors_bridge_and_are_dsl_addressable(tmp_path) -> None:
    panel = make_synthetic_panel(T=60, N=5, feature_slots={})
    bar_dates = panel.dates
    ref_date = "2010-01-20"       # well within the synthetic panel's 2010-01-04 + 60d window

    connectors = [
        TwseInstitutionalConnector(transport=_t86_transport(ref_date.replace("-", ""), "0050"),
                                   tickers=("0050",), sleep_seconds=0,
                                   store_path=tmp_path / "t86.json"),
        TaifexPositioningConnector(transport=_taifex_transport(ref_date.replace("-", "")),
                                   contracts=("TX",), store_path=tmp_path / "store.json"),
    ]
    slots = bridge_altdata_feature_slots(
        bar_dates=bar_dates, start="2010-01-01", end="2010-03-01",
        connectors=connectors, aliases=TAIWAN_ALTDATA_ALIASES)

    assert "twse_inst:tw50_foreign_net" in slots
    assert "twse_inst:tw50_total_net" in slots
    assert "taifex_oi:tx_top5_net_pct_oi" in slots
    assert slots["taifex_oi:tx_top5_net_pct_oi"][
        np.where(bar_dates >= np.datetime64(ref_date) + np.timedelta64(1, "D"))[0][0]
    ] == (100 - 80) / 1000

    enriched = dataclasses.replace(panel, feature_slots={**panel.feature_slots, **slots})
    terminals = available_terminals(enriched)
    assert "twse_inst:tw50_foreign_net" in terminals
    assert "taifex_oi:tx_top5_net_pct_oi" in terminals


def test_no_altdata_slots_degrades_gracefully_when_all_sources_fail(tmp_path) -> None:
    """Mirrors bridge_altdata_feature_slots' own documented failure policy: a transport failure
    degrades to a skip, never an abort -- the taiwan branch still returns a usable (price-only)
    panel."""
    panel = make_synthetic_panel(T=60, N=5, feature_slots={})

    def failing(url: str):
        raise RuntimeError("network down")

    connectors = [
        TwseInstitutionalConnector(transport=failing, tickers=("0050",), sleep_seconds=0,
                                   store_path=tmp_path / "t86.json"),
        TaifexPositioningConnector(transport=failing, contracts=("TX",),
                                   store_path=tmp_path / "store.json"),
    ]
    slots = bridge_altdata_feature_slots(
        bar_dates=panel.dates, start="2010-01-01", end="2010-03-01",
        connectors=connectors, aliases=TAIWAN_ALTDATA_ALIASES)
    assert slots == {}
