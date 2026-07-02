"""Crucible P1b→P1a bridge — connector series become Panel feature slots addressable by the DSL.

Proves the loop closes: build_feature_slots fetches fixture connectors, as-of-joins onto a bar
calendar, and yields a {terminal: (T,) array} dict; a Panel built with it resolves the terminal
through eval_on_panel (the P1a grammar registry), broadcasting the (T,) macro series to a constant
(T,N) matrix — exactly the substrate the overlay/conditioner path scores.
"""
from __future__ import annotations

import numpy as np
import pytest

from finrl_pro_ds.crucible.data import (
    CftcCotConnector,
    FredConnector,
    SeriesRef,
    SlotRequest,
    build_feature_slots,
    is_valid_terminal,
)
from finrl_pro_ds.signals.features import Panel
from finrl_pro_ds.signals.generation.dsl_signal import eval_on_panel

# Business-day bar calendar spanning the fixtures.
BARS = np.array([f"2024-01-{d:02d}" for d in (2, 3, 4, 5, 8, 9, 10, 11, 12)], dtype="datetime64[ns]")

_FRED_DGS10 = {"observations": [
    {"date": "2024-01-02", "value": "4.00"},
    {"date": "2024-01-03", "value": "4.05"},
    {"date": "2024-01-04", "value": "."},        # missing → dropped
    {"date": "2024-01-05", "value": "4.10"},
]}

_COT_ROWS = [
    {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "cftc_contract_market_code": "088691",
     "comm_positions_long_all": "100", "comm_positions_short_all": "40",
     "noncomm_positions_long_all": "0", "noncomm_positions_short_all": "0", "open_interest_all": "1"},
]


def _fred() -> FredConnector:
    return FredConnector(transport=lambda url: _FRED_DGS10,
                         series=(("DGS10", "10Y yield", "D"),), release_lag_days=1)


def _cot() -> CftcCotConnector:
    return CftcCotConnector(transport=lambda url: _COT_ROWS, markets=(("088691", "GOLD"),))


def _requests() -> list[SlotRequest]:
    return [
        SlotRequest(_fred(), SeriesRef("fred", "DGS10", "macro")),                 # terminal=fred:DGS10
        SlotRequest(_cot(), SeriesRef("cot", "088691:comm_net", "positioning"),
                    terminal="cot:gold_comm_net"),   # COT id is NOT DSL-legal → explicit alias
    ]


def test_bridge_builds_asof_joined_slots() -> None:
    slots = build_feature_slots(_requests(), BARS, start="2024-01-01", end="2024-01-31")
    assert set(slots) == {"fred:DGS10", "cot:gold_comm_net"}
    for arr in slots.values():
        assert arr.shape == (BARS.shape[0],)

    dgs = slots["fred:DGS10"]
    # DGS10 2024-01-02 releases 2024-01-03 → NaN on the 2nd, 4.00 from the 3rd onward.
    assert np.isnan(dgs[0])                                   # bar 2024-01-02
    assert dgs[1] == 4.00                                     # bar 2024-01-03 (release day)
    assert dgs[BARS.shape[0] - 1] == 4.10                    # last bar carries the latest reading

    cot = slots["cot:gold_comm_net"]
    # COT report 2024-01-02 releases +3d = 2024-01-05 → NaN before, 60 (=100-40) from the 5th.
    assert np.isnan(cot[0])
    assert cot[np.where(BARS == np.datetime64("2024-01-05", "ns"))[0][0]] == 60.0


def test_bridge_slot_is_dsl_addressable_on_a_panel() -> None:
    """The payoff: a slot built by the bridge resolves as a DSL terminal and broadcasts (T,)->(T,N)."""
    slots = build_feature_slots(_requests(), BARS, start="2024-01-01", end="2024-01-31")
    t, n = BARS.shape[0], 3
    rng = np.random.default_rng(0)
    px = np.exp(np.cumsum(0.01 * rng.standard_normal((t, n)), axis=0) + 3.0)
    panel = Panel(BARS, tuple(f"A{i}" for i in range(n)), px, px * 1.001, px * 0.999, px,
                  rng.uniform(1e6, 1e8, (t, n)), np.ones((t, n), bool), px * 1e6,
                  rng.integers(0, 3, size=n), {"survivorship_free": True}, feature_slots=slots)

    mat = eval_on_panel("fred:DGS10", panel)                 # resolves the P1a terminal registry
    assert mat.shape == (t, n)
    # a (T,) macro slot broadcasts to a matrix CONSTANT across the cross-section (why rank()==0 and
    # the overlay path is needed) — check on a bar where the slot is populated.
    row = mat[1]                                             # bar 2024-01-03
    assert np.allclose(row, row[0])
    # PIT: on 2024-01-03 the available reading is the Jan-02 datum (4.00, released that day); the
    # Jan-03 datum (4.05) is not public until Jan-04 (+1d lag).
    assert row[0] == 4.00


def test_bridge_rejects_non_dsl_terminal_and_duplicates() -> None:
    # The raw COT id 088691:comm_net (digit-leading, two colons) is not DSL-legal → must raise.
    assert not is_valid_terminal("cot:088691:comm_net")
    assert is_valid_terminal("fred:DGS10")
    bad = [SlotRequest(_cot(), SeriesRef("cot", "088691:comm_net", "positioning"))]
    with pytest.raises(ValueError, match="not a DSL-legal"):
        build_feature_slots(bad, BARS, start="2024-01-01", end="2024-01-31")

    dup = [
        SlotRequest(_fred(), SeriesRef("fred", "DGS10", "macro")),
        SlotRequest(_fred(), SeriesRef("fred", "DGS10", "macro")),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_feature_slots(dup, BARS, start="2024-01-01", end="2024-01-31")


def test_bridge_require_valid_raises_on_flagged_series() -> None:
    # A staleness-failing series: two points 4y apart → validate_series fails; require_valid raises.
    stale = {"observations": [{"date": "2018-01-02", "value": "1.0"},
                              {"date": "2024-01-02", "value": "2.0"}]}
    conn = FredConnector(transport=lambda url: stale, series=(("X", "x", "D"),))
    reqs = [SlotRequest(conn, SeriesRef("fred", "X", "macro"))]
    bars = np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="data-quality gate"):
        build_feature_slots(reqs, bars, start="2018-01-01", end="2024-12-31", require_valid=True)
