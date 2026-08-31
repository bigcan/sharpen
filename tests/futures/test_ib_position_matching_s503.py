"""Unit tests for IBFuturesBroker.get_single_position match logic (S503 fix).

Locks in the strict conId match introduced for the gmgp1-gold 2× position
drift, and the diagnostic behavior when ib_insync returns unexpected state.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sharpen.futures.execution.ib_futures_broker import IBFuturesBroker


def _make_broker(target_conid=712565978, target_localsym="MGCM6"):
    broker = IBFuturesBroker()
    broker._ib = MagicMock()
    cm = MagicMock()
    cm.ib_contract = SimpleNamespace(conId=target_conid, localSymbol=target_localsym)
    cm.multiplier = 10
    broker._contract_manager = cm
    broker._portfolio_value = 100_000.0
    broker._get_mid_price = AsyncMock(return_value=4683.0)
    return broker


def _pos(conid, sym, qty):
    return SimpleNamespace(
        contract=SimpleNamespace(conId=conid, localSymbol=sym),
        position=float(qty),
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_single_matching_position_returns_correct_fraction():
    broker = _make_broker()
    broker._ib.positions.return_value = [_pos(712565978, "MGCM6", 1)]

    fraction = _run(broker.get_single_position("MGC"))

    # 1 × 10 × 4683 / 100_000 = 0.4683
    assert fraction == pytest.approx(0.4683, abs=1e-3)
    assert broker._position_contracts == 1


def test_zero_position_short_circuits_to_zero():
    broker = _make_broker()
    broker._ib.positions.return_value = []

    fraction = _run(broker.get_single_position("MGC"))

    assert fraction == 0.0
    assert broker._position_contracts == 0


def test_stale_localsym_match_with_wrong_conid_is_ignored():
    """The previous OR-match would treat a stale Position with the same
    localSymbol but a different conId as ours. Strict-conId match drops it."""
    broker = _make_broker(target_conid=712565978)
    broker._ib.positions.return_value = [
        _pos(999999999, "MGCM6", 5),  # stale: same symbol, wrong conId
    ]

    fraction = _run(broker.get_single_position("MGC"))

    assert fraction == 0.0
    assert broker._position_contracts == 0


def test_falls_back_to_localsym_when_target_conid_missing():
    """Cold-start qualification race: contract.conId may be 0. Fall back to
    localSymbol so we don't blind-spot the position."""
    broker = _make_broker(target_conid=0, target_localsym="MGCM6")
    broker._ib.positions.return_value = [_pos(712565978, "MGCM6", 1)]

    fraction = _run(broker.get_single_position("MGC"))

    assert fraction == pytest.approx(0.4683, abs=1e-3)
    assert broker._position_contracts == 1


def test_unrelated_contract_in_list_is_filtered_out():
    broker = _make_broker(target_conid=712565978, target_localsym="MGCM6")
    broker._ib.positions.return_value = [
        _pos(111111111, "ESM6", 10),  # unrelated S&P contract
        _pos(712565978, "MGCM6", 1),  # ours
    ]

    fraction = _run(broker.get_single_position("MGC"))

    assert fraction == pytest.approx(0.4683, abs=1e-3)
    assert broker._position_contracts == 1


def test_duplicate_match_picks_largest_abs_qty(caplog):
    """If ib_insync ever returns two entries for the same conId (shouldn't,
    but the OR match could conflate stale entries), pick the largest |qty|
    and log loudly — engine reconcile catches genuine divergence."""
    broker = _make_broker(target_conid=712565978, target_localsym="MGCM6")
    broker._ib.positions.return_value = [
        _pos(712565978, "MGCM6", 2),
        _pos(712565978, "MGCM6", 1),
    ]

    import logging

    with caplog.at_level(logging.ERROR):
        fraction = _run(broker.get_single_position("MGC"))

    assert broker._position_contracts == 2
    assert any("Multiple matches" in r.message for r in caplog.records)
    # 2 contracts × multiplier 10 × $4683 / $100K = 0.9366
    assert fraction == pytest.approx(2 * 10 * 4683 / 100_000, abs=1e-3)
