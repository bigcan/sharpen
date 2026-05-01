"""Step 0b — broker capture-callback hook unit tests.

Verifies that ExchangePerpBroker and BybitPerpBroker invoke a registered
capture callback after every fetch_positions / fetch_balance call, and that
callback errors are swallowed (must never disrupt trading).

This is the broker-side half of the E1 architect-pass Step 0b. The engine-side
wiring (writer + bar bounding) is covered by an end-to-end fixture replay
test once the live-replay capture has been collected post-un-halt.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from finrl_pro_ds.crypto.execution.bybit_perp_broker import BybitPerpBroker
from finrl_pro_ds.crypto.execution.exchange_perp_broker import ExchangePerpBroker


def _fake_position_record(notional: float = 5000.0, side: str = "long") -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "notional": notional,
        "side": side,
        "contracts": 0.1,
    }


def _fake_balance_record(equity: float = 10000.0) -> dict:
    return {
        "total": {"USDT": equity},
        "free": {"USDT": equity * 0.5},
        "used": {"USDT": equity * 0.5},
    }


def _make_epb_with_mock_exchange(positions, balance):
    broker = ExchangePerpBroker(exchange="bybit", testnet=True,
                                api_key="x", api_secret="y")
    broker._exchange = MagicMock()
    broker._exchange.fetch_positions = AsyncMock(return_value=positions)
    broker._exchange.fetch_balance = AsyncMock(return_value=balance)
    return broker


def _make_bybit_with_mock_exchange(positions, balance):
    broker = BybitPerpBroker(testnet=True, api_key="x", api_secret="y")
    broker._exchange = MagicMock()
    broker._exchange.fetch_positions = AsyncMock(return_value=positions)
    broker._exchange.fetch_balance = AsyncMock(return_value=balance)
    return broker


def test_epb_capture_callback_fires_on_get_positions():
    positions = [_fake_position_record()]
    balance = _fake_balance_record()
    broker = _make_epb_with_mock_exchange(positions, balance)

    captured = []
    broker.set_capture_callback(lambda name, raw: captured.append((name, raw)))

    result = asyncio.run(broker.get_positions(["BTC"]))
    assert isinstance(result, np.ndarray)

    events = [name for name, _ in captured]
    assert events == ["fetch_positions", "fetch_balance"]
    assert captured[0][1] == positions
    assert captured[1][1] == balance


def test_epb_capture_callback_fires_on_get_account_info():
    broker = _make_epb_with_mock_exchange([], _fake_balance_record(20000.0))

    captured = []
    broker.set_capture_callback(lambda name, raw: captured.append((name, raw)))

    info = asyncio.run(broker.get_account_info())
    assert info["total_equity"] == 20000.0

    assert len(captured) == 1
    assert captured[0][0] == "fetch_balance"


def test_epb_capture_callback_disabled_when_unset():
    broker = _make_epb_with_mock_exchange([_fake_position_record()],
                                          _fake_balance_record())
    # No callback registered → broker calls succeed without side-effects.
    asyncio.run(broker.get_positions(["BTC"]))
    asyncio.run(broker.get_account_info())
    # Absence of crash is the check.


def test_epb_capture_callback_errors_swallowed():
    """Callback errors must NEVER disrupt trading."""
    broker = _make_epb_with_mock_exchange([_fake_position_record()],
                                          _fake_balance_record())

    def bad_callback(name, raw):
        raise RuntimeError("simulated capture failure")

    broker.set_capture_callback(bad_callback)

    # Both calls must succeed despite callback raising.
    result = asyncio.run(broker.get_positions(["BTC"]))
    assert isinstance(result, np.ndarray)
    info = asyncio.run(broker.get_account_info())
    assert info["total_equity"] == 10000.0


def test_bybit_capture_callback_fires_on_get_positions():
    positions = [_fake_position_record()]
    balance = _fake_balance_record()
    broker = _make_bybit_with_mock_exchange(positions, balance)

    captured = []
    broker.set_capture_callback(lambda name, raw: captured.append((name, raw)))

    asyncio.run(broker.get_positions(["BTC"]))

    events = [name for name, _ in captured]
    assert events == ["fetch_positions", "fetch_balance"]


def test_bybit_capture_callback_errors_swallowed():
    broker = _make_bybit_with_mock_exchange([_fake_position_record()],
                                            _fake_balance_record())

    def bad(name, raw):
        raise ValueError("x")

    broker.set_capture_callback(bad)
    result = asyncio.run(broker.get_positions(["BTC"]))
    assert isinstance(result, np.ndarray)


def test_capture_callback_can_be_unregistered():
    broker = _make_epb_with_mock_exchange([_fake_position_record()],
                                          _fake_balance_record())

    captured = []
    broker.set_capture_callback(lambda name, raw: captured.append((name, raw)))
    asyncio.run(broker.get_account_info())
    assert len(captured) == 1

    broker.set_capture_callback(None)
    asyncio.run(broker.get_account_info())
    # Still 1 — second call did not invoke the (now-unset) callback.
    assert len(captured) == 1
