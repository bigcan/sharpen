"""Unit tests for CTraderBroker — no connection required."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from finrl_pro_ds.cfd.execution.ctrader_broker import CTraderBroker, _VOLUME_SCALE
from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
    OrderResult,
    RebalanceResult,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def broker():
    """Create a CTraderBroker with test defaults (no connection)."""
    env = {
        "CTRADER_CLIENT_ID": "test_id",
        "CTRADER_CLIENT_SECRET": "test_secret",
        "CTRADER_ACCESS_TOKEN": "test_token",
        "CTRADER_ACCOUNT_ID": "12345",
    }
    with patch.dict(os.environ, env):
        b = CTraderBroker(
            testnet=True,
            symbol="XAUUSD",
            lot_size=100.0,
            min_lot=0.01,
            tick_size=0.01,
            leverage=30,
            taker_fee=0.00015,
        )
    return b


# ---------------------------------------------------------------
# Identity tests
# ---------------------------------------------------------------

def test_exchange_id(broker):
    assert broker.exchange_id == "ctrader"


def test_testnet_flag_true(broker):
    assert broker.testnet is True


def test_testnet_flag_false():
    env = {
        "CTRADER_CLIENT_ID": "test_id",
        "CTRADER_CLIENT_SECRET": "test_secret",
        "CTRADER_ACCESS_TOKEN": "test_token",
        "CTRADER_ACCOUNT_ID": "12345",
    }
    with patch.dict(os.environ, env):
        b = CTraderBroker(testnet=False)
    assert b.testnet is False


# ---------------------------------------------------------------
# Position <-> Lots conversion
# ---------------------------------------------------------------

def test_position_to_lots_full_long(broker):
    """$100K portfolio, Gold $3000/oz, 100oz/lot: pos=1.0 -> 0.33 lots."""
    lots = broker._position_to_lots(1.0, 100_000.0, 3000.0)
    # notional = 100000, lots = 100000 / (3000 * 100) = 0.333...
    assert abs(lots - 0.33) < 0.01


def test_position_to_lots_zero(broker):
    lots = broker._position_to_lots(0.0, 100_000.0, 3000.0)
    assert lots == 0.0


def test_position_to_lots_short(broker):
    lots = broker._position_to_lots(-0.5, 100_000.0, 3000.0)
    assert lots < 0
    assert abs(abs(lots) - 0.17) < 0.01


def test_position_to_lots_zero_price(broker):
    lots = broker._position_to_lots(1.0, 100_000.0, 0.0)
    assert lots == 0.0


def test_lots_to_position_roundtrip(broker):
    """Verify lots->position->lots roundtrip."""
    original_lots = 0.33
    fraction = broker._lots_to_position(original_lots, 100_000.0, 3000.0)
    assert 0 < fraction <= 1.0
    recovered_lots = broker._position_to_lots(fraction, 100_000.0, 3000.0)
    assert abs(recovered_lots - original_lots) < 0.02


def test_lots_to_position_clamps(broker):
    """Position fraction should be clamped to [-1, 1]."""
    # Huge position: 10 lots * 3000 * 100 = $3M on $100K → clamp to 1.0
    fraction = broker._lots_to_position(10.0, 100_000.0, 3000.0)
    assert fraction == 1.0

    # Negative huge position
    fraction = broker._lots_to_position(-10.0, 100_000.0, 3000.0)
    assert fraction == -1.0


# ---------------------------------------------------------------
# Volume encoding
# ---------------------------------------------------------------

def test_volume_encoding():
    """1.50 lots -> API volume 150."""
    assert CTraderBroker.lots_to_api_volume(1.50) == 150


def test_volume_encoding_min_lot():
    """0.01 lots -> API volume 1."""
    assert CTraderBroker.lots_to_api_volume(0.01) == 1


def test_volume_encoding_zero():
    assert CTraderBroker.lots_to_api_volume(0.0) == 0


def test_volume_decoding():
    """API volume 150 -> 1.50 lots."""
    assert CTraderBroker.api_volume_to_lots(150) == 1.50


def test_volume_roundtrip():
    for lots in [0.01, 0.05, 0.10, 0.33, 1.00, 5.50]:
        vol = CTraderBroker.lots_to_api_volume(lots)
        recovered = CTraderBroker.api_volume_to_lots(vol)
        assert abs(recovered - lots) < 0.005


# ---------------------------------------------------------------
# Funding rates (CFD = no funding)
# ---------------------------------------------------------------

def test_funding_rates_return_zero(broker):
    import asyncio
    result = asyncio.run(broker.get_funding_rates(["XAUUSD", "EURUSD"]))
    assert result == {"XAUUSD": 0.0, "EURUSD": 0.0}


# ---------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------

def test_env_var_missing_client_id():
    env = {
        "CTRADER_CLIENT_SECRET": "s",
        "CTRADER_ACCESS_TOKEN": "t",
        "CTRADER_ACCOUNT_ID": "1",
    }
    with patch.dict(os.environ, env, clear=True):
        b = CTraderBroker()
    # connect() should raise, not constructor
    assert b._client_id == ""


def test_connect_raises_without_credentials():
    import asyncio
    env = {"CTRADER_ACCOUNT_ID": "1"}
    with patch.dict(os.environ, env, clear=True):
        b = CTraderBroker()
    with pytest.raises(ValueError, match="Missing cTrader credentials"):
        asyncio.run(b.connect())


# ---------------------------------------------------------------
# OrderResult / RebalanceResult reuse
# ---------------------------------------------------------------

def test_order_result_creation():
    """Verify OrderResult from exchange_perp_broker works for cTrader."""
    order = OrderResult(
        asset="XAUUSD",
        symbol="XAUUSD",
        side="buy",
        order_type="market",
        quantity=0.10,
        price=3000.0,
        filled_quantity=0.10,
        avg_fill_price=3000.05,
        fee=4.50,
        status="filled",
        order_id="123456",
    )
    assert order.status == "filled"
    assert order.fee == 4.50


def test_rebalance_result_creation():
    result = RebalanceResult(
        orders=[],
        total_fees=0.0,
        n_executed=0,
        n_failed=0,
    )
    assert result.n_failed == 0
