"""Tests for ExchangePerpBroker — unit tests (no exchange connection required).

Integration tests against testnet are marked with @pytest.mark.integration
and skipped by default (run with: pytest -m integration).
"""

import numpy as np
import pytest
import pytest_asyncio

from sharpen.crypto.execution.exchange_perp_broker import (
    _FEE_TABLE,
    ExchangePerpBroker,
    OrderResult,
    RebalanceResult,
)


class TestExchangePerpBrokerUnit:
    """Unit tests — no exchange connection."""

    def test_default_exchange_is_bybit(self):
        broker = ExchangePerpBroker()
        assert broker.exchange_id == "bybit"
        assert broker.testnet is True

    def test_binance_exchange(self):
        broker = ExchangePerpBroker(exchange="binance", testnet=True)
        assert broker.exchange_id == "binance"

    def test_symbol_construction(self):
        broker = ExchangePerpBroker(exchange="bybit")
        assert broker._to_symbol("BTC") == "BTC/USDT:USDT"
        assert broker._to_symbol("ETH") == "ETH/USDT:USDT"

    def test_fee_estimation_bybit(self):
        broker = ExchangePerpBroker(exchange="bybit")
        fee = broker._estimate_fee(10000.0, is_maker=False)
        assert abs(fee - 10000.0 * 0.00055) < 1e-8

        fee_maker = broker._estimate_fee(10000.0, is_maker=True)
        assert abs(fee_maker - 10000.0 * 0.0001) < 1e-8

    def test_fee_estimation_binance(self):
        broker = ExchangePerpBroker(exchange="binance")
        fee = broker._estimate_fee(10000.0, is_maker=False)
        assert abs(fee - 10000.0 * 0.0005) < 1e-8

    def test_fee_estimation_unknown_exchange_falls_back(self):
        broker = ExchangePerpBroker(exchange="unknown_exchange")
        # Should fall back to bybit fees
        fee = broker._estimate_fee(10000.0, is_maker=False)
        assert abs(fee - 10000.0 * 0.00055) < 1e-8

    def test_fee_table_has_required_exchanges(self):
        assert "bybit" in _FEE_TABLE
        assert "binance" in _FEE_TABLE

    def test_order_result_dataclass(self):
        result = OrderResult(
            asset="BTC", symbol="BTC/USDT:USDT", side="buy",
            order_type="market", quantity=0.001, price=50000.0,
            filled_quantity=0.001, avg_fill_price=50000.0,
            fee=0.0275, status="filled",
        )
        assert result.asset == "BTC"
        assert result.status == "filled"

    def test_rebalance_result_defaults(self):
        result = RebalanceResult()
        assert result.total_fees == 0.0
        assert result.n_executed == 0
        assert len(result.orders) == 0

    def test_connect_requires_valid_exchange(self):
        """Invalid exchange name should raise on connect()."""
        broker = ExchangePerpBroker(exchange="definitely_not_real")
        with pytest.raises(ValueError, match="not found in CCXT"):
            import asyncio
            asyncio.run(broker.connect())

    def test_testnet_mainnet_env_var_naming(self):
        """Verify env var naming convention is correct."""
        import os

        # Set test env vars
        os.environ["BYBIT_TESTNET_API_KEY"] = "test_key_123"
        os.environ["BYBIT_TESTNET_API_SECRET"] = "test_secret_456"

        try:
            broker = ExchangePerpBroker(exchange="bybit", testnet=True)
            assert broker.api_key == "test_key_123"
            assert broker.api_secret == "test_secret_456"
        finally:
            del os.environ["BYBIT_TESTNET_API_KEY"]
            del os.environ["BYBIT_TESTNET_API_SECRET"]


@pytest.mark.integration
class TestExchangePerpBrokerIntegration:
    """Integration tests against exchange testnet.

    Requires API keys in environment variables.
    Run with: pytest -m integration tests/crypto/test_exchange_broker.py
    """

    @pytest_asyncio.fixture
    async def bybit_broker(self):
        broker = ExchangePerpBroker(exchange="bybit", testnet=True)
        await broker.connect()
        yield broker
        await broker.close()

    @pytest.mark.asyncio
    async def test_bybit_connect_and_markets(self, bybit_broker):
        """Should connect to Bybit testnet and load markets."""
        assert bybit_broker._exchange is not None
        assert len(bybit_broker._exchange.markets) > 0

    @pytest.mark.asyncio
    async def test_bybit_get_account_info(self, bybit_broker):
        """Should fetch account info from testnet."""
        info = await bybit_broker.get_account_info()
        assert "total_equity" in info
        assert "available_balance" in info

    @pytest.mark.asyncio
    async def test_bybit_get_positions(self, bybit_broker):
        """Should fetch positions (even if empty)."""
        positions = await bybit_broker.get_positions(["BTC"])
        assert isinstance(positions, np.ndarray)
        assert len(positions) == 1
