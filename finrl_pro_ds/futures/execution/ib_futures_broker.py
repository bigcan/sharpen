"""Interactive Brokers futures broker for CME Gold (GC/MGC).

Implements the same duck-typed interface as ExchangePerpBroker:
    - connect(), close()
    - execute_position_change(asset, current_pos, target_pos, portfolio_value) -> OrderResult
    - get_single_position(asset) -> float
    - get_account_info() -> dict
    - emergency_flatten(assets) -> RebalanceResult
    - get_funding_rates(assets) -> dict

Attributes accessed by LiveTradingEngine:
    - exchange_id: str
    - testnet: bool
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
    OrderResult,
    RebalanceResult,
)
from finrl_pro_ds.futures.execution.contract_manager import FuturesContractManager

logger = logging.getLogger(__name__)


class IBFuturesBroker:
    """IB futures broker for Gold (GC/MGC) paper and live trading.

    Uses ib_insync for IB API connectivity. Accepts a pre-connected IB
    instance or connects on demand.
    """

    def __init__(
        self,
        ib=None,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        symbol: str = "MGC",
        order_type: str = "LMT",
        limit_offset_ticks: int = 2,
        market_fallback_timeout: float = 30.0,
        commission_per_side: float = 0.62,
        roll_days_before_expiry: int = 5,
    ):
        """
        Args:
            ib: Pre-connected ib_insync.IB instance (or None to connect on demand).
            host: IB Gateway host.
            port: IB Gateway port (4002=paper, 4001=live).
            client_id: IB client ID.
            symbol: "MGC" (micro, 10 oz) or "GC" (full, 100 oz).
            order_type: "LMT" (limit) or "MKT" (market).
            limit_offset_ticks: Cross N ticks into the book for limit orders.
            market_fallback_timeout: Seconds before limit falls back to market.
            commission_per_side: Commission per contract per side (MGC ~$0.62).
            roll_days_before_expiry: Days before expiry to trigger contract roll.
        """
        self._ib = ib
        self._host = host
        self._port = port
        self._client_id = client_id
        self._symbol = symbol
        self._order_type = order_type
        self._limit_offset_ticks = limit_offset_ticks
        self._market_fallback_timeout = market_fallback_timeout
        self._commission = commission_per_side
        self._contract_manager: Optional[FuturesContractManager] = None

        # Required by LiveTradingEngine
        self.exchange_id = "ib"
        self.testnet = (port in (4002, 7497))

        # Internal position tracking (contracts, signed)
        self._position_contracts: int = 0
        self._portfolio_value: float = 0.0

    async def connect(self) -> None:
        """Connect to IB Gateway/TWS. No-op if already connected."""
        if self._ib is not None and self._ib.isConnected():
            logger.info("IB already connected, skipping connect()")
        else:
            from ib_insync import IB
            self._ib = IB()
            await self._ib.connectAsync(
                self._host, self._port, clientId=self._client_id,
                timeout=20,
            )
            logger.info(f"Connected to IB at {self._host}:{self._port}")

        # Resolve front-month contract
        self._contract_manager = FuturesContractManager(
            ib=self._ib,
            symbol=self._symbol,
            roll_days_before_expiry=5,
        )
        await self._contract_manager.resolve_front_month()
        logger.info(
            f"IBFuturesBroker ready: {self._contract_manager.contract.local_symbol} "
            f"({'PAPER' if self.testnet else 'LIVE'})"
        )

    async def close(self) -> None:
        """Disconnect from IB."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB")

    async def execute_position_change(
        self,
        asset: str,
        current_position: float,
        target_position: float,
        portfolio_value: float,
    ) -> OrderResult:
        """Execute a position change on the front-month Gold contract.

        Converts position fraction [-1, 1] to integer contracts and
        places an order for the delta.

        Args:
            asset: Logical asset name (e.g., "GC").
            current_position: Current position fraction [-1, 1].
            target_position: Target position fraction [-1, 1].
            portfolio_value: Current portfolio value in USD.

        Returns:
            OrderResult with execution details.
        """
        self._portfolio_value = portfolio_value

        # Check for contract roll
        if self._contract_manager.check_roll_needed():
            logger.warning("Contract roll needed — flattening before roll")
            await self._flatten_for_roll()
            await self._contract_manager.roll_contract()

        # Convert fractions to contracts
        price = await self._get_mid_price()
        target_contracts = self._position_to_contracts(target_position, portfolio_value, price)
        current_contracts = self._position_contracts

        delta = target_contracts - current_contracts
        if delta == 0:
            return OrderResult(
                asset=asset,
                symbol=self._contract_manager.contract.local_symbol,
                side="none",
                order_type="skipped",
                quantity=0,
                price=price,
                filled_quantity=0,
                avg_fill_price=0.0,
                fee=0.0,
                status="skipped",
            )

        side = "BUY" if delta > 0 else "SELL"
        quantity = abs(delta)

        logger.info(
            f"Executing: {side} {quantity} {self._contract_manager.contract.local_symbol} "
            f"(pos {current_contracts} -> {target_contracts}, "
            f"frac {current_position:.3f} -> {target_position:.3f})"
        )

        result = await self._place_order(side, quantity, price)
        if result.status == "filled":
            self._position_contracts = target_contracts

        return result

    async def get_single_position(self, asset: str) -> float:
        """Get current position as a fraction of portfolio value.

        Queries IB for actual position and converts to [-1, 1] fraction.
        """
        positions = self._ib.positions()
        contract = self._contract_manager.ib_contract

        n_contracts = 0
        for pos in positions:
            if (pos.contract.conId == contract.conId
                    or pos.contract.localSymbol == contract.localSymbol):
                n_contracts = int(pos.position)
                break

        self._position_contracts = n_contracts

        if self._portfolio_value <= 0:
            return 0.0

        price = await self._get_mid_price()
        fraction = self._contracts_to_position(n_contracts, self._portfolio_value, price)
        return fraction

    async def get_account_info(self) -> dict:
        """Fetch IB account summary.

        Returns:
            dict with keys: total_equity, available_balance, used_margin
        """
        summary = self._ib.accountSummary()
        result = {
            "total_equity": 0.0,
            "available_balance": 0.0,
            "used_margin": 0.0,
        }

        for item in summary:
            if item.tag == "NetLiquidation" and item.currency == "USD":
                result["total_equity"] = float(item.value)
            elif item.tag == "AvailableFunds" and item.currency == "USD":
                result["available_balance"] = float(item.value)
            elif item.tag == "InitMarginReq" and item.currency == "USD":
                result["used_margin"] = float(item.value)

        self._portfolio_value = result["total_equity"]
        return result

    async def emergency_flatten(self, assets: list[str]) -> RebalanceResult:
        """Close all positions via market orders."""
        orders = []
        if self._position_contracts != 0:
            side = "SELL" if self._position_contracts > 0 else "BUY"
            quantity = abs(self._position_contracts)
            price = await self._get_mid_price()

            logger.warning(f"EMERGENCY FLATTEN: {side} {quantity} contracts")
            result = await self._place_order(side, quantity, price, force_market=True)
            orders.append(result)

            if result.status == "filled":
                self._position_contracts = 0

        return RebalanceResult(
            orders=orders,
            total_fees=sum(o.fee for o in orders),
            n_executed=sum(1 for o in orders if o.status == "filled"),
            n_failed=sum(1 for o in orders if o.status == "failed"),
        )

    async def get_funding_rates(self, assets: list[str]) -> dict[str, float]:
        """Futures don't have funding rates. Return 0.0 for compatibility."""
        return {a: 0.0 for a in assets}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_mid_price(self) -> float:
        """Get current mid price from IB market data.

        Tries live data first, then falls back to delayed data
        (IB error 354 means live data not subscribed).
        """
        contract = self._contract_manager.ib_contract

        # Request delayed data as fallback (market type 3 = delayed)
        self._ib.reqMarketDataType(3)
        ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=True)

        # Wait for snapshot with timeout
        for _ in range(50):
            await asyncio.sleep(0.1)

            # Try midpoint first
            mid = ticker.midpoint()
            if mid is not None and not np.isnan(mid) and mid > 0:
                self._ib.cancelMktData(contract)
                self._ib.reqMarketDataType(1)  # Reset to live
                return mid

            # Try last trade price
            last = ticker.last
            if last is not None and not np.isnan(last) and last > 0:
                self._ib.cancelMktData(contract)
                self._ib.reqMarketDataType(1)
                return last

            # Try close price (available when market is closed)
            close = ticker.close
            if close is not None and not np.isnan(close) and close > 0:
                self._ib.cancelMktData(contract)
                self._ib.reqMarketDataType(1)
                return close

        self._ib.cancelMktData(contract)
        self._ib.reqMarketDataType(1)
        raise RuntimeError("Failed to get market price from IB within 5s")

    async def _place_order(
        self,
        side: str,
        quantity: int,
        reference_price: float,
        force_market: bool = False,
    ) -> OrderResult:
        """Place an order via IB and wait for fill.

        Args:
            side: "BUY" or "SELL"
            quantity: Number of contracts (positive)
            reference_price: Current mid price for limit offset
            force_market: If True, use market order regardless of config
        """
        from ib_insync import LimitOrder, MarketOrder

        contract = self._contract_manager.ib_contract
        local_sym = self._contract_manager.contract.local_symbol
        tick = self._contract_manager.tick_size

        # Determine order type
        if force_market or self._order_type == "MKT":
            order = MarketOrder(side, quantity)
            order_type_str = "market"
        else:
            # Limit order crossing N ticks into the book
            if side == "BUY":
                limit_price = reference_price + self._limit_offset_ticks * tick
            else:
                limit_price = reference_price - self._limit_offset_ticks * tick
            # Round to tick size
            limit_price = round(limit_price / tick) * tick
            order = LimitOrder(side, quantity, limit_price)
            order_type_str = "limit"

        trade = self._ib.placeOrder(contract, order)
        logger.info(
            f"Order placed: {side} {quantity} {local_sym} "
            f"({order_type_str}, ref={reference_price:.2f})"
        )

        # Wait for fill with timeout
        filled = await self._wait_for_fill(trade, timeout=self._market_fallback_timeout)

        if not filled and order_type_str == "limit":
            # Fallback to market order
            logger.warning(f"Limit order not filled in {self._market_fallback_timeout}s, switching to market")
            self._ib.cancelOrder(order)
            await asyncio.sleep(1.0)

            market_order = MarketOrder(side, quantity)
            trade = self._ib.placeOrder(contract, market_order)
            filled = await self._wait_for_fill(trade, timeout=30.0)
            order_type_str = "market_fallback"

        if filled:
            avg_price = trade.orderStatus.avgFillPrice
            filled_qty = int(trade.orderStatus.filled)
            fee = self._commission * filled_qty
            status = "filled"
        else:
            avg_price = 0.0
            filled_qty = 0
            fee = 0.0
            status = "failed"
            logger.error(f"Order FAILED: {side} {quantity} {local_sym}")

        return OrderResult(
            asset=self._symbol,
            symbol=local_sym,
            side=side.lower(),
            order_type=order_type_str,
            quantity=float(quantity),
            price=reference_price,
            filled_quantity=float(filled_qty),
            avg_fill_price=avg_price,
            fee=fee,
            status=status,
            order_id=str(trade.order.orderId) if trade.order else "",
        )

    async def _wait_for_fill(self, trade, timeout: float) -> bool:
        """Wait for a trade to fill with timeout."""
        elapsed = 0.0
        interval = 0.2
        while elapsed < timeout:
            if trade.isDone():
                return trade.orderStatus.status == "Filled"
            await asyncio.sleep(interval)
            elapsed += interval
        return False

    async def _flatten_for_roll(self) -> None:
        """Flatten position before contract roll."""
        if self._position_contracts != 0:
            result = await self.emergency_flatten([self._symbol])
            logger.info(f"Flattened for roll: {result.n_executed} orders executed")

    def _position_to_contracts(
        self, position_fraction: float, portfolio_value: float, price: float,
    ) -> int:
        """Convert position fraction [-1, 1] to signed integer contracts.

        For MGC at $3000/oz, multiplier=10: contract_notional = $30,000.
        With $100K portfolio, position=1.0 => round(100000/30000) = 3 contracts.
        """
        if abs(position_fraction) < 1e-6 or portfolio_value <= 0 or price <= 0:
            return 0

        multiplier = self._contract_manager.multiplier
        contract_notional = price * multiplier
        n_contracts = round(abs(position_fraction) * portfolio_value / contract_notional)

        return int(np.sign(position_fraction)) * n_contracts

    def _contracts_to_position(
        self, n_contracts: int, portfolio_value: float, price: float,
    ) -> float:
        """Convert signed contract count to position fraction [-1, 1]."""
        if portfolio_value <= 0 or price <= 0:
            return 0.0
        multiplier = self._contract_manager.multiplier
        notional = n_contracts * price * multiplier
        return float(np.clip(notional / portfolio_value, -1.0, 1.0))
