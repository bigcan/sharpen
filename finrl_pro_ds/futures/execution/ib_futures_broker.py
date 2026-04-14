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
        market_data_timeout: float = 10.0,
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
        self._roll_days_before_expiry = roll_days_before_expiry
        self._market_data_timeout = max(1.0, float(market_data_timeout))
        self._contract_manager: Optional[FuturesContractManager] = None

        # Required by LiveTradingEngine
        self.exchange_id = "ib"
        self.testnet = (port in (4002, 7497))

        # Internal position tracking (contracts, signed)
        self._position_contracts: int = 0
        self._portfolio_value: float = 0.0

        # Cached market price from updatePortfolio events.
        # ib_insync removes zero-position items from its portfolio cache,
        # so we must maintain our own price cache.
        self._cached_market_price: float = 0.0

    async def connect(self) -> None:
        """Connect to IB Gateway/TWS. No-op if already connected.

        FIX IB-01: Clean up old IB instance before creating a new one to
        prevent event handler leaks and thread accumulation on reconnect.
        This was the root cause of the 5d10h crash (Session 356).
        """
        if self._ib is not None and self._ib.isConnected():
            logger.info("IB already connected, skipping connect()")
        else:
            # FIX IB-01: Tear down stale IB instance before creating a new one
            if self._ib is not None:
                try:
                    self._ib.updatePortfolioEvent -= self._on_portfolio_update
                except Exception:
                    pass
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
                self._ib = None
                logger.info("Cleaned up stale IB instance before reconnect")

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
            roll_days_before_expiry=self._roll_days_before_expiry,
        )
        await self._contract_manager.resolve_front_month()

        # Cache market price from portfolio updates (ib_insync discards
        # zero-position items, so we listen and cache ourselves).
        self._ib.updatePortfolioEvent += self._on_portfolio_update

        logger.info(
            f"IBFuturesBroker ready: {self._contract_manager.contract.local_symbol} "
            f"({'PAPER' if self.testnet else 'LIVE'})"
        )

    async def close(self) -> None:
        """Disconnect from IB."""
        if self._ib:
            # FIX IB-07: Unsubscribe event handler to prevent leak
            try:
                self._ib.updatePortfolioEvent -= self._on_portfolio_update
            except Exception:
                pass
            if self._ib.isConnected():
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

        # Short-circuit no-op trades: when target and current position are both
        # zero, no trade is possible regardless of price. Skipping the mid-price
        # fetch keeps the engine alive through IB HMDS stalls (Error 162/322)
        # during bootstrap on a freshly-rolled contract, where _get_mid_price
        # has no portfolio-price fallback (updatePortfolio only fires on
        # non-zero positions).
        if target_position == 0.0 and self._position_contracts == 0:
            return OrderResult(
                asset=asset,
                symbol=self._contract_manager.contract.local_symbol,
                side="none",
                order_type="skipped",
                quantity=0,
                price=0.0,
                filled_quantity=0,
                avg_fill_price=0.0,
                fee=0.0,
                status="skipped",
            )

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
            # FIX IB-06: Use actual filled qty instead of assuming target was met.
            # In the limit-to-market fallback path, actual fill can differ from target.
            actual_filled = int(result.filled_quantity)
            if side == "BUY":
                self._position_contracts = current_contracts + actual_filled
            else:
                self._position_contracts = current_contracts - actual_filled

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

        # BUG-15: If _portfolio_value is uninitialized, fetch it now.
        # Without this, position always returns 0.0 on first call.
        if self._portfolio_value <= 0:
            await self.get_account_info()
        if self._portfolio_value <= 0:
            logger.warning(
                f"Portfolio value still 0 after account fetch — "
                f"returning raw contract count {n_contracts} as position"
            )
            return float(np.clip(n_contracts, -1.0, 1.0))

        # Zero position → fraction is 0 regardless of price. Skip the mid-price
        # fetch so HMDS stalls don't raise on every bar during bootstrap.
        if n_contracts == 0:
            return 0.0

        price = await self._get_mid_price()
        fraction = self._contracts_to_position(n_contracts, self._portfolio_value, price)
        return fraction

    async def get_account_info(self) -> dict:
        """Fetch IB account summary.

        Returns:
            dict with keys: total_equity, available_balance, used_margin
        """
        # BUG-08: accountSummary() returns a stale/empty cache.
        # accountValues() is fed by the automatic account subscription
        # that ib_insync maintains once connected.
        values = self._ib.accountValues()
        result = {
            "total_equity": 0.0,
            "available_balance": 0.0,
            "used_margin": 0.0,
        }

        for item in values:
            if item.currency != "USD":
                continue
            if item.tag == "NetLiquidation":
                result["total_equity"] = float(item.value)
            elif item.tag == "AvailableFunds":
                result["available_balance"] = float(item.value)
            elif item.tag == "InitMarginReq":
                result["used_margin"] = float(item.value)

        if result["total_equity"] == 0.0:
            logger.warning("get_account_info: NetLiquidation is 0 — IB subscription may not be ready")

        self._portfolio_value = result["total_equity"]
        return result

    async def emergency_flatten(self, assets: list[str]) -> RebalanceResult:
        """Close all positions via market orders."""
        # Refresh position from IB before flattening — internal tracking
        # may have drifted (e.g. after crash recovery).
        try:
            await self.get_single_position(assets[0] if assets else "")
        except Exception as e:
            logger.warning(f"Could not refresh position before flatten: {e}")

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

        Tries snapshot (2 attempts to ride through ushmds idle/wake cycles),
        then falls back to portfolio market price, then to reqHistoricalTicks.
        IB error 322 on snapshot + Error 162 "HMDS no data" during ushmds
        "inactive" phases are transient; a single retry avoids false
        emergency_flatten trips (Session 434).
        """
        contract = self._contract_manager.ib_contract
        n_attempts = 2
        per_attempt_timeout = self._market_data_timeout
        n_iters = int(per_attempt_timeout / 0.1)

        for attempt in range(1, n_attempts + 1):
            self._ib.reqMarketDataType(3)  # delayed as fallback
            ticker = self._ib.reqMktData(contract, genericTickList="", snapshot=True)

            for _ in range(n_iters):
                await asyncio.sleep(0.1)

                mid = ticker.midpoint()
                if mid is not None and not np.isnan(mid) and mid > 0:
                    self._ib.cancelMktData(contract)
                    self._ib.reqMarketDataType(1)
                    return mid

                last = ticker.last
                if last is not None and not np.isnan(last) and last > 0:
                    self._ib.cancelMktData(contract)
                    self._ib.reqMarketDataType(1)
                    return last

                close = ticker.close
                if close is not None and not np.isnan(close) and close > 0:
                    self._ib.cancelMktData(contract)
                    self._ib.reqMarketDataType(1)
                    return close

            self._ib.cancelMktData(contract)
            if attempt < n_attempts:
                logger.warning(
                    f"reqMktData snapshot empty (attempt {attempt}/{n_attempts}) "
                    f"— retrying after {per_attempt_timeout:.0f}s farm-wake wait"
                )
                await asyncio.sleep(2.0)

        self._ib.reqMarketDataType(1)

        # Fallback 1: portfolio market price (populated by updatePortfolio
        # events — requires an open position)
        portfolio_price = self._get_portfolio_market_price()
        if portfolio_price > 0:
            logger.warning(
                f"reqMktData snapshot failed — using portfolio price: "
                f"{portfolio_price:.2f}"
            )
            return portfolio_price

        # Fallback 2: historical last tick (bypasses snapshot bug; uses
        # ushmds which is the same farm that flickers but a different code
        # path that succeeds when snapshot returns Error 322)
        try:
            ticks = await asyncio.wait_for(
                asyncio.to_thread(
                    self._ib.reqHistoricalTicks,
                    contract,
                    "",
                    "",
                    1,
                    "TRADES",
                    useRth=False,
                    ignoreSize=True,
                ),
                timeout=5.0,
            )
            if ticks:
                price = float(ticks[-1].price)
                if price > 0:
                    logger.warning(
                        f"reqMktData snapshot failed — using historical tick: "
                        f"{price:.2f}"
                    )
                    return price
        except Exception as exc:
            logger.warning(f"reqHistoricalTicks fallback failed: {exc}")

        raise RuntimeError(
            f"Failed to get market price from IB within "
            f"{n_attempts * per_attempt_timeout:.0f}s (snapshot + hist-tick exhausted)"
        )

    def _on_portfolio_update(self, item) -> None:
        """Cache market price from updatePortfolio events.

        ib_insync removes zero-position items from its internal cache,
        but updatePortfolio events still carry a valid marketPrice.
        We cache it here so _get_mid_price() can use it as a fallback.
        """
        if self._contract_manager is None:
            return
        contract = self._contract_manager.ib_contract
        if (item.contract.conId == contract.conId
                or item.contract.localSymbol == contract.localSymbol):
            price = item.marketPrice
            if price is not None and not np.isnan(price) and price > 0:
                self._cached_market_price = float(price)

    def _get_portfolio_market_price(self) -> float:
        """Return cached market price from updatePortfolio events."""
        return self._cached_market_price

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
        # FIX IB-04: Track partial fills across limit-to-market fallback
        limit_filled = 0
        market_filled = 0
        limit_avg_price = 0.0
        market_avg_price = 0.0

        if not filled and order_type_str == "limit":
            # Fallback to market order for unfilled remainder
            logger.warning(f"Limit order not filled in {self._market_fallback_timeout}s, switching to market")
            # FIX IB-04: Wait for limit trade to reach terminal state after cancel
            # to get accurate partial fill count. The 1s sleep was insufficient.
            self._ib.cancelOrder(trade.order)
            for _ in range(20):  # Wait up to 2s for terminal state
                if trade.isDone():
                    break
                await asyncio.sleep(0.1)

            # Account for partial fills before cancellation
            limit_filled = int(trade.orderStatus.filled)
            limit_avg_price = trade.orderStatus.avgFillPrice if limit_filled > 0 else 0.0
            remaining = quantity - limit_filled
            market_filled = 0
            market_avg_price = 0.0
            if remaining > 0:
                market_order = MarketOrder(side, remaining)
                trade = self._ib.placeOrder(contract, market_order)
                filled = await self._wait_for_fill(trade, timeout=30.0)
                if filled:
                    market_filled = int(trade.orderStatus.filled)
                    market_avg_price = trade.orderStatus.avgFillPrice
            else:
                filled = True  # Limit order fully filled before cancel completed
            order_type_str = "market_fallback"

        if filled:
            # FIX IB-04: Combine fills from limit + market for accurate tracking
            if order_type_str == "market_fallback" and limit_filled > 0:
                filled_qty = limit_filled + market_filled
                total_notional = (limit_avg_price * limit_filled
                                  + market_avg_price * market_filled)
                avg_price = total_notional / filled_qty if filled_qty > 0 else 0.0
            else:
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

        When portfolio < contract notional (e.g. GC at $465K vs $262K portfolio),
        round() gives 0 for most fractions. We round up to 1 contract when the
        agent expresses meaningful conviction (fraction >= 0.1), since reaching
        here means the position change already passed the deadband check.
        IB margin requirements provide the real leverage safety net.
        """
        if abs(position_fraction) < 1e-6 or portfolio_value <= 0 or price <= 0:
            return 0

        multiplier = self._contract_manager.multiplier
        contract_notional = price * multiplier
        n_contracts = round(abs(position_fraction) * portfolio_value / contract_notional)

        # Floor: ensure at least 1 contract when agent has conviction
        if n_contracts == 0 and abs(position_fraction) >= 0.1:
            n_contracts = 1

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
