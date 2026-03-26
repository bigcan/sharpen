"""Exchange-Agnostic Perpetual Futures Broker.

Handles order execution, position management, and fill monitoring
for USDT-margined perpetual futures on any CCXT-supported exchange.

Supports: Bybit, Binance, OKX (via CCXT unified API).
Paper trading via testnet, live trading via mainnet.

Usage:
    broker = ExchangePerpBroker(exchange="bybit", testnet=True)
    await broker.connect()
    result = await broker.execute_rebalance(target_weights, current_positions, assets, pv)
    # Or single-asset:
    order = await broker.execute_position_change("BTC", 0.0, 0.5, 10000.0)
    await broker.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of a single order execution."""
    asset: str
    symbol: str
    side: str              # "buy" or "sell"
    order_type: str        # "limit" or "market"
    quantity: float
    price: float
    filled_quantity: float
    avg_fill_price: float
    fee: float
    status: str            # "filled", "partial", "cancelled", "failed", "skipped"
    order_id: str = ""
    error: str = ""


@dataclass
class RebalanceResult:
    """Result of a full portfolio rebalance."""
    orders: list[OrderResult] = field(default_factory=list)
    total_fees: float = 0.0
    total_slippage_est: float = 0.0
    n_executed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    target_weights: Optional[np.ndarray] = None
    actual_weights: Optional[np.ndarray] = None


# Exchange-specific fallback fee schedules (taker/maker rates)
_FEE_TABLE: dict[str, dict[str, float]] = {
    "bybit":   {"maker": 0.0001,  "taker": 0.00055},
    "binance": {"maker": 0.0002,  "taker": 0.0005},
    "okx":     {"maker": 0.0002,  "taker": 0.0005},
}


class ExchangePerpBroker:
    """Exchange-agnostic perpetual futures broker via CCXT unified API.

    Supports both testnet (paper trading) and mainnet (live trading).
    All positions are USDT-margined, 1x leverage (no leverage) by default.

    Generalizes BybitPerpBroker to work with any CCXT-supported exchange.
    """

    QUOTE = "USDT"

    def __init__(
        self,
        exchange: str = "bybit",
        testnet: bool = True,
        api_key: str | None = None,
        api_secret: str | None = None,
        order_type: str = "limit",
        limit_offset_pct: float = 0.0005,
        market_fallback_timeout: float = 60.0,
        min_trade_pct: float = 0.005,
        reconciliation_interval: float = 300.0,
    ):
        self.exchange_id = exchange.lower()
        self.testnet = testnet

        # Resolve API credentials from env vars with exchange-specific naming
        # Convention: {EXCHANGE}_TESTNET_API_KEY / {EXCHANGE}_MAINNET_API_KEY
        # Fallback: {EXCHANGE}_API_KEY (generic)
        ex_upper = self.exchange_id.upper()
        if testnet:
            self.api_key = api_key or os.getenv(
                f"{ex_upper}_TESTNET_API_KEY",
                os.getenv(f"{ex_upper}_API_KEY", ""),
            )
            self.api_secret = api_secret or os.getenv(
                f"{ex_upper}_TESTNET_API_SECRET",
                os.getenv(f"{ex_upper}_API_SECRET", ""),
            )
        else:
            self.api_key = api_key or os.getenv(
                f"{ex_upper}_MAINNET_API_KEY",
                os.getenv(f"{ex_upper}_API_KEY", ""),
            )
            self.api_secret = api_secret or os.getenv(
                f"{ex_upper}_MAINNET_API_SECRET",
                os.getenv(f"{ex_upper}_API_SECRET", ""),
            )

        self.order_type = order_type
        self.limit_offset_pct = limit_offset_pct
        self.market_fallback_timeout = market_fallback_timeout
        self.min_trade_pct = min_trade_pct
        self.reconciliation_interval = reconciliation_interval
        self._exchange = None
        self._traded_assets: set[str] = set()

    def _ensure_connected(self) -> None:
        """Raise RuntimeError if exchange is not connected (replaces assert)."""
        if self._exchange is None:
            raise RuntimeError(
                "Exchange not connected. Call await broker.connect() first.",
            )

    async def connect(self) -> None:
        """Initialize CCXT exchange connection."""
        import ccxt.async_support as ccxt

        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{self.exchange_id}' not found in CCXT. "
                f"Available: bybit, binance, okx, ...",
            )

        # FIX AUD-M03: Fail fast on empty credentials
        if not self.api_key or not self.api_secret:
            raise ValueError(
                f"Missing API credentials for {self.exchange_id}. "
                f"Set {self.exchange_id.upper()}_TESTNET_API_KEY and "
                f"{self.exchange_id.upper()}_TESTNET_API_SECRET env vars.",
            )

        self._exchange = exchange_class({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",       # Perpetual futures
                "defaultSettle": "USDT",
            },
        })

        if self.testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info(
                f"{self.exchange_id.upper()} broker: TESTNET mode (paper trading)",
            )
        else:
            logger.info(
                f"{self.exchange_id.upper()} broker: MAINNET mode (LIVE trading)",
            )

        await self._exchange.load_markets()
        logger.info(
            f"{self.exchange_id.upper()} broker connected: "
            f"{len(self._exchange.markets)} markets loaded",
        )

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    def _to_symbol(self, base: str) -> str:
        """Convert base asset to CCXT perpetual symbol."""
        return f"{base}/{self.QUOTE}:{self.QUOTE}"

    # -------------------------------------------------------------------
    # Single-asset operations (GMGP1 / DeepScalper use case)
    # -------------------------------------------------------------------
    async def execute_position_change(
        self,
        asset: str,
        current_position: float,
        target_position: float,
        portfolio_value: float,
    ) -> OrderResult:
        """Execute a single-asset position change.

        Translates continuous position fraction into order execution.
        Handles both direction flips (long→short) and partial changes (0.5→0.8).

        Args:
            asset: Base asset symbol (e.g., "BTC").
            current_position: Current position fraction [-1, 1].
            target_position: Target position fraction [-1, 1].
            portfolio_value: Current portfolio value in USDT.

        Returns:
            OrderResult with execution details.
        """
        self._ensure_connected()

        delta = target_position - current_position

        # Skip dust trades
        if abs(delta) < self.min_trade_pct:
            return OrderResult(
                asset=asset,
                symbol=self._to_symbol(asset),
                side="buy" if delta > 0 else "sell",
                order_type="skipped",
                quantity=0,
                price=0,
                filled_quantity=0,
                avg_fill_price=0,
                fee=0,
                status="skipped",
                error=f"delta {delta:.4f} < min_trade_pct {self.min_trade_pct}",
            )

        # Ensure leverage is set
        await self.set_leverage([asset])

        symbol = self._to_symbol(asset)
        notional = abs(delta) * portfolio_value
        side = "buy" if delta > 0 else "sell"

        return await self._execute_order(
            asset=asset,
            symbol=symbol,
            side=side,
            notional=notional,
        )

    # -------------------------------------------------------------------
    # Portfolio operations (multi-asset rebalance)
    # -------------------------------------------------------------------
    async def execute_rebalance(
        self,
        target_weights: np.ndarray,
        current_positions: np.ndarray,
        assets: list[str],
        portfolio_value: float,
    ) -> RebalanceResult:
        """Execute a full portfolio rebalance.

        Calculates deltas, filters dust trades, and executes orders.

        Args:
            target_weights: Target signed weights per asset [-1, 1].
            current_positions: Current signed weights per asset.
            assets: List of base asset symbols.
            portfolio_value: Current portfolio value in USDT.

        Returns:
            RebalanceResult with execution details.
        """
        self._ensure_connected()
        assert len(target_weights) == len(assets) == len(current_positions)

        # Ensure leverage is set to 1x for all assets being traded
        self._traded_assets = set(assets)
        await self.set_leverage(assets)

        result = RebalanceResult(
            target_weights=target_weights.copy(),
        )

        deltas = target_weights - current_positions

        for i, asset in enumerate(assets):
            delta = deltas[i]

            # Skip dust trades
            if abs(delta) < self.min_trade_pct:
                result.n_skipped += 1
                continue

            symbol = self._to_symbol(asset)
            notional = abs(delta) * portfolio_value
            side = "buy" if delta > 0 else "sell"

            try:
                order_result = await self._execute_order(
                    asset=asset,
                    symbol=symbol,
                    side=side,
                    notional=notional,
                )
                result.orders.append(order_result)
                result.total_fees += order_result.fee

                if order_result.status in ("filled", "partial"):
                    result.n_executed += 1
                else:
                    result.n_failed += 1

            except Exception as e:
                logger.error(f"Order failed for {asset}: {e}")
                result.orders.append(OrderResult(
                    asset=asset, symbol=symbol, side=side,
                    order_type="failed", quantity=0, price=0,
                    filled_quantity=0, avg_fill_price=0, fee=0,
                    status="failed", error=str(e),
                ))
                result.n_failed += 1

        # Reconcile positions after rebalance
        actual_positions = await self.get_positions(assets)
        result.actual_weights = actual_positions

        logger.info(
            f"Rebalance complete: {result.n_executed} executed, "
            f"{result.n_skipped} skipped, {result.n_failed} failed, "
            f"fees={result.total_fees:.4f} USDT",
        )

        return result

    # -------------------------------------------------------------------
    # Order execution
    # -------------------------------------------------------------------
    async def _execute_order(
        self,
        asset: str,
        symbol: str,
        side: str,
        notional: float,
    ) -> OrderResult:
        """Execute a single order (limit with market fallback).

        Args:
            asset: Base asset symbol.
            symbol: CCXT symbol.
            side: "buy" or "sell".
            notional: Trade value in USDT.

        Returns:
            OrderResult with execution details.
        """
        # Get current price
        ticker = await self._exchange.fetch_ticker(symbol)
        mid_price = ticker.get("last", 0)
        if ticker.get("bid") and ticker.get("ask"):
            mid_price = (ticker["bid"] + ticker["ask"]) / 2
        if not mid_price or mid_price < 1e-10:
            raise ValueError(f"Invalid mid_price for {symbol}: {mid_price}")

        # Calculate quantity using CCXT precision helpers (handles step-size vs decimals)
        quantity = notional / mid_price
        quantity = float(self._exchange.amount_to_precision(symbol, quantity))

        # Enforce minimum quantity — skip if below exchange minimum
        market_info = self._exchange.market(symbol)
        min_qty = market_info.get("limits", {}).get("amount", {}).get("min", 0.001)
        if quantity < min_qty:
            logger.info(f"Order quantity {quantity} below minimum {min_qty} for {symbol}, skipping")
            return OrderResult(
                asset=asset, symbol=symbol, side=side,
                order_type="skipped", quantity=quantity, price=mid_price,
                filled_quantity=0, avg_fill_price=0, fee=0,
                status="skipped", error=f"quantity {quantity} < min {min_qty}",
            )

        if self.order_type == "limit":
            # Place limit order crossing spread for immediate fill with price protection
            if side == "buy":
                price = mid_price * (1 + self.limit_offset_pct)
            else:
                price = mid_price * (1 - self.limit_offset_pct)

            price = float(self._exchange.price_to_precision(symbol, price))

            order = await self._exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=quantity,
                price=price,
                params={"timeInForce": "GTC"},
            )

            # Wait for fill or timeout → fallback to market
            filled_order = await self._wait_for_fill(
                order["id"], symbol, timeout=self.market_fallback_timeout,
            )

            if filled_order and filled_order["status"] == "closed":
                # Use actual fee from exchange response when available
                actual_fee = filled_order.get("fee") or {}
                fee_cost = float(actual_fee.get("cost", 0)) if isinstance(actual_fee, dict) else 0
                if fee_cost <= 0:
                    fee_cost = self._estimate_fee(notional, is_maker=False)

                return OrderResult(
                    asset=asset, symbol=symbol, side=side,
                    order_type="limit", quantity=quantity, price=price,
                    filled_quantity=filled_order.get("filled", quantity),
                    avg_fill_price=filled_order.get("average", price),
                    fee=fee_cost,
                    status="filled", order_id=order["id"],
                )

            # Cancel unfilled limit and fall through to market
            already_filled = 0.0
            try:
                await self._exchange.cancel_order(order["id"], symbol)
            except Exception:
                pass
            # Fetch actual fill status after cancel to avoid double-execution
            try:
                final_order = await self._exchange.fetch_order(order["id"], symbol)
                already_filled = float(final_order.get("filled", 0) if final_order else 0)
            except Exception:
                logger.warning(f"Could not fetch order status for {order['id']}, assuming unfilled")

            # Subtract partial fill from remaining quantity to avoid double-sizing
            remaining_qty = max(quantity - already_filled, 0.0)
            if remaining_qty < min_qty:
                # Partial fill covered the order, return partial result
                return OrderResult(
                    asset=asset, symbol=symbol, side=side,
                    order_type="limit", quantity=quantity, price=price,
                    filled_quantity=already_filled,
                    avg_fill_price=price,
                    fee=self._estimate_fee(already_filled * mid_price, is_maker=False),
                    status="partial" if already_filled > 0 else "cancelled",
                    order_id=order["id"],
                )
        else:
            remaining_qty = quantity

        # Market order (default or fallback)
        order = await self._exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=remaining_qty,
        )

        # H1: Use remaining_qty as fallback, not full quantity
        actual_fee = order.get("fee") or {}
        fee_cost = float(actual_fee.get("cost", 0)) if isinstance(actual_fee, dict) else 0
        if fee_cost <= 0:
            fee_cost = self._estimate_fee(remaining_qty * mid_price, is_maker=False)

        return OrderResult(
            asset=asset, symbol=symbol, side=side,
            order_type="market", quantity=remaining_qty, price=mid_price,
            filled_quantity=order.get("filled", remaining_qty),
            avg_fill_price=order.get("average", mid_price),
            fee=fee_cost,
            status="filled" if order.get("status") == "closed" else "partial",
            order_id=order.get("id", ""),
        )

    async def _wait_for_fill(
        self, order_id: str, symbol: str, timeout: float,
    ) -> dict | None:
        """Poll order status until filled or timeout."""
        elapsed = 0.0
        poll_interval = 2.0

        while elapsed < timeout:
            try:
                order = await self._exchange.fetch_order(order_id, symbol)
                if order["status"] in ("closed", "canceled", "cancelled", "expired"):
                    return order
            except Exception as e:
                logger.debug(f"Order poll error: {e}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return None  # Timeout

    def _estimate_fee(self, notional: float, is_maker: bool) -> float:
        """Estimate fee for a trade using exchange-specific fee table."""
        fee_entry = _FEE_TABLE.get(self.exchange_id, _FEE_TABLE["bybit"])
        rate = fee_entry["maker"] if is_maker else fee_entry["taker"]
        return notional * rate

    # -------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------
    async def get_positions(self, assets: list[str]) -> np.ndarray:
        """Fetch current positions and convert to signed weights.

        Returns:
            Array of signed weights per asset.
        """
        self._ensure_connected()

        positions = np.zeros(len(assets), dtype=np.float64)

        try:
            exchange_positions = await self._exchange.fetch_positions()

            # Get account balance for weight calculation
            balance = await self._exchange.fetch_balance()
            total_equity = float(balance.get("total", {}).get("USDT", 0))

            if total_equity < 1.0:
                return positions

            for pos in exchange_positions:
                symbol = pos.get("symbol", "")
                # Extract base asset from symbol
                base = symbol.split("/")[0] if "/" in symbol else ""

                if base in assets:
                    idx = assets.index(base)
                    notional = float(pos.get("notional", 0))
                    side = pos.get("side", "")

                    weight = notional / total_equity
                    if side == "short":
                        weight = -weight

                    positions[idx] = weight

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")

        return positions

    async def get_single_position(self, asset: str) -> float:
        """Fetch current position for a single asset as signed weight.

        Returns:
            Signed weight in [-1, 1]. 0.0 if no position or error.
        """
        positions = await self.get_positions([asset])
        return float(positions[0])

    async def get_account_info(self) -> dict:
        """Fetch account balance and margin info."""
        self._ensure_connected()

        balance = await self._exchange.fetch_balance()

        return {
            "total_equity": float(balance.get("total", {}).get("USDT", 0)),
            "available_balance": float(balance.get("free", {}).get("USDT", 0)),
            "used_margin": float(balance.get("used", {}).get("USDT", 0)),
        }

    async def set_leverage(self, assets: list[str], leverage: int = 1) -> None:
        """Set leverage for all traded symbols."""
        self._ensure_connected()

        for asset in assets:
            symbol = self._to_symbol(asset)
            try:
                await self._exchange.set_leverage(leverage, symbol)
                logger.debug(f"Leverage set to {leverage}x for {symbol}")
            except Exception as e:
                msg = str(e).lower()
                if "already" in msg or "not modified" in msg or "not changed" in msg:
                    logger.debug(f"Leverage already set for {symbol}: {e}")
                else:
                    logger.warning(f"Failed to set leverage for {symbol}: {e}")

    async def get_funding_rates(self, assets: list[str]) -> dict[str, float]:
        """Fetch current funding rates for all assets."""
        self._ensure_connected()

        rates = {}
        for asset in assets:
            symbol = self._to_symbol(asset)
            try:
                info = await self._exchange.fetch_funding_rate(symbol)
                rates[asset] = float(info.get("fundingRate", 0.0))
            except Exception as e:
                logger.debug(f"Funding rate fetch for {asset}: {e}")
                rates[asset] = 0.0

        return rates

    async def emergency_flatten(self, assets: list[str]) -> RebalanceResult:
        """Emergency: close all positions via market orders.

        Used on unrecoverable errors or kill switch activation.
        """
        self._ensure_connected()

        logger.warning(f"EMERGENCY FLATTEN: Closing all positions for {assets}")

        current_positions = await self.get_positions(assets)
        target_weights = np.zeros(len(assets), dtype=np.float64)

        account = await self.get_account_info()
        portfolio_value = account["total_equity"]

        # FIX AUD-H03: Use saved order_type and restore after, with retry logic.
        # This avoids the race condition of mutating self.order_type during
        # concurrent async operations.
        saved_order_type = self.order_type
        self.order_type = "market"
        last_result = None
        try:
            for attempt in range(3):
                last_result = await self.execute_rebalance(
                    target_weights=target_weights,
                    current_positions=current_positions,
                    assets=assets,
                    portfolio_value=portfolio_value,
                )
                if last_result.n_failed == 0:
                    break
                logger.warning(
                    f"Emergency flatten attempt {attempt+1}/3: "
                    f"{last_result.n_failed} failed, retrying...",
                )
                await asyncio.sleep(2.0)
                # Re-fetch for retry
                current_positions = await self.get_positions(assets)
        finally:
            self.order_type = saved_order_type

        logger.warning(
            f"EMERGENCY FLATTEN complete: {last_result.n_executed} closed, "
            f"{last_result.n_failed} failed",
        )
        return last_result
