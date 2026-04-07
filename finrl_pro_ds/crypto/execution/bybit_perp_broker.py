"""Bybit Perpetual Futures Broker for Synapse Crypto 1H.

Handles order execution, position management, and fill monitoring
for USDT-margined perpetual futures on Bybit (paper via testnet, live via mainnet).

Uses CCXT unified interface with Bybit V5 API.

Usage:
    broker = BybitPerpBroker(testnet=True)
    await broker.connect()
    result = await broker.execute_rebalance(target_weights, current_positions)
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
    status: str            # "filled", "partial", "cancelled", "failed"
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


class BybitPerpBroker:
    """Bybit perpetual futures broker via CCXT.

    Supports both testnet (paper trading) and mainnet (live trading).
    All positions are USDT-margined, cross-margin, 1x leverage (no leverage).
    """

    QUOTE = "USDT"

    def __init__(
        self,
        testnet: bool = True,
        api_key: str | None = None,
        api_secret: str | None = None,
        order_type: str = "limit",
        limit_offset_pct: float = 0.0005,
        market_fallback_timeout: float = 60.0,
        min_trade_pct: float = 0.005,
        reconciliation_interval: float = 300.0,
    ):
        self.testnet = testnet
        # Use separate env vars for testnet vs mainnet to prevent accidental live trading
        if testnet:
            self.api_key = api_key or os.getenv("BYBIT_TESTNET_API_KEY", os.getenv("BYBIT_API_KEY", ""))
            self.api_secret = api_secret or os.getenv("BYBIT_TESTNET_API_SECRET", os.getenv("BYBIT_API_SECRET", ""))
        else:
            self.api_key = api_key or os.getenv("BYBIT_MAINNET_API_KEY", os.getenv("BYBIT_API_KEY", ""))
            self.api_secret = api_secret or os.getenv("BYBIT_MAINNET_API_SECRET", os.getenv("BYBIT_API_SECRET", ""))
        self.order_type = order_type
        self.limit_offset_pct = limit_offset_pct
        self.market_fallback_timeout = market_fallback_timeout
        self.min_trade_pct = min_trade_pct
        self.reconciliation_interval = reconciliation_interval
        self._exchange = None
        self._traded_assets: set[str] = set()

    async def connect(self) -> None:
        """Initialize CCXT exchange connection."""
        import ccxt.async_support as ccxt

        self._exchange = ccxt.bybit({
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
            logger.info("Bybit broker: TESTNET mode (paper trading)")
        else:
            logger.info("Bybit broker: MAINNET mode (live trading)")

        await self._exchange.load_markets()
        logger.info(f"Bybit broker connected: {len(self._exchange.markets)} markets loaded")

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    def _to_symbol(self, base: str) -> str:
        """Convert base asset to CCXT Bybit perpetual symbol."""
        return f"{base}/{self.QUOTE}:{self.QUOTE}"

    # -------------------------------------------------------------------
    # Portfolio operations
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
        assert self._exchange is not None, "Call connect() first"
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
        """Estimate fee for a trade."""
        rate = 0.0001 if is_maker else 0.00055  # Bybit futures maker/taker
        return notional * rate

    # -------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------
    async def get_positions(self, assets: list[str]) -> np.ndarray:
        """Fetch current positions from Bybit and convert to weights.

        Returns:
            Array of signed weights per asset.
        """
        assert self._exchange is not None

        positions = np.zeros(len(assets), dtype=np.float64)

        try:
            bybit_positions = await self._exchange.fetch_positions()

            # Get account balance for weight calculation
            balance = await self._exchange.fetch_balance()
            total_equity = float(balance.get("total", {}).get("USDT", 0))

            if total_equity < 1.0:
                return positions

            for pos in bybit_positions:
                symbol = pos.get("symbol", "")
                # Extract base asset from symbol
                base = symbol.split("/")[0] if "/" in symbol else ""

                if base in assets:
                    idx = assets.index(base)
                    notional = float(pos.get("notional") or 0)
                    side = pos.get("side", "")

                    weight = notional / total_equity
                    if side == "short":
                        weight = -weight

                    positions[idx] = weight

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")

        return positions

    async def get_account_info(self) -> dict:
        """Fetch account balance and margin info."""
        assert self._exchange is not None

        balance = await self._exchange.fetch_balance()

        return {
            "total_equity": float(balance.get("total", {}).get("USDT", 0)),
            "available_balance": float(balance.get("free", {}).get("USDT", 0)),
            "used_margin": float(balance.get("used", {}).get("USDT", 0)),
        }

    async def set_leverage(self, assets: list[str], leverage: int = 1) -> None:
        """Set leverage to 1x for all traded symbols."""
        assert self._exchange is not None

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
        assert self._exchange is not None

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
