"""Alpaca implementation of the BrokerClient interface."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

from finrl_pro.execution.broker import BrokerClient


class AlpacaBroker(BrokerClient):
    """Alpaca broker client for paper/live trading."""

    def __init__(self, api_key: str = "", api_secret: str = "", paper: bool = True) -> None:
        self._key = api_key or os.getenv("ALPACA_API_KEY_ID", "")
        self._secret = api_secret or os.getenv("ALPACA_API_SECRET_KEY", "")
        if not self._key or not self._secret:
            raise ValueError("Alpaca credentials not provided.")
        
        self._client = TradingClient(self._key, self._secret, paper=paper)

    def get_account(self) -> Dict[str, Any]:
        """Fetch account information."""
        acct = self._client.get_account()
        return {
            "id": acct.id,
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "currency": acct.currency,
            "status": acct.status,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open positions."""
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit an order."""
        # Map side string to Enum
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce.DAY  # Default
        
        if time_in_force.lower() == "gtc":
            tif_enum = TimeInForce.GTC
        
        # Construct Request
        if order_type.lower() == "market":
            req = MarketOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=side_enum,
                time_in_force=tif_enum
            )
        elif order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("Limit price required for limit orders.")
            req = LimitOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=limit_price
            )
        else:
            raise NotImplementedError(f"Order type {order_type} not supported.")

        # Submit
        order = self._client.submit_order(req)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else 0.0,
            "status": order.status,
            "type": order.type,
        }

    def close_all_positions(self, cancel_orders: bool = True) -> List[Any]:
        """Liquidate all positions."""
        return self._client.close_all_positions(cancel_orders=cancel_orders)
