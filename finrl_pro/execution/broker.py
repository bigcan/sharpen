"""Broker interface for live/paper trading execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BrokerClient(ABC):
    """Abstract base class for broker execution clients."""

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Return account details (cash, equity, status)."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Return current open positions."""
        pass

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" or "sell"
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit an order to the broker."""
        pass

    @abstractmethod
    def close_all_positions(self, cancel_orders: bool = True) -> List[Any]:
        """Liquidate all positions and optionally cancel open orders."""
        pass
