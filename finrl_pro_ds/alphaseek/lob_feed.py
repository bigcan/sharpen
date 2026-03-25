"""BybitLOBFeed — async L2 orderbook WebSocket feed for AlphaSeek.

Connects to Bybit V5 WebSocket and maintains the latest L2 orderbook snapshot.
Outputs flat dicts matching the AlphaSeekFeatureEngine.process_snapshot() contract.

Usage:
    feed = BybitLOBFeed(symbol="BTC/USDT:USDT", depth=20, testnet=True)
    await feed.connect()
    snapshot = await feed.get_latest_snapshot()
    await feed.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Number of depth levels to extract for the feature engine
_FEATURE_DEPTH = 5


class BybitLOBFeed:
    """Async Bybit L2 orderbook feed via ccxt.pro WebSocket.

    Falls back to REST fetchOrderBook() if WebSocket is unavailable.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT:USDT",
        depth: int = 20,
        testnet: bool = True,
        api_key: str = "",
        api_secret: str = "",
        stale_threshold_s: float = 10.0,
    ):
        self.symbol = symbol
        self.depth = depth
        self.testnet = testnet
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        self.stale_threshold_s = stale_threshold_s

        self._exchange = None
        self._latest_ob: Optional[dict] = None
        self._last_update_ts: float = 0.0
        self._connected = False
        self._use_ws = True  # try WebSocket first

    async def connect(self) -> None:
        """Initialize exchange connection and start WebSocket subscription."""
        try:
            import ccxt.pro as ccxtpro

            self._exchange = ccxtpro.bybit(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "swap",
                        "sandboxMode": self.testnet,
                    },
                }
            )
            if self.testnet:
                self._exchange.set_sandbox_mode(True)

            self._use_ws = True
            self._connected = True
            logger.info(
                f"BybitLOBFeed connected (WS mode): {self.symbol}, "
                f"depth={self.depth}, testnet={self.testnet}"
            )
        except ImportError:
            logger.warning(
                "ccxt.pro not available, falling back to REST polling. "
                "Install with: pip install ccxt[pro]"
            )
            import ccxt.async_support as ccxt_async

            self._exchange = ccxt_async.bybit(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "swap",
                        "sandboxMode": self.testnet,
                    },
                }
            )
            if self.testnet:
                self._exchange.set_sandbox_mode(True)

            self._use_ws = False
            self._connected = True
            logger.info(
                f"BybitLOBFeed connected (REST fallback): {self.symbol}, "
                f"depth={self.depth}, testnet={self.testnet}"
            )

    async def _fetch_orderbook_ws(self) -> dict:
        """Fetch orderbook via WebSocket (blocks until next update)."""
        ob = await self._exchange.watch_order_book(self.symbol, limit=self.depth)
        return ob

    async def _fetch_orderbook_rest(self) -> dict:
        """Fetch orderbook via REST API."""
        ob = await self._exchange.fetch_order_book(self.symbol, limit=self.depth)
        return ob

    async def get_latest_snapshot(self) -> dict:
        """Fetch the latest orderbook and return as a flat snapshot dict.

        Returns dict matching FeatureEngine.process_snapshot() contract:
            best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
            spread, bid_prices_5, bid_qtys_5, ask_prices_5, ask_qtys_5
        """
        if not self._connected or self._exchange is None:
            raise RuntimeError("BybitLOBFeed not connected. Call connect() first.")

        try:
            if self._use_ws:
                ob = await self._fetch_orderbook_ws()
            else:
                ob = await self._fetch_orderbook_rest()
        except Exception as e:
            logger.error(f"Orderbook fetch failed: {e}")
            if self._use_ws:
                logger.warning("Falling back to REST for this tick")
                try:
                    ob = await self._fetch_orderbook_rest()
                except Exception as e2:
                    logger.error(f"REST fallback also failed: {e2}")
                    raise
            else:
                raise

        self._latest_ob = ob
        self._last_update_ts = time.monotonic()

        return self._ob_to_snapshot(ob)

    def _ob_to_snapshot(self, ob: dict) -> dict:
        """Convert ccxt orderbook to flat snapshot dict."""
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        best_bid_price = bids[0][0] if bids else 0.0
        best_bid_qty = bids[0][1] if bids else 0.0
        best_ask_price = asks[0][0] if asks else 0.0
        best_ask_qty = asks[0][1] if asks else 0.0

        spread = best_ask_price - best_bid_price

        # Extract top N levels for depth features
        bid_prices_5 = [b[0] for b in bids[:_FEATURE_DEPTH]]
        bid_qtys_5 = [b[1] for b in bids[:_FEATURE_DEPTH]]
        ask_prices_5 = [a[0] for a in asks[:_FEATURE_DEPTH]]
        ask_qtys_5 = [a[1] for a in asks[:_FEATURE_DEPTH]]

        return {
            "best_bid_price": best_bid_price,
            "best_bid_qty": best_bid_qty,
            "best_ask_price": best_ask_price,
            "best_ask_qty": best_ask_qty,
            "spread": spread,
            "bid_prices_5": bid_prices_5,
            "bid_qtys_5": bid_qtys_5,
            "ask_prices_5": ask_prices_5,
            "ask_qtys_5": ask_qtys_5,
            "timestamp_ms": ob.get("timestamp", int(time.time() * 1000)),
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self._exchange is not None

    @property
    def is_stale(self) -> bool:
        """True if latest data is older than stale_threshold_s."""
        if self._last_update_ts == 0.0:
            return True
        return (time.monotonic() - self._last_update_ts) > self.stale_threshold_s

    @property
    def last_update_age_s(self) -> float:
        """Seconds since last successful orderbook update."""
        if self._last_update_ts == 0.0:
            return float("inf")
        return time.monotonic() - self._last_update_ts

    async def close(self) -> None:
        """Close exchange connection."""
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange: {e}")
            self._exchange = None
        self._connected = False
        logger.info("BybitLOBFeed closed")
