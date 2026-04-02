"""DXtrade Broker Stub for Prop Firm Trading.

Implements the implicit broker interface used by ``LiveTradingEngine``.
Currently a stub — full DXtrade REST/WebSocket integration will be added
once API credentials and documentation are available from Velotrade.

The stub supports dry-run mode: all methods are implemented but order
execution raises ``NotImplementedError``.  This allows end-to-end pipeline
testing (config loading, component wiring, obs building, agent inference)
without actual DXtrade connectivity.

Usage:
    broker = DXtradePerpBroker(testnet=True)
    await broker.connect()  # Stub: no-op
    # broker.execute_position_change(...) -> NotImplementedError
"""

from __future__ import annotations

import logging
import os

import numpy as np

from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
    OrderResult,
    RebalanceResult,
)

logger = logging.getLogger(__name__)


class DXtradePerpBroker:
    """DXtrade perpetual futures broker for prop firm crypto trading.

    DXtrade API (when implemented):
    - REST: /api/v1/accounts, /api/v1/orders, /api/v1/positions
    - WebSocket: wss://dxtrade.{server}/ws for streaming
    - Auth: Bearer token from login endpoint

    Env vars:
        DXTRADE_SERVER      — DXtrade server hostname
        DXTRADE_USERNAME    — Account username
        DXTRADE_PASSWORD    — Account password
        DXTRADE_ACCOUNT_ID  — Trading account ID
    """

    # Required attributes for LiveTradingEngine
    exchange_id: str = "dxtrade"

    def __init__(
        self,
        testnet: bool = True,
        server: str | None = None,
        username: str | None = None,
        password: str | None = None,
        account_id: str | None = None,
        order_type: str = "market",
        leverage: int = 6,
        min_trade_pct: float = 0.005,
        taker_fee: float = 0.0005,
    ) -> None:
        self.testnet = testnet
        self.server = server or os.getenv("DXTRADE_SERVER", "")
        self._username = username or os.getenv("DXTRADE_USERNAME", "")
        self._password = password or os.getenv("DXTRADE_PASSWORD", "")
        self._account_id = account_id or os.getenv("DXTRADE_ACCOUNT_ID", "")
        self.order_type = order_type
        self.leverage = leverage
        self.min_trade_pct = min_trade_pct
        self.taker_fee = taker_fee

        # Session state (populated on connect)
        self._session_token: str | None = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate with DXtrade and initialize session.

        TODO: Implement REST login flow once API docs available.
        """
        if not self.server:
            logger.warning(
                "DXtrade: No server configured (DXTRADE_SERVER). "
                "Running in stub mode — order execution will raise NotImplementedError."
            )
            self._connected = False
            return

        # TODO: POST /api/v1/login with credentials → session token
        raise NotImplementedError(
            "DXtrade REST login not yet implemented. "
            "Set DXTRADE_SERVER, DXTRADE_USERNAME, DXTRADE_PASSWORD, DXTRADE_ACCOUNT_ID."
        )

    async def close(self) -> None:
        """Close DXtrade session."""
        self._session_token = None
        self._connected = False
        logger.info("DXtrade: Session closed")

    # ------------------------------------------------------------------
    # Order Execution
    # ------------------------------------------------------------------

    async def execute_position_change(
        self,
        asset: str,
        current_position: float,
        target_position: float,
        portfolio_value: float,
    ) -> OrderResult:
        """Execute a single-asset position change via DXtrade API.

        TODO: Implement once DXtrade API docs are available.
        """
        delta = abs(target_position - current_position)
        if delta < self.min_trade_pct:
            return OrderResult(
                asset=asset,
                symbol=f"{asset}/USDT",
                side="buy" if target_position > current_position else "sell",
                order_type="skipped",
                quantity=0.0,
                price=0.0,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                fee=0.0,
                status="skipped",
            )

        raise NotImplementedError(
            "DXtrade order execution not yet implemented. "
            "Use --dry-run mode for pipeline testing."
        )

    async def execute_rebalance(
        self,
        target_weights: np.ndarray,
        current_positions: np.ndarray,
        assets: list[str],
        portfolio_value: float,
    ) -> RebalanceResult:
        """Multi-asset portfolio rebalance via DXtrade API."""
        raise NotImplementedError("DXtrade rebalance not yet implemented.")

    # ------------------------------------------------------------------
    # Position & Account Queries
    # ------------------------------------------------------------------

    async def get_positions(self, assets: list[str]) -> np.ndarray:
        """Fetch positions from DXtrade, return signed weights."""
        if not self._connected:
            return np.zeros(len(assets), dtype=np.float64)
        raise NotImplementedError("DXtrade position query not yet implemented.")

    async def get_single_position(self, asset: str) -> float:
        """Single-asset position as signed weight [-1, 1]."""
        if not self._connected:
            return 0.0
        raise NotImplementedError("DXtrade single position query not yet implemented.")

    async def get_account_info(self) -> dict:
        """Return account equity and margin info."""
        if not self._connected:
            return {
                "total_equity": 0.0,
                "available_balance": 0.0,
                "used_margin": 0.0,
            }
        raise NotImplementedError("DXtrade account info not yet implemented.")

    async def set_leverage(
        self,
        assets: list[str],
        leverage: int | None = None,
    ) -> None:
        """Set leverage for symbols (default: self.leverage)."""
        lev = leverage or self.leverage
        logger.info(f"DXtrade: Leverage set to {lev}x (stub — no API call)")

    async def get_funding_rates(self, assets: list[str]) -> dict[str, float]:
        """DXtrade prop firm doesn't expose funding rates directly.

        Returns zeros — funding costs are embedded in the spread.
        """
        return {asset: 0.0 for asset in assets}

    async def emergency_flatten(self, assets: list[str]) -> RebalanceResult:
        """Close all positions via market orders."""
        if not self._connected:
            logger.warning("DXtrade: Cannot flatten — not connected")
            return RebalanceResult(n_failed=len(assets))
        raise NotImplementedError("DXtrade emergency flatten not yet implemented.")
