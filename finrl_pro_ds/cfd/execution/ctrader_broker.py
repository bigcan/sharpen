"""cTrader Open API broker for XAUUSD CFD paper/live trading.

Implements the same duck-typed interface as IBFuturesBroker and
ExchangePerpBroker:
    - connect(), close()
    - execute_position_change(asset, current_pos, target_pos, portfolio_value) -> OrderResult
    - get_single_position(asset) -> float
    - get_account_info() -> dict
    - emergency_flatten(assets) -> RebalanceResult
    - get_funding_rates(assets) -> dict

Attributes accessed by LiveTradingEngine:
    - exchange_id: str
    - testnet: bool

Uses ctrader-open-api Python SDK (Protobuf over TCP, Twisted reactor).
Requires env vars: CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET,
                   CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from typing import Optional

import numpy as np

from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
    OrderResult,
    RebalanceResult,
)

logger = logging.getLogger(__name__)


def _with_reconnect(method):
    """Decorator: check connection before API call, reconnect on failure.

    Wraps async methods that use ``_send_request`` so that a dropped TCP
    connection triggers an automatic reconnect-and-retry cycle instead of
    propagating the error immediately.
    """

    @functools.wraps(method)
    async def wrapper(self: "CTraderBroker", *args, **kwargs):
        # Fast-path: connection looks healthy
        if self._connected and self._client is not None:
            try:
                return await method(self, *args, **kwargs)
            except (RuntimeError, asyncio.TimeoutError, OSError) as exc:
                # First failure — fall through to reconnect path
                logger.warning(
                    "API call %s failed (%s), attempting reconnect",
                    method.__name__,
                    exc,
                )

        # Connection is down — try to restore it
        reconnected = await self._reconnect()
        if not reconnected:
            raise RuntimeError(
                f"cTrader reconnect failed — cannot execute {method.__name__}"
            )
        return await method(self, *args, **kwargs)

    return wrapper

# Volume encoding: cTrader API represents volume in hundredths of a lot.
# 1.00 lot = volume 100, 0.01 lot = volume 1.
_VOLUME_SCALE = 100


class CTraderBroker:
    """cTrader CFD broker for XAUUSD paper and live trading.

    Uses ctrader-open-api for connectivity via Protobuf over TCP.
    Supports IC Markets demo (paper) and FTMO live (challenge) accounts.
    """

    def __init__(
        self,
        testnet: bool = True,
        symbol: str = "XAUUSD",
        lot_size: float = 100.0,
        min_lot: float = 0.01,
        tick_size: float = 0.01,
        leverage: int = 30,
        order_type: str = "MKT",
        taker_fee: float = 0.00015,
        market_fallback_timeout: float = 10.0,
        max_reconnect_attempts: int = 3,
    ):
        """
        Args:
            testnet: True for IC Markets demo, False for FTMO live.
            symbol: CFD symbol name (e.g., "XAUUSD").
            lot_size: Units per standard lot (100 oz for Gold).
            min_lot: Minimum lot increment (0.01 = 1 oz).
            tick_size: Minimum price increment ($0.01 for CFD Gold).
            leverage: Account leverage (FTMO Gold = 1:30).
            order_type: Order type ("MKT" for market).
            taker_fee: Effective spread as fraction (1.5bps = 0.00015).
            market_fallback_timeout: Seconds to wait for order fill.
            max_reconnect_attempts: Max reconnect attempts with exponential
                backoff (5s, 10s, 20s, ...) before giving up.
        """
        self._testnet = testnet
        self._symbol_name = symbol
        self._lot_size = lot_size
        self._min_lot = min_lot
        self._tick_size = tick_size
        self._leverage = leverage
        self._order_type = order_type
        self._taker_fee = taker_fee
        self._market_fallback_timeout = market_fallback_timeout
        self._max_reconnect_attempts = max_reconnect_attempts

        # Required by LiveTradingEngine
        self.exchange_id = "ctrader"
        self.testnet = testnet

        # Credentials from env vars
        self._client_id = os.environ.get("CTRADER_CLIENT_ID", "")
        self._client_secret = os.environ.get("CTRADER_CLIENT_SECRET", "")
        self._access_token = os.environ.get("CTRADER_ACCESS_TOKEN", "")
        self._account_id = int(os.environ.get("CTRADER_ACCOUNT_ID", "0"))
        self._refresh_token = os.environ.get("CTRADER_REFRESH_TOKEN", "")

        # Connection state
        self._client = None
        self._connected = False
        self._symbol_id: Optional[int] = None
        self._symbol_digits: int = 2
        self._money_digits: int = 2
        self._position_lots: float = 0.0
        self._portfolio_value: float = 0.0
        self._bid: float = 0.0
        self._ask: float = 0.0
        self._mid_price: float = 0.0
        self._token_refresh_task: Optional[asyncio.Task] = None
        self._response_events: dict[int, asyncio.Event] = {}
        self._response_data: dict[int, object] = {}
        self._msg_id_counter = 0

        # Reconnect state
        self._reconnect_lock = asyncio.Lock()
        self._consecutive_reconnects = 0

    async def connect(self) -> None:
        """Connect to cTrader Open API and authenticate."""
        if self._connected:
            return
        if not self._client_id or not self._client_secret:
            raise ValueError(
                "Missing cTrader credentials. Set CTRADER_CLIENT_ID and "
                "CTRADER_CLIENT_SECRET environment variables."
            )
        if not self._access_token:
            raise ValueError(
                "Missing CTRADER_ACCESS_TOKEN. Obtain via OAuth flow at "
                "https://openapi.ctrader.com/apps/auth"
            )
        if not self._account_id:
            raise ValueError(
                "Missing CTRADER_ACCOUNT_ID. Set the numeric trading account ID."
            )

        from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAAccountAuthReq,
            ProtoOAApplicationAuthReq,
            ProtoOASubscribeSpotsReq,
            ProtoOASymbolByIdReq,
            ProtoOASymbolsListReq,
            ProtoOATraderReq,
        )

        host = EndPoints.PROTOBUF_DEMO_HOST if self._testnet else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT

        logger.info(
            f"Connecting to cTrader {'DEMO' if self._testnet else 'LIVE'} "
            f"at {host}:{port}"
        )

        # Create client and set callbacks
        self._client = Client(host, port, TcpProtocol)
        self._client.setMessageReceivedCallback(self._on_message)
        self._client.setDisconnectedCallback(self._on_disconnected)

        # Start Twisted service — must run on the reactor thread since
        # startService() calls reactor.connectTCP() internally.
        from twisted.internet import reactor

        reactor.callFromThread(self._client.startService)

        # Wait for TCP connection
        connected_deferred = self._client.whenConnected(failAfterFailures=3)
        await self._deferred_to_future(connected_deferred, timeout=15.0)
        self._connected = True

        # Step 1: Application auth
        app_auth = ProtoOAApplicationAuthReq()
        app_auth.clientId = self._client_id
        app_auth.clientSecret = self._client_secret
        app_res = await self._send_request(app_auth, timeout=10.0)
        self._check_error(Protobuf.extract(app_res), "Application auth")
        logger.info("Application authenticated")

        # Step 2: Account auth
        acct_auth = ProtoOAAccountAuthReq()
        acct_auth.ctidTraderAccountId = self._account_id
        acct_auth.accessToken = self._access_token
        acct_res = await self._send_request(acct_auth, timeout=10.0)
        self._check_error(Protobuf.extract(acct_res), "Account auth")
        logger.info(f"Account {self._account_id} authenticated")

        # Step 3: Resolve symbol
        sym_list_req = ProtoOASymbolsListReq()
        sym_list_req.ctidTraderAccountId = self._account_id
        sym_list_res = await self._send_request(sym_list_req, timeout=15.0)

        sym_res_payload = Protobuf.extract(sym_list_res)
        self._check_error(sym_res_payload, "Symbol list")
        for sym in sym_res_payload.symbol:
            if sym.symbolName.upper() == self._symbol_name.upper():
                self._symbol_id = sym.symbolId
                break

        if self._symbol_id is None:
            available = [s.symbolName for s in sym_res_payload.symbol[:20]]
            raise ValueError(
                f"Symbol '{self._symbol_name}' not found. "
                f"Available (first 20): {available}"
            )

        # Step 4: Get symbol details (digits, lot size)
        sym_detail_req = ProtoOASymbolByIdReq()
        sym_detail_req.ctidTraderAccountId = self._account_id
        sym_detail_req.symbolId.append(self._symbol_id)
        sym_detail_res = await self._send_request(sym_detail_req, timeout=10.0)

        sym_detail_payload = Protobuf.extract(sym_detail_res)
        if sym_detail_payload.symbol:
            sym_info = sym_detail_payload.symbol[0]
            self._symbol_digits = sym_info.digits
            if sym_info.lotSize:
                self._lot_size = sym_info.lotSize / _VOLUME_SCALE
            if sym_info.minVolume:
                self._min_lot = sym_info.minVolume / _VOLUME_SCALE
            logger.info(
                f"Symbol {self._symbol_name}: id={self._symbol_id}, "
                f"digits={self._symbol_digits}, lot_size={self._lot_size}, "
                f"min_lot={self._min_lot}"
            )

        # Step 5: Get trader info (balance, moneyDigits)
        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = self._account_id
        trader_res = await self._send_request(trader_req, timeout=10.0)

        trader_payload = Protobuf.extract(trader_res)
        trader = trader_payload.trader
        self._money_digits = trader.moneyDigits
        balance = trader.balance / (10 ** self._money_digits)
        self._portfolio_value = balance
        logger.info(
            f"Trader: balance={balance:.2f}, moneyDigits={self._money_digits}, "
            f"leverage={trader.leverageInCents / 100}"
        )

        # Step 6: Subscribe to live spot prices
        spot_req = ProtoOASubscribeSpotsReq()
        spot_req.ctidTraderAccountId = self._account_id
        spot_req.symbolId.append(self._symbol_id)
        await self._send_request(spot_req, timeout=10.0)
        logger.info(f"Subscribed to {self._symbol_name} spot prices")

        # Step 7: Start token refresh background task
        if self._refresh_token:
            self._token_refresh_task = asyncio.create_task(
                self._token_refresh_loop()
            )

        mode = "DEMO (IC Markets)" if self._testnet else "LIVE (FTMO)"
        logger.info(f"CTraderBroker ready: {self._symbol_name} CFD ({mode})")

    async def close(self) -> None:
        """Disconnect from cTrader."""
        if self._token_refresh_task and not self._token_refresh_task.done():
            self._token_refresh_task.cancel()
            try:
                await self._token_refresh_task
            except asyncio.CancelledError:
                pass

        if self._client is not None:
            try:
                self._client.stopService()
            except Exception as e:
                logger.warning(f"Error stopping cTrader client: {e}")
            self._client = None

        self._connected = False
        logger.info("CTraderBroker disconnected")

    # ---------------------------------------------------------------
    # Connection health check + auto-reconnect
    # ---------------------------------------------------------------

    def _check_connection(self) -> bool:
        """Check whether the TCP connection to cTrader is alive.

        Returns True if the client exists and is marked connected.
        The ``_on_disconnected`` callback sets ``_connected = False``
        when the Twisted transport drops, so this is a reliable check.
        """
        if self._client is None:
            return False
        if not self._connected:
            return False
        # Twisted transport-level check (if available)
        try:
            transport = getattr(self._client, "_TcpClient__transport", None)
            if transport is not None and getattr(transport, "connected", None) is False:
                return False
        except Exception:
            pass
        return True

    async def _reconnect(self) -> bool:
        """Re-establish the cTrader TCP connection and full auth flow.

        Uses exponential backoff: 5s, 10s, 20s, ... between attempts.
        Returns True if reconnection succeeded, False if all attempts
        exhausted.

        Thread-safe: only one reconnect attempt runs at a time via an
        asyncio lock. Concurrent callers wait for the result.
        """
        async with self._reconnect_lock:
            # Another coroutine may have reconnected while we waited
            if self._check_connection():
                return True

            base_delay = 5.0
            for attempt in range(1, self._max_reconnect_attempts + 1):
                delay = base_delay * (2 ** (attempt - 1))  # 5, 10, 20, ...
                self._consecutive_reconnects += 1

                logger.warning(
                    "cTrader reconnect attempt %d/%d "
                    "(backoff %.0fs, consecutive=%d)",
                    attempt,
                    self._max_reconnect_attempts,
                    delay,
                    self._consecutive_reconnects,
                )

                # Tear down old client cleanly
                await self._teardown_client()

                if attempt > 1:
                    await asyncio.sleep(delay)

                try:
                    # Refresh the OAuth token before reconnecting if we
                    # have a refresh token — the old access token may
                    # have expired while the connection was down.
                    if self._refresh_token:
                        await self._try_refresh_token()

                    # Reuse the existing connect() method which handles
                    # the full 7-step auth flow.
                    await self.connect()

                    if self._connected:
                        logger.warning(
                            "cTrader reconnected successfully on attempt %d",
                            attempt,
                        )
                        self._consecutive_reconnects = 0
                        return True

                except Exception as exc:
                    logger.warning(
                        "cTrader reconnect attempt %d failed: %s",
                        attempt,
                        exc,
                    )

            logger.error(
                "cTrader reconnect FAILED after %d attempts",
                self._max_reconnect_attempts,
            )
            return False

    async def _teardown_client(self) -> None:
        """Stop the old Twisted client and cancel the token refresh task.

        Called before reconnecting to ensure a clean slate.  Does NOT
        reset ``_position_lots`` or ``_portfolio_value`` — those must
        survive reconnection so the broker doesn't lose position state.
        """
        # Cancel token refresh task
        if self._token_refresh_task and not self._token_refresh_task.done():
            self._token_refresh_task.cancel()
            try:
                await self._token_refresh_task
            except asyncio.CancelledError:
                pass
            self._token_refresh_task = None

        # Stop Twisted client
        if self._client is not None:
            try:
                self._client.stopService()
            except Exception as e:
                logger.debug("Error stopping cTrader client during teardown: %s", e)
            self._client = None

        self._connected = False

    async def _try_refresh_token(self) -> None:
        """Attempt a single token refresh.  Non-fatal on failure."""
        try:
            from ctrader_open_api import Auth

            token_data = Auth.refreshToken(
                refreshToken=self._refresh_token,
                clientId=self._client_id,
                clientSecret=self._client_secret,
            )
            if "accessToken" in token_data:
                self._access_token = token_data["accessToken"]
                if "refreshToken" in token_data:
                    self._refresh_token = token_data["refreshToken"]
                logger.info("Access token refreshed during reconnect")
            else:
                logger.warning("Token refresh returned no accessToken: %s", token_data)
        except Exception as exc:
            logger.warning("Token refresh failed during reconnect: %s", exc)

    @_with_reconnect
    async def execute_position_change(
        self,
        asset: str,
        current_position: float,
        target_position: float,
        portfolio_value: float,
    ) -> OrderResult:
        """Execute a position change on XAUUSD CFD.

        Args:
            asset: Asset symbol (used for logging).
            current_position: Current position fraction [-1, 1].
            target_position: Target position fraction [-1, 1].
            portfolio_value: Current portfolio value in USD.

        Returns:
            OrderResult with execution details.
        """
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOANewOrderReq,
        )
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
            ProtoOAOrderType,
            ProtoOATradeSide,
        )

        self._portfolio_value = portfolio_value
        price = self._mid_price
        if price <= 0:
            return OrderResult(
                asset=asset,
                symbol=self._symbol_name,
                side="none",
                order_type="failed",
                quantity=0.0,
                price=0.0,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                fee=0.0,
                status="failed",
                error="No price available — spot subscription may not be active",
            )

        target_lots = self._position_to_lots(target_position, portfolio_value, price)
        current_lots = self._position_lots
        delta_lots = target_lots - current_lots

        # Skip dust trades
        if abs(delta_lots) < self._min_lot:
            return OrderResult(
                asset=asset,
                symbol=self._symbol_name,
                side="none",
                order_type="skipped",
                quantity=abs(delta_lots),
                price=price,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                fee=0.0,
                status="skipped",
                error=f"Delta {delta_lots:.4f} lots below min {self._min_lot}",
            )

        # Round to min_lot precision
        delta_lots_rounded = round(delta_lots / self._min_lot) * self._min_lot
        if abs(delta_lots_rounded) < self._min_lot:
            delta_lots_rounded = self._min_lot * (1 if delta_lots > 0 else -1)

        side = "buy" if delta_lots_rounded > 0 else "sell"
        api_side = ProtoOATradeSide.BUY if delta_lots_rounded > 0 else ProtoOATradeSide.SELL
        api_volume = int(round(abs(delta_lots_rounded) * _VOLUME_SCALE))

        if api_volume < 1:
            return OrderResult(
                asset=asset,
                symbol=self._symbol_name,
                side=side,
                order_type="skipped",
                quantity=abs(delta_lots_rounded),
                price=price,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                fee=0.0,
                status="skipped",
                error="Volume rounds to 0",
            )

        logger.info(
            f"Placing {side.upper()} {abs(delta_lots_rounded):.2f} lots "
            f"{self._symbol_name} @ ~{price:.2f} (volume={api_volume})"
        )

        try:
            order_req = ProtoOANewOrderReq()
            order_req.ctidTraderAccountId = self._account_id
            order_req.symbolId = self._symbol_id
            order_req.orderType = ProtoOAOrderType.MARKET
            order_req.tradeSide = api_side
            order_req.volume = api_volume

            exec_res = await self._send_request(
                order_req, timeout=self._market_fallback_timeout
            )
            exec_payload = Protobuf.extract(exec_res)

            # Parse execution event
            fill_price = price  # Default
            filled_volume = api_volume
            order_id = ""

            if exec_payload.HasField("order"):
                order = exec_payload.order
                order_id = str(order.orderId)
                if order.executionPrice:
                    fill_price = order.executionPrice / (10 ** self._symbol_digits)
                if order.executedVolume:
                    filled_volume = order.executedVolume

            if exec_payload.HasField("deal"):
                deal = exec_payload.deal
                if deal.executionPrice:
                    fill_price = deal.executionPrice / (10 ** self._symbol_digits)
                if deal.filledVolume:
                    filled_volume = deal.filledVolume

            filled_lots = filled_volume / _VOLUME_SCALE
            filled_notional = filled_lots * self._lot_size * fill_price
            fee = filled_notional * self._taker_fee

            # Update internal position
            if side == "buy":
                self._position_lots += filled_lots
            else:
                self._position_lots -= filled_lots

            logger.info(
                f"Filled {side.upper()} {filled_lots:.2f} lots @ {fill_price:.2f}, "
                f"fee=${fee:.2f}, position now {self._position_lots:.2f} lots"
            )

            return OrderResult(
                asset=asset,
                symbol=self._symbol_name,
                side=side,
                order_type="market",
                quantity=abs(delta_lots_rounded),
                price=price,
                filled_quantity=filled_lots,
                avg_fill_price=fill_price,
                fee=fee,
                status="filled",
                order_id=order_id,
            )

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return OrderResult(
                asset=asset,
                symbol=self._symbol_name,
                side=side,
                order_type="market",
                quantity=abs(delta_lots_rounded),
                price=price,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                fee=0.0,
                status="failed",
                error=str(e),
            )

    @_with_reconnect
    async def get_single_position(self, asset: str) -> float:
        """Fetch current position for XAUUSD as a signed fraction [-1, 1]."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAReconcileReq,
        )

        try:
            recon_req = ProtoOAReconcileReq()
            recon_req.ctidTraderAccountId = self._account_id
            recon_res = await self._send_request(recon_req, timeout=10.0)

            recon_payload = Protobuf.extract(recon_res)
            signed_lots = 0.0

            for pos in recon_payload.position:
                trade_data = pos.tradeData
                if trade_data.symbolId == self._symbol_id:
                    lots = trade_data.volume / _VOLUME_SCALE
                    if trade_data.tradeSide == 2:  # SELL
                        lots = -lots
                    signed_lots += lots

            self._position_lots = signed_lots
            price = self._mid_price if self._mid_price > 0 else 1.0
            fraction = self._lots_to_position(
                signed_lots, self._portfolio_value, price
            )
            return fraction

        except Exception as e:
            logger.error(f"Failed to fetch position: {e}")
            return 0.0

    @_with_reconnect
    async def get_account_info(self) -> dict:
        """Fetch account equity, available balance, and used margin."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAGetPositionUnrealizedPnLReq,
            ProtoOAReconcileReq,
            ProtoOATraderReq,
        )

        try:
            # Get balance
            trader_req = ProtoOATraderReq()
            trader_req.ctidTraderAccountId = self._account_id
            trader_res = await self._send_request(trader_req, timeout=10.0)

            trader_payload = Protobuf.extract(trader_res)
            trader = trader_payload.trader
            balance = trader.balance / (10 ** self._money_digits)

            # Get used margin from open positions
            recon_req = ProtoOAReconcileReq()
            recon_req.ctidTraderAccountId = self._account_id
            recon_res = await self._send_request(recon_req, timeout=10.0)

            recon_payload = Protobuf.extract(recon_res)
            used_margin = 0.0
            for pos in recon_payload.position:
                if pos.usedMargin:
                    used_margin += pos.usedMargin / (10 ** self._money_digits)

            # Get unrealized PnL (NOT swap — swap is overnight financing)
            unrealized_pnl = 0.0
            if recon_payload.position:
                try:
                    pnl_req = ProtoOAGetPositionUnrealizedPnLReq()
                    pnl_req.ctidTraderAccountId = self._account_id
                    pnl_res = await self._send_request(pnl_req, timeout=10.0)
                    pnl_payload = Protobuf.extract(pnl_res)

                    pnl_digits = pnl_payload.moneyDigits or self._money_digits
                    for pnl in pnl_payload.positionUnrealizedPnL:
                        unrealized_pnl += pnl.netUnrealizedPnL / (10 ** pnl_digits)
                except Exception as e:
                    logger.warning(f"Failed to fetch unrealized PnL: {e}")

            total_equity = balance + unrealized_pnl
            available = total_equity - used_margin
            self._portfolio_value = total_equity

            return {
                "total_equity": total_equity,
                "available_balance": available,
                "used_margin": used_margin,
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {
                "total_equity": self._portfolio_value,
                "available_balance": self._portfolio_value,
                "used_margin": 0.0,
            }

    async def get_funding_rates(self, assets: list[str]) -> dict[str, float]:
        """CFDs have no funding rates. Return zeros."""
        return {a: 0.0 for a in assets}

    @_with_reconnect
    async def emergency_flatten(self, assets: list[str]) -> RebalanceResult:
        """Close all positions via market orders."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAClosePositionReq,
            ProtoOAReconcileReq,
        )

        orders = []
        n_failed = 0

        for attempt in range(3):
            try:
                recon_req = ProtoOAReconcileReq()
                recon_req.ctidTraderAccountId = self._account_id
                recon_res = await self._send_request(recon_req, timeout=10.0)
                recon_payload = Protobuf.extract(recon_res)

                open_positions = [
                    p for p in recon_payload.position
                    if p.tradeData.symbolId == self._symbol_id
                ]

                if not open_positions:
                    logger.info("No open positions to flatten")
                    self._position_lots = 0.0
                    break

                # Reset per-attempt counters (orders accumulate across retries
                # intentionally for full audit trail, but n_failed resets)
                n_failed = 0
                for pos in open_positions:
                    try:
                        close_req = ProtoOAClosePositionReq()
                        close_req.ctidTraderAccountId = self._account_id
                        close_req.positionId = pos.positionId
                        close_req.volume = pos.tradeData.volume

                        await self._send_request(
                            close_req, timeout=self._market_fallback_timeout
                        )

                        lots = pos.tradeData.volume / _VOLUME_SCALE
                        side = "sell" if pos.tradeData.tradeSide == 1 else "buy"
                        orders.append(OrderResult(
                            asset=self._symbol_name,
                            symbol=self._symbol_name,
                            side=side,
                            order_type="market",
                            quantity=lots,
                            price=self._mid_price,
                            filled_quantity=lots,
                            avg_fill_price=self._mid_price,
                            fee=lots * self._lot_size * self._mid_price * self._taker_fee,
                            status="filled",
                        ))
                        logger.info(
                            f"Emergency flatten: closed position {pos.positionId} "
                            f"({lots:.2f} lots)"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to close position {pos.positionId}: {e}"
                        )
                        n_failed += 1
                        orders.append(OrderResult(
                            asset=self._symbol_name,
                            symbol=self._symbol_name,
                            side="unknown",
                            order_type="market",
                            quantity=pos.tradeData.volume / _VOLUME_SCALE,
                            price=0.0,
                            filled_quantity=0.0,
                            avg_fill_price=0.0,
                            fee=0.0,
                            status="failed",
                            error=str(e),
                        ))

                if n_failed == 0:
                    self._position_lots = 0.0
                    break

                logger.warning(
                    f"Emergency flatten attempt {attempt + 1}: "
                    f"{n_failed} failures, retrying..."
                )
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Emergency flatten attempt {attempt + 1} failed: {e}")
                n_failed += 1
                await asyncio.sleep(1.0)

        total_fees = sum(o.fee for o in orders)
        return RebalanceResult(
            orders=orders,
            total_fees=total_fees,
            n_executed=sum(1 for o in orders if o.status == "filled"),
            n_failed=n_failed,
        )

    # ---------------------------------------------------------------
    # Position <-> Lots conversion
    # ---------------------------------------------------------------

    def _position_to_lots(
        self, fraction: float, portfolio_value: float, price: float
    ) -> float:
        """Convert position fraction [-1, 1] to signed lots.

        Example: $100K portfolio, Gold @ $3000/oz, 100 oz/lot
            fraction=1.0 → notional=$100K → lots = 100000/(3000*100) = 0.333

        When portfolio < lot notional (e.g. XAUUSD lot=$470K vs $200K portfolio),
        round() gives 0 for most fractions.  We round up to min_lot when the
        agent expresses meaningful conviction (fraction >= 0.1), matching
        the IB broker's contract-floor logic.  The broker's margin system
        provides the real leverage safety net.
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0
        notional = abs(fraction) * portfolio_value
        lots = notional / (price * self._lot_size)
        # Round to min_lot precision
        lots = round(lots / self._min_lot) * self._min_lot
        # Floor: at least min_lot when agent has conviction (BUG-18 fix)
        if lots < self._min_lot and abs(fraction) >= 0.1:
            lots = self._min_lot
        return lots * np.sign(fraction)

    def _lots_to_position(
        self, lots: float, portfolio_value: float, price: float
    ) -> float:
        """Convert signed lots to position fraction [-1, 1]."""
        if portfolio_value <= 0 or price <= 0:
            return 0.0
        notional = abs(lots) * price * self._lot_size
        fraction = notional / portfolio_value
        return float(np.clip(fraction * np.sign(lots), -1.0, 1.0))

    @staticmethod
    def lots_to_api_volume(lots: float) -> int:
        """Convert lots to cTrader API volume units (hundredths).

        1.00 lot = volume 100
        0.01 lot = volume 1
        """
        return int(round(abs(lots) * _VOLUME_SCALE))

    @staticmethod
    def api_volume_to_lots(volume: int) -> float:
        """Convert cTrader API volume units back to lots."""
        return volume / _VOLUME_SCALE

    # ---------------------------------------------------------------
    # Twisted/asyncio bridge + message handling
    # ---------------------------------------------------------------

    @staticmethod
    def _check_error(payload, step: str) -> None:
        """Raise on ProtoOAErrorRes with a clear message."""
        if hasattr(payload, "errorCode"):
            raise RuntimeError(
                f"cTrader {step} failed: {payload.errorCode} — "
                f"{getattr(payload, 'description', 'no description')}"
            )

    def _next_msg_id(self) -> str:
        """Generate a unique client message ID."""
        self._msg_id_counter += 1
        return f"finrl_{self._msg_id_counter}_{int(time.time() * 1000)}"

    async def _send_request(self, message, timeout: float = 10.0):
        """Send a protobuf request and await the response.

        Uses the SDK's built-in Deferred-based send() method with a
        client message ID to match request/response pairs.

        Raises RuntimeError if the connection is down, which is caught
        by the ``@_with_reconnect`` decorator on public API methods.
        """
        if self._client is None or not self._connected:
            raise RuntimeError("Not connected to cTrader")

        msg_id = self._next_msg_id()

        response_deferred = self._client.send(
            message,
            clientMsgId=msg_id,
            responseTimeoutInSeconds=timeout,
        )

        return await self._deferred_to_future(response_deferred, timeout=timeout + 2)

    async def _deferred_to_future(self, deferred, timeout: float = 15.0):
        """Convert a Twisted Deferred to an asyncio-awaitable result.

        Works by polling the Deferred's callback chain. This avoids
        needing the asyncio reactor to be installed (which is fragile).
        """
        result_container = {"value": None, "error": None, "done": False}

        def on_success(result):
            result_container["value"] = result
            result_container["done"] = True
            return result

        def on_error(failure):
            result_container["error"] = failure
            result_container["done"] = True

        deferred.addCallback(on_success)
        deferred.addErrback(on_error)

        # Poll until resolved
        deadline = time.monotonic() + timeout
        while not result_container["done"]:
            if time.monotonic() > deadline:
                raise asyncio.TimeoutError(
                    f"cTrader request timed out after {timeout}s"
                )
            await asyncio.sleep(0.05)

        if result_container["error"] is not None:
            failure = result_container["error"]
            if hasattr(failure, "raiseException"):
                failure.raiseException()
            raise RuntimeError(f"cTrader request failed: {failure}")

        return result_container["value"]

    def _on_message(self, client, message):
        """Callback for incoming messages from cTrader."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
            ProtoOAPayloadType,
        )

        payload_type = message.payloadType

        # Handle spot price updates
        if payload_type == ProtoOAPayloadType.PROTO_OA_SPOT_EVENT:
            try:
                spot = Protobuf.extract(message)
                if spot.symbolId == self._symbol_id:
                    if spot.HasField("bid"):
                        self._bid = spot.bid / (10 ** self._symbol_digits)
                    if spot.HasField("ask"):
                        self._ask = spot.ask / (10 ** self._symbol_digits)
                    if self._bid > 0 and self._ask > 0:
                        self._mid_price = (self._bid + self._ask) / 2.0
            except Exception as e:
                logger.debug(f"Error parsing spot event: {e}")

        # Handle execution events (logged for monitoring)
        elif payload_type == ProtoOAPayloadType.PROTO_OA_EXECUTION_EVENT:
            logger.debug(f"Execution event received (payloadType={payload_type})")

        # Handle errors
        elif payload_type == ProtoOAPayloadType.PROTO_OA_ERROR_RES:
            try:
                err = Protobuf.extract(message)
                logger.error(
                    f"cTrader error: {err.errorCode} — {err.description}"
                    if hasattr(err, "description") else
                    f"cTrader error: {err.errorCode}"
                )
            except Exception:
                logger.error(f"cTrader error (payloadType={payload_type})")

    def _on_disconnected(self, client, reason):
        """Callback when connection drops.

        Sets ``_connected = False`` so that the next API call via
        ``@_with_reconnect`` triggers an automatic reconnection.
        """
        was_connected = self._connected
        self._connected = False
        if was_connected:
            logger.warning(
                "cTrader TCP connection lost (reason: %s). "
                "Next API call will trigger auto-reconnect.",
                reason,
            )
        else:
            logger.debug("cTrader disconnected callback (was already disconnected): %s", reason)

    async def _token_refresh_loop(self) -> None:
        """Background task to refresh OAuth access token before expiry.

        cTrader access tokens typically expire in 2-3 hours.
        Refreshes 10 minutes before estimated expiry.
        """
        from ctrader_open_api import Auth

        refresh_interval = 7200  # Assume 2h expiry, refresh at 1h50m
        retry_delays = [30, 60, 120]

        while True:
            try:
                await asyncio.sleep(refresh_interval - 600)  # 10 min before expiry

                if not self._refresh_token:
                    logger.warning("No refresh token — cannot auto-refresh")
                    return

                logger.info("Refreshing cTrader access token...")
                token_data = Auth.refreshToken(
                    refreshToken=self._refresh_token,
                    clientId=self._client_id,
                    clientSecret=self._client_secret,
                )

                if "accessToken" in token_data:
                    self._access_token = token_data["accessToken"]
                    if "refreshToken" in token_data:
                        self._refresh_token = token_data["refreshToken"]
                    logger.info("Access token refreshed successfully")
                else:
                    logger.error(f"Token refresh failed: {token_data}")
                    for delay in retry_delays:
                        await asyncio.sleep(delay)
                        token_data = Auth.refreshToken(
                            refreshToken=self._refresh_token,
                            clientId=self._client_id,
                            clientSecret=self._client_secret,
                        )
                        if "accessToken" in token_data:
                            self._access_token = token_data["accessToken"]
                            logger.info("Token refresh succeeded on retry")
                            break
                    else:
                        logger.critical(
                            "All token refresh retries failed. "
                            "Trading will fail when current token expires."
                        )

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Token refresh error: {e}")
                await asyncio.sleep(60)
