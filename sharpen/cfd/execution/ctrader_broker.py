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
import calendar
import functools
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from sharpen.crypto.execution.exchange_perp_broker import (
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

# Volume encoding: cTrader Open API represents order/position `volume` in
# 0.01 of a base-currency unit (centi-units). `lotSize` and `minVolume` on
# ProtoOASymbol are also in centi-units.  So:
#   api_volume   = lots × lotSize_raw          (cents of base / lot × lots)
#   lots         = api_volume / lotSize_raw
#   units_per_lot = lotSize_raw / _VOLUME_SCALE (e.g. XAUUSD: 10000 → 100 oz)
# The lot↔volume conversion therefore needs `_lot_size_raw` (instance state),
# NOT a constant — earlier code multiplied lots by _VOLUME_SCALE directly,
# which produced volume=100 for 1 XAUUSD lot and caused the broker to fill
# only 1 oz while engine bookkeeping counted 100 oz (100× phantom sizing,
# root cause of the S458 gmgp1-xauusd halt).
_VOLUME_SCALE = 100

# Price encoding: cTrader spot event bid/ask (uint64) are always scaled by 10^5,
# regardless of the symbol's display digits.  E.g. XAUUSD at $4686.055 → 468605500.
# Double fields (Position.price, Deal/Order.executionPrice) are already decimal.
_SPOT_PRICE_SCALE = 100_000


class CTraderBroker:
    """cTrader CFD broker for XAUUSD paper and live trading.

    Uses ctrader-open-api for connectivity via Protobuf over TCP.
    Supports IC Markets demo (paper) and FTMO live (challenge) accounts.
    """

    # FIX CT-04: Reconnect timing constants
    _RECONNECT_COOLDOWN_SECS = 120  # Cooldown after exhausting all attempts
    _ALREADY_LOGGED_IN_WAIT = 150  # Wait for server ghost-session to expire
    _SERVER_ISSUE_COOLDOWN_SECS = 300  # Longer cooldown for CANT_ROUTE_REQUEST

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
        # Raw cTrader `lotSize` in centi-units of the base currency — the
        # authoritative scale for every lot↔volume conversion on this broker.
        # Populated on connect() from ProtoOASymbol.lotSize; seeded here from
        # the constructor default so tests that skip connect() still work.
        self._lot_size_raw = int(round(lot_size * _VOLUME_SCALE))
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

        # FIND-03: token state file — in-memory token rotations are persisted
        # here so a container restart picks up the latest refreshed tokens
        # instead of falling back to the stale `.env` snapshot. State dir is
        # a named Docker volume (`/app/state`), so the file survives restarts.
        # File is chmod 0600 (secrets).
        state_path_env = os.environ.get(
            "CTRADER_TOKEN_STATE_FILE",
            "/app/state/ctrader_tokens.json",
        )
        self._token_state_file = Path(state_path_env)
        self._token_state_loaded = False
        # Epoch seconds of the most recent rotation we know about
        # (loaded from state file or set on every successful refresh).
        # 0.0 means unknown — proactive pre-connect refresh skips that case
        # to avoid burning a refresh on every fresh-OAuth deploy.
        self._token_acquired_at: float = 0.0
        # Recovery threshold: if the cached access token is older than this
        # at connect() time, refresh BEFORE Step 2 Account auth so the
        # engine can recover after a >2h docker-restart-loop outage (S490).
        # Default 3600s (1h) — well below the ~2h cTrader TTL.
        try:
            self._proactive_refresh_age_sec: float = float(
                os.environ.get("CTRADER_PROACTIVE_REFRESH_AGE_SEC", "3600")
            )
        except ValueError:
            self._proactive_refresh_age_sec = 3600.0
        self._load_tokens_from_state_file()

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
        self._reconnect_cooldown_until = 0.0  # FIX CT-04: prevent reconnect spam
        self._already_logged_in_seen = False  # FIX CT-07: flag set by _on_message

    @property
    def client(self):
        """Active ctrader_open_api.Client instance (None if disconnected)."""
        return self._client

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

        # S490 follow-up #2: proactive pre-connect refresh.
        # Cached access token may be stale after a >2h docker restart loop.
        # If we know the rotation timestamp and it's older than the threshold,
        # refresh BEFORE Step 2 Account auth so we don't waste a TCP+handshake
        # round-trip just to fail with CH_ACCESS_TOKEN_INVALID. Failure here is
        # non-fatal — fall through and let Account auth surface the visible
        # error if the cached token is also dead.
        await self._maybe_proactive_refresh()

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

        # FIX CT-11: Wrap TCP connect + handshake in try/except so any failure
        # (timeout, auth, symbol resolve) tears down the connection.
        try:
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
                    self._lot_size_raw = int(sym_info.lotSize)
                    self._lot_size = self._lot_size_raw / _VOLUME_SCALE
                # minVolume is in centi-units; convert to lots via lotSize_raw
                # (NOT via _VOLUME_SCALE — that confuses centi-units with lots
                # and under-reports min_lot by a factor of units-per-lot).
                if sym_info.minVolume and self._lot_size_raw > 0:
                    self._min_lot = sym_info.minVolume / self._lot_size_raw
                logger.info(
                    f"Symbol {self._symbol_name}: id={self._symbol_id}, "
                    f"digits={self._symbol_digits}, "
                    f"lot_size={self._lot_size} units/lot, "
                    f"min_lot={self._min_lot:.4f} lots, "
                    f"lot_size_raw={self._lot_size_raw} centi-units/lot"
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

            # FIX CT-09: Post-connect health verification
            await self.get_account_info()

        except Exception as exc:
            logger.error(
                "CT-11: Protocol handshake failed after TCP connect — "
                "tearing down so retry starts fresh: %s",
                exc,
            )
            self._connected = False
            self._symbol_id = None
            if self._token_refresh_task and not self._token_refresh_task.done():
                self._token_refresh_task.cancel()
                self._token_refresh_task = None
            if self._client is not None:
                try:
                    self._client.stopService()
                except Exception:
                    pass
                self._client = None
            raise

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

        # FIX CT-04: Send account logout so server releases session
        if self._client is not None and self._connected:
            try:
                await self._send_account_logout()
            except Exception:
                pass  # Best-effort

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

        FIX CT-04: After exhausting attempts, enters a cooldown period
        to prevent reconnect spam. On ALREADY_LOGGED_IN, waits for the
        server ghost-session to expire instead of hammering new TCP
        connections (which reset the server's idle timer).

        Thread-safe: only one reconnect attempt runs at a time via an
        asyncio lock. Concurrent callers wait for the result.
        """
        async with self._reconnect_lock:
            # Another coroutine may have reconnected while we waited
            if self._check_connection():
                return True

            # FIX CT-04: Respect cooldown from previous failed cycle
            now = time.time()
            if now < self._reconnect_cooldown_until:
                remaining = self._reconnect_cooldown_until - now
                logger.debug(
                    "cTrader reconnect in cooldown (%.0fs remaining)",
                    remaining,
                )
                return False

            base_delay = 5.0
            already_logged_in_seen = False
            # FIX CT-07: Reset the async flag before the reconnect cycle
            self._already_logged_in_seen = False

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

                await asyncio.sleep(min(delay, 2.0) if attempt == 1 else delay)

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
                        self._reconnect_cooldown_until = 0.0
                        return True

                except Exception as exc:
                    exc_str = str(exc)
                    logger.warning(
                        "cTrader reconnect attempt %d failed: %s",
                        attempt,
                        exc,
                    )

                    # FIX CT-04/CT-07: On ALREADY_LOGGED_IN, stop immediately
                    # — more TCP connections just reset the server's idle
                    # timer on the ghost session.  Wait for it to expire,
                    # then make one final attempt.
                    # CT-07: Also check _already_logged_in_seen flag set by
                    # _on_message, since the Deferred may time out with a
                    # generic TimeoutError instead of propagating the error.
                    if (
                        "ALREADY_LOGGED_IN" in exc_str
                        or self._already_logged_in_seen
                    ):
                        already_logged_in_seen = True
                        logger.warning(
                            "CT-07/AUD: Ghost session detected (ALREADY_LOGGED_IN) — "
                            "waiting %ds for server-side expiry before FINAL attempt",
                            self._ALREADY_LOGGED_IN_WAIT,
                        )
                        await self._teardown_client()
                        self._already_logged_in_seen = False  # Reset for final try
                        await asyncio.sleep(self._ALREADY_LOGGED_IN_WAIT)

                        try:
                            if self._refresh_token:
                                await self._try_refresh_token()
                            await self.connect()
                            if self._connected:
                                logger.warning(
                                    "cTrader reconnected after ghost-session wait"
                                )
                                self._consecutive_reconnects = 0
                                self._reconnect_cooldown_until = 0.0
                                return True
                        except Exception as exc2:
                            logger.error(
                                "Final reconnect after ghost-session wait "
                                "failed: %s",
                                exc2,
                            )
                        break  # Don't retry further — enter cooldown

                    # FIX AUD: Handle CANT_ROUTE_REQUEST server-side infrastructure issue.
                    # Usually happens when IC Markets Demo is disconnected from Open API.
                    if "CANT_ROUTE_REQUEST" in exc_str:
                        logger.error(
                            "AUD: cTrader server error (CANT_ROUTE_REQUEST) — "
                            "IC Markets Demo server likely down. Entering "
                            "longer %ds cooldown.",
                            self._SERVER_ISSUE_COOLDOWN_SECS,
                        )
                        self._reconnect_cooldown_until = (
                            time.time() + self._SERVER_ISSUE_COOLDOWN_SECS
                        )
                        await self._teardown_client()
                        return False

            logger.error(
                "cTrader reconnect FAILED after %d attempts%s",
                self._max_reconnect_attempts,
                " (ghost session)" if already_logged_in_seen else "",
            )

            # FIX CT-04: Enter cooldown to prevent reconnect spam.
            # The engine loop and @_with_reconnect both trigger _reconnect()
            # on every API call — without cooldown this creates non-stop
            # reconnect cycles that waste resources and spam the server.
            self._reconnect_cooldown_until = (
                time.time() + self._RECONNECT_COOLDOWN_SECS
            )
            logger.warning(
                "cTrader entering %ds reconnect cooldown",
                self._RECONNECT_COOLDOWN_SECS,
            )
            return False

    async def _teardown_client(self) -> None:
        """Stop the old Twisted client and cancel the token refresh task.

        Called before reconnecting to ensure a clean slate.  Does NOT
        reset ``_position_lots`` or ``_portfolio_value`` — those must
        survive reconnection so the broker doesn't lose position state.

        FIX CT-04: Sends account logout before closing TCP so the server
        releases the session immediately instead of waiting for timeout.
        """
        # Cancel token refresh task
        if self._token_refresh_task and not self._token_refresh_task.done():
            self._token_refresh_task.cancel()
            try:
                await self._token_refresh_task
            except asyncio.CancelledError:
                pass
            self._token_refresh_task = None

        # FIX CT-04: Send account logout before closing TCP.
        # This tells the server to release the session so the next
        # connect() doesn't get ALREADY_LOGGED_IN.
        if self._client is not None and self._connected:
            try:
                await self._send_account_logout()
            except Exception:
                pass  # Best-effort — connection may already be dead

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
                self._persist_tokens_to_state_file()
                logger.info("Access token refreshed during reconnect")
            else:
                logger.warning("Token refresh returned no accessToken: %s", token_data)
        except Exception as exc:
            logger.warning("Token refresh failed during reconnect: %s", exc)

    def _should_proactive_refresh(self) -> bool:
        """Decide whether connect() should refresh the token before Step 2.

        Returns True when:
        - we have a refresh_token (no point trying without one), AND
        - the cached access token's age is known (loaded from state file or
          set by a prior in-process refresh), AND
        - that age exceeds ``_proactive_refresh_age_sec``.

        We deliberately skip cold starts (``_token_acquired_at == 0``) so a
        fresh-OAuth deploy doesn't burn a refresh on the very first connect —
        in that path the operator just minted the token and it's valid.
        """
        if not self._refresh_token:
            return False
        if self._token_acquired_at <= 0:
            return False
        age = time.time() - self._token_acquired_at
        return age > self._proactive_refresh_age_sec

    async def _maybe_proactive_refresh(self) -> None:
        """Refresh the access token before connect() if it's likely expired.

        Uses ``Auth.refreshToken`` (synchronous HTTP, no TCP session needed)
        run in an executor so the event loop stays responsive. Failure is
        logged and swallowed — connect() proceeds with whatever token is in
        memory and Account auth surfaces the real error if both are dead.
        """
        if not self._should_proactive_refresh():
            return

        age = time.time() - self._token_acquired_at
        logger.info(
            "Proactive cTrader token refresh: age=%.0fs > threshold=%.0fs",
            age, self._proactive_refresh_age_sec,
        )
        try:
            from ctrader_open_api import Auth

            loop = asyncio.get_running_loop()
            token_data = await loop.run_in_executor(
                None,
                lambda: Auth.refreshToken(
                    refreshToken=self._refresh_token,
                    clientId=self._client_id,
                    clientSecret=self._client_secret,
                ),
            )
            if "accessToken" in token_data:
                self._access_token = token_data["accessToken"]
                if "refreshToken" in token_data:
                    self._refresh_token = token_data["refreshToken"]
                self._persist_tokens_to_state_file()
                logger.info("Access token refreshed (proactive pre-connect)")
            else:
                logger.warning(
                    "Proactive token refresh returned no accessToken: %s",
                    token_data,
                )
        except Exception as exc:
            logger.warning(
                "Proactive token refresh failed (%s) — proceeding with "
                "cached token; Account auth will surface the real error "
                "if it's also dead.",
                exc,
            )

    # ------------------------------------------------------------------
    # FIND-03: token state-file persistence across container restarts
    # ------------------------------------------------------------------
    def _load_tokens_from_state_file(self) -> None:
        """Prefer tokens from state file over `.env` when both exist.

        The S490 crash-storm root cause: `_token_refresh_loop` rotated tokens
        in memory, but the `.env` file was never updated. Restart → stale env
        tokens → CH_ACCESS_TOKEN_INVALID → refresh fails with ACCESS_DENIED.
        Loading from a persistent state file closes that gap.
        """
        if self._token_state_loaded:
            return
        try:
            if not self._token_state_file.exists():
                return
            data = json.loads(self._token_state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Token state file unreadable at %s (%s) — falling back to env.",
                self._token_state_file, exc,
            )
            return

        try:
            account = int(data.get("account_id", 0) or 0)
        except (TypeError, ValueError):
            account = 0
        if self._account_id and account and account != self._account_id:
            logger.warning(
                "Token state file account_id=%s does not match configured %s — "
                "ignoring stale tokens.",
                account, self._account_id,
            )
            return

        access = data.get("access_token") or ""
        refresh = data.get("refresh_token") or ""
        if access:
            self._access_token = access
        if refresh:
            self._refresh_token = refresh
        rotated_at_iso = data.get("rotated_at") or ""
        if rotated_at_iso:
            try:
                # Stored as UTC ISO ("YYYY-MM-DDTHH:MM:SSZ") — parse without
                # tz fallback so a rogue local-time entry is rejected loudly.
                ts_struct = time.strptime(rotated_at_iso, "%Y-%m-%dT%H:%M:%SZ")
                self._token_acquired_at = float(calendar.timegm(ts_struct))
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Token state rotated_at unparseable (%s): %s — "
                    "treating token age as unknown",
                    rotated_at_iso, exc,
                )
                self._token_acquired_at = 0.0
        self._token_state_loaded = True
        logger.info(
            "Loaded cTrader tokens from state file %s (rotated_at=%s)",
            self._token_state_file, rotated_at_iso or "unknown",
        )

    def _persist_tokens_to_state_file(self) -> None:
        """Atomically write the current tokens to the state file, chmod 0600.

        Best-effort: a failed write logs but does not abort trading — the
        rotated tokens still live in memory for the current session.
        """
        if not self._access_token:
            return
        path = self._token_state_file
        rotated_epoch = time.time()
        payload = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "account_id": self._account_id,
            "client_id": self._client_id,
            "rotated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(rotated_epoch)
            ),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to tmp in the same directory, then atomic rename.
            fd, tmp = tempfile.mkstemp(
                dir=str(path.parent), prefix=".ctrader_tokens.", suffix=".json.tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.chmod(tmp, 0o600)
                os.replace(tmp, str(path))
                tmp = None
            finally:
                if tmp is not None and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            self._token_acquired_at = rotated_epoch
            logger.debug("Persisted cTrader tokens to %s", path)
        except Exception as exc:
            logger.warning(
                "Failed to persist cTrader tokens to %s: %s (in-memory only).",
                path, exc,
            )
            # Even if disk write failed, the in-memory tokens were just
            # rotated — track that age so a subsequent connect() within
            # this process doesn't proactively re-refresh on a fresh token.
            self._token_acquired_at = rotated_epoch

    async def _send_account_logout(self) -> None:
        """FIX CT-04: Send account logout so server releases the session.

        Best-effort — if the connection is already dead this will fail
        silently.  The goal is to prevent ALREADY_LOGGED_IN on the next
        connect() after a restart or reconnect.
        """
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAAccountLogoutReq,
            )

            logout_req = ProtoOAAccountLogoutReq()
            logout_req.ctidTraderAccountId = self._account_id
            await self._send_request(logout_req, timeout=5.0)
            logger.debug("Account logout sent")
        except Exception as exc:
            logger.debug("Account logout failed (expected if connection dead): %s", exc)

    @_with_reconnect
    async def execute_position_change(
        self,
        asset: str,
        current_position: float,
        target_position: float,
        portfolio_value: float,
    ) -> OrderResult:
        """Execute a position change on XAUUSD CFD.

        FIX CT-06: On hedging-mode accounts, direction changes (long->short
        or short->long) CLOSE existing positions first via
        ``ProtoOAClosePositionReq``, then open the new position.  This
        prevents accumulating opposing positions.  Same-direction size
        changes still use ``ProtoOANewOrderReq`` for efficiency.

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
            ProtoOAClosePositionReq,
            ProtoOANewOrderReq,
            ProtoOAReconcileReq,
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
        api_volume = self.lots_to_api_volume(delta_lots_rounded)

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

        # ---------------------------------------------------------------
        # FIX CT-06: Close existing positions before opening in the
        # opposite direction.  On hedging accounts, ProtoOANewOrderReq
        # always creates a NEW position — it never closes an existing one.
        # Detect direction change and close first.
        # ---------------------------------------------------------------
        direction_change = (
            current_lots >= self._min_lot and delta_lots_rounded < 0
        ) or (
            current_lots <= -self._min_lot and delta_lots_rounded > 0
        )

        total_fee = 0.0
        close_fill_price = price  # Default for fee calculation

        if direction_change:
            logger.info(
                "CT-06: Direction change detected (current=%.2f lots, "
                "delta=%.2f lots). Closing existing positions first.",
                current_lots,
                delta_lots_rounded,
            )

            # Enumerate actual open positions via reconcile
            try:
                recon_req = ProtoOAReconcileReq()
                recon_req.ctidTraderAccountId = self._account_id
                recon_res = await self._send_request(recon_req, timeout=10.0)
                recon_payload = Protobuf.extract(recon_res)

                open_positions = [
                    p for p in recon_payload.position
                    if p.tradeData.symbolId == self._symbol_id
                ]

                for pos in open_positions:
                    try:
                        close_req = ProtoOAClosePositionReq()
                        close_req.ctidTraderAccountId = self._account_id
                        close_req.positionId = pos.positionId
                        close_req.volume = pos.tradeData.volume

                        close_res = await self._send_request(
                            close_req,
                            timeout=self._market_fallback_timeout,
                        )
                        close_payload = Protobuf.extract(close_res)

                        # Extract fill price from close response
                        if close_payload.HasField("deal"):
                            deal = close_payload.deal
                            if deal.HasField("executionPrice"):
                                close_fill_price = deal.executionPrice

                        lots = self.api_volume_to_lots(pos.tradeData.volume)
                        close_notional = lots * self._lot_size * close_fill_price
                        total_fee += close_notional * self._taker_fee

                        logger.info(
                            "CT-06: Closed position %d (%.2f lots) "
                            "@ %.2f before direction change",
                            pos.positionId,
                            lots,
                            close_fill_price,
                        )

                    except Exception as e:
                        logger.error(
                            "CT-06: Failed to close position %d: %s. "
                            "Aborting direction change to avoid orphans.",
                            pos.positionId,
                            e,
                        )
                        # Abort: don't open the new position if we can't
                        # close the old one — that would create more orphans.
                        return OrderResult(
                            asset=asset,
                            symbol=self._symbol_name,
                            side=side,
                            order_type="market",
                            quantity=abs(delta_lots_rounded),
                            price=price,
                            filled_quantity=0.0,
                            avg_fill_price=0.0,
                            fee=total_fee,
                            status="failed",
                            error=(
                                f"CT-06: Could not close position "
                                f"{pos.positionId} before reversal: {e}"
                            ),
                        )

                # All positions closed — update internal state
                self._position_lots = 0.0

                # Recalculate: we need to open target_lots from flat
                new_volume = self.lots_to_api_volume(target_lots)
                if new_volume < 1:
                    # Target is effectively flat after closing
                    logger.info(
                        "CT-06: Target position rounds to 0 after close. "
                        "Staying flat."
                    )
                    return OrderResult(
                        asset=asset,
                        symbol=self._symbol_name,
                        side=side,
                        order_type="market",
                        quantity=0.0,
                        price=price,
                        filled_quantity=0.0,
                        avg_fill_price=close_fill_price,
                        fee=total_fee,
                        status="filled",
                        error="Closed to flat (target rounds to 0)",
                    )

                # Update side and volume for the new position
                api_side = (
                    ProtoOATradeSide.BUY
                    if target_lots > 0
                    else ProtoOATradeSide.SELL
                )
                side = "buy" if target_lots > 0 else "sell"
                api_volume = new_volume

                logger.info(
                    "CT-06: Opening new %s position: %.2f lots "
                    "(volume=%d) after closing old positions",
                    side.upper(),
                    abs(target_lots),
                    api_volume,
                )

            except Exception as e:
                logger.error(
                    "CT-06: Reconcile failed during direction change: %s. "
                    "Falling back to direct NewOrderReq (may create orphan).",
                    e,
                )
                # Fall through to the normal new-order path as a last resort.
                # This is worse than clean close-then-open, but better than
                # refusing to trade entirely.

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
                if order.HasField("executionPrice"):
                    fill_price = order.executionPrice  # double, already decimal
                if order.executedVolume:
                    filled_volume = order.executedVolume

            if exec_payload.HasField("deal"):
                deal = exec_payload.deal
                if deal.HasField("executionPrice"):
                    fill_price = deal.executionPrice  # double, already decimal
                if deal.filledVolume:
                    filled_volume = deal.filledVolume

            filled_lots = self.api_volume_to_lots(filled_volume)
            filled_notional = filled_lots * self._lot_size * fill_price
            fee = filled_notional * self._taker_fee + total_fee

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

            # ---------------------------------------------------------
            # FIX CT-12: The order protobuf was already sent to cTrader
            # before the Deferred timed out.  The server may have filled
            # it even though we never got the response.  Reconcile to
            # detect ghost fills and sync internal state.
            # ---------------------------------------------------------
            pre_lots = self._position_lots
            try:
                await asyncio.sleep(2)  # Give cTrader time to process
                recon_req = ProtoOAReconcileReq()
                recon_req.ctidTraderAccountId = self._account_id
                recon_res = await self._send_request(recon_req, timeout=10.0)
                recon_payload = Protobuf.extract(recon_res)

                broker_lots = 0.0
                for pos in recon_payload.position:
                    td = pos.tradeData
                    if td.symbolId == self._symbol_id:
                        lots = self.api_volume_to_lots(td.volume)
                        if td.tradeSide == 2:  # SELL
                            lots = -lots
                        broker_lots += lots

                delta = abs(broker_lots - pre_lots)
                if delta >= self._min_lot * 0.5:
                    # Ghost fill detected — order DID execute
                    self._position_lots = broker_lots
                    filled_lots = abs(broker_lots - pre_lots)
                    filled_notional = filled_lots * self._lot_size * price
                    fee = filled_notional * self._taker_fee + total_fee
                    logger.warning(
                        "CT-12: Post-timeout reconcile detected ghost fill. "
                        "Broker=%.2f lots, expected=%.2f lots. "
                        "Treating as filled.",
                        broker_lots,
                        pre_lots,
                    )
                    return OrderResult(
                        asset=asset,
                        symbol=self._symbol_name,
                        side=side,
                        order_type="market",
                        quantity=abs(delta_lots_rounded),
                        price=price,
                        filled_quantity=filled_lots,
                        avg_fill_price=price,
                        fee=fee,
                        status="filled",
                        error="CT-12: ghost fill recovered via reconcile",
                    )
            except Exception as recon_exc:
                logger.warning(
                    "CT-12: Post-timeout reconcile also failed: %s. "
                    "Returning failed — position may be desynced.",
                    recon_exc,
                )

            return OrderResult(
                asset=asset,
                symbol=self._symbol_name,
                side=side,
                order_type="market",
                quantity=abs(delta_lots_rounded),
                price=price,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                fee=total_fee,
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
                    lots = self.api_volume_to_lots(trade_data.volume)
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
            # Return last known position — returning 0.0 (fake flat) would
            # cause the engine to open new positions, creating double exposure.
            price = self._mid_price if self._mid_price > 0 else 1.0
            return self._lots_to_position(
                self._position_lots, self._portfolio_value, price
            )

    @_with_reconnect
    async def get_account_info(self) -> dict:
        """Fetch account equity, available balance, and used margin."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
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

            # Get open positions (used margin + unrealized PnL)
            recon_req = ProtoOAReconcileReq()
            recon_req.ctidTraderAccountId = self._account_id
            recon_res = await self._send_request(recon_req, timeout=10.0)

            recon_payload = Protobuf.extract(recon_res)
            used_margin = 0.0
            unrealized_pnl = 0.0
            n_positions = 0
            pnl = 0.0
            current_price = self._mid_price

            for pos in recon_payload.position:
                if pos.usedMargin:
                    used_margin += pos.usedMargin / (10 ** self._money_digits)

                # Calculate unrealized PnL locally from position entry price
                # and current mid_price.  The old ProtoOAGetPositionUnrealizedPnLReq
                # call silently failed on many brokers, leaving PnL at 0.
                # pos.price is a protobuf double — already decimal.
                trade_data = pos.tradeData
                if trade_data.symbolId == self._symbol_id and current_price > 0:
                    lots = self.api_volume_to_lots(trade_data.volume)
                    entry_price = pos.price  # double, already decimal
                    # BUY=1, SELL=2
                    direction = 1.0 if trade_data.tradeSide == 1 else -1.0
                    if entry_price and entry_price > 0:
                        pnl = direction * lots * self._lot_size * (
                            current_price - entry_price
                        )
                        unrealized_pnl += pnl
                    n_positions += 1

            if recon_payload.position and current_price > 0:
                logger.info(
                    f"Account: balance=${balance:,.2f}, "
                    f"unrealized_pnl=${unrealized_pnl:,.2f}, "
                    f"n_positions={n_positions}, mid=${current_price:,.2f}"
                )

            total_equity = balance + unrealized_pnl
            available = total_equity - used_margin
            self._portfolio_value = total_equity

            # S510 forensics: expose components so the engine sanity guard can
            # log a breakdown when a phantom equity is rejected. Disambiguates
            # protobuf mis-pairing (balance spike) from stale-mid_price orphan
            # PnL (unrealized_pnl spike).
            return {
                "total_equity": total_equity,
                "available_balance": available,
                "used_margin": used_margin,
                "balance": balance,
                "unrealized_pnl": unrealized_pnl,
                "n_positions": n_positions,
                "mid_price": current_price,
            }

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {
                "total_equity": self._portfolio_value,
                "available_balance": self._portfolio_value,
                "used_margin": 0.0,
                "balance": 0.0,
                "unrealized_pnl": 0.0,
                "n_positions": 0,
                "mid_price": self._mid_price,
            }

    @_with_reconnect
    async def check_orphaned_positions(self) -> list[dict]:
        """FIX CT-05: Enumerate ALL individual positions for this symbol.

        On hedging-mode accounts, ``get_single_position()`` returns the NET
        of all positions — which can be 0.0 even with multiple opposing
        positions open.  This method returns each position individually so
        the caller can detect and warn about orphaned positions.

        Returns:
            List of dicts with keys: position_id, side, lots, entry_price.
            Empty list if no positions.
        """
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAReconcileReq,
        )

        try:
            recon_req = ProtoOAReconcileReq()
            recon_req.ctidTraderAccountId = self._account_id
            recon_res = await self._send_request(recon_req, timeout=10.0)

            recon_payload = Protobuf.extract(recon_res)
            positions = []

            for pos in recon_payload.position:
                trade_data = pos.tradeData
                if trade_data.symbolId == self._symbol_id:
                    lots = self.api_volume_to_lots(trade_data.volume)
                    side = "BUY" if trade_data.tradeSide == 1 else "SELL"
                    entry_price = pos.price if pos.HasField("price") else 0.0
                    positions.append({
                        "position_id": pos.positionId,
                        "side": side,
                        "lots": lots,
                        "entry_price": entry_price,
                    })

            if len(positions) > 1:
                logger.critical(
                    "CT-05 ORPHANED POSITIONS DETECTED: %d individual positions "
                    "open for %s on hedging-mode account. Net position may "
                    "appear flat while real exposure exists. Positions: %s",
                    len(positions),
                    self._symbol_name,
                    positions,
                )
            elif len(positions) == 1:
                p = positions[0]
                logger.info(
                    "Single position confirmed: %s %.2f lots @ %.2f "
                    "(positionId=%d)",
                    p["side"],
                    p["lots"],
                    p["entry_price"],
                    p["position_id"],
                )
            else:
                logger.info("No open positions for %s", self._symbol_name)

            return positions

        except Exception as e:
            logger.error("Failed to enumerate positions: %s", e)
            return []

    @_with_reconnect
    async def close_position_by_id(self, position_id: int, volume: int) -> bool:
        """Close a specific position by ID using ProtoOAClosePositionReq.

        Used by CT-06 (close-before-reverse) and CT-05 (orphan cleanup).

        Args:
            position_id: The cTrader position ID.
            volume: Volume to close in API units (hundredths of a lot).

        Returns:
            True if closed successfully, False on failure.
        """
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAClosePositionReq,
        )

        try:
            close_req = ProtoOAClosePositionReq()
            close_req.ctidTraderAccountId = self._account_id
            close_req.positionId = position_id
            close_req.volume = volume

            await self._send_request(
                close_req, timeout=self._market_fallback_timeout
            )

            lots = self.api_volume_to_lots(volume)
            logger.info(
                "Closed position %d (%.2f lots)", position_id, lots
            )
            return True

        except Exception as e:
            logger.error("Failed to close position %d: %s", position_id, e)
            return False

    async def get_funding_rates(self, assets: list[str]) -> dict[str, float]:
        """CFDs have no funding rates. Return zeros."""
        return {a: 0.0 for a in assets}

    @_with_reconnect
    async def get_deal_list(self, from_days_ago: int = 7) -> list[dict]:
        """Fetch historical deals (fills) for this account.

        Read-only account query used by
        ``scripts/analyze_xauusd_fill_costs.py`` to measure real per-side
        commission + spread so HPO configs can plug a measured steady-state
        ``taker_fee`` instead of a curriculum that never ramps (S461/S463).

        Commission is converted to deposit-currency units
        (``commission_raw / 10^money_digits``) and returned as a positive
        number — the raw payload reports it as a signed int where negative
        means "charged to trader".

        Args:
            from_days_ago: Lookback window in days.

        Returns:
            List of deal dicts filtered to this broker's symbol. Keys:
            ``dealId``, ``orderId``, ``positionId``, ``executionTimestamp``
            (ms), ``tradeSide`` ("BUY"/"SELL"), ``lots`` (unsigned float),
            ``executionPrice`` (float), ``commission`` (float, deposit
            currency, positive = charged).
        """
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOADealListReq,
        )

        now_ms = int(time.time() * 1000)
        from_ms = now_ms - from_days_ago * 86_400_000

        req = ProtoOADealListReq()
        req.ctidTraderAccountId = self._account_id
        req.fromTimestamp = from_ms
        req.toTimestamp = now_ms

        res = await self._send_request(req, timeout=30.0)
        payload = Protobuf.extract(res)
        self._check_error(payload, "get_deal_list")

        deals: list[dict] = []

        for deal in payload.deal:
            if self._symbol_id is not None and deal.symbolId != self._symbol_id:
                continue

            lots = self.api_volume_to_lots(deal.filledVolume)
            side = "BUY" if deal.tradeSide == 1 else "SELL"
            exec_price = float(getattr(deal, "executionPrice", 0.0) or 0.0)
            commission_raw = int(getattr(deal, "commission", 0) or 0)
            # Prefer per-deal moneyDigits (authoritative per fill) over the
            # account-level scaling stored on the broker.
            money_digits = (
                deal.moneyDigits
                if deal.HasField("moneyDigits")
                else self._money_digits
            )
            commission = abs(commission_raw) / (10 ** money_digits)

            deals.append({
                "dealId": deal.dealId,
                "orderId": deal.orderId,
                "positionId": deal.positionId,
                "executionTimestamp": deal.executionTimestamp,
                "tradeSide": side,
                "lots": lots,
                "executionPrice": exec_price,
                "commission": commission,
            })

        logger.info(
            "get_deal_list: %d deals over %d days (symbol=%s)",
            len(deals), from_days_ago, self._symbol_name,
        )
        return deals

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
                lots_closed_this_attempt = 0.0
                for pos in open_positions:
                    try:
                        close_req = ProtoOAClosePositionReq()
                        close_req.ctidTraderAccountId = self._account_id
                        close_req.positionId = pos.positionId
                        close_req.volume = pos.tradeData.volume

                        await self._send_request(
                            close_req, timeout=self._market_fallback_timeout
                        )

                        lots = self.api_volume_to_lots(pos.tradeData.volume)
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
                        # FIX CT-01: Track lots closed for partial-success position update
                        lots_closed_this_attempt += lots
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
                            quantity=self.api_volume_to_lots(pos.tradeData.volume),
                            price=0.0,
                            filled_quantity=0.0,
                            avg_fill_price=0.0,
                            fee=0.0,
                            status="failed",
                            error=str(e),
                        ))

                # FIX CT-01: Update _position_lots even on partial success.
                # Previously only zeroed on full success, leaving stale state
                # that caused wrong trade sizing after partial flatten.
                if lots_closed_this_attempt > 0:
                    sign = 1.0 if self._position_lots >= 0 else -1.0
                    self._position_lots = sign * max(
                        0.0, abs(self._position_lots) - lots_closed_this_attempt
                    )

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
        """Convert position fraction [-1, 1] to signed lots, capped by
        the configured leverage.

        Example: $100K portfolio, Gold @ $3000/oz, 100 oz/lot, leverage 30x
            fraction=1.0 → notional=$100K → lots = 100000/(3000*100) = 0.333

        The agent's ``fraction`` targets notional = |fraction| × PV
        (unleveraged fraction of equity).  Defensive guard: if the broker's
        minimum lot × lot_size × price ever exceeds the configured leverage
        cap, skip rather than floor up (so a mis-parameterized symbol or
        small account can't silently over-leverage via the floor-up path).
        With correctly-parsed min_lot (0.01 for XAUUSD on IC Markets), this
        is inert on a $10k+ account — it's a belt-and-braces check.
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0

        # Leverage cap: max notional the config allows
        leverage = max(float(self._leverage), 1.0)
        max_notional = portfolio_value * leverage
        min_lot_notional = self._min_lot * self._lot_size * price

        # Account too small to hold even one min_lot under the leverage cap —
        # can't express any position without over-leveraging.  Skip.
        if min_lot_notional > max_notional:
            if abs(fraction) >= 0.1:
                logger.warning(
                    "Sizing skip: min_lot %.4f @ $%.2f (notional $%.0f) "
                    "exceeds leverage cap ($%.0f = PV $%.0f × %.1fx). "
                    "Agent fraction=%.3f. Request micro-lot (0.01) account "
                    "or increase paper balance.",
                    self._min_lot, price, min_lot_notional,
                    max_notional, portfolio_value, leverage, fraction,
                )
            return 0.0

        notional = abs(fraction) * portfolio_value
        lots = notional / (price * self._lot_size)
        lots = round(lots / self._min_lot) * self._min_lot

        # Cap at max allowed under leverage (round down so we stay within cap)
        max_lots = int((max_notional / (price * self._lot_size)) / self._min_lot) * self._min_lot
        if lots > max_lots:
            lots = max_lots

        # Floor up to min_lot when agent has conviction.  Safe now — the
        # skip-guard above already verified min_lot_notional ≤ max_notional.
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

    def lots_to_api_volume(self, lots: float) -> int:
        """Convert lots to cTrader API `volume` (centi-units of base).

        cTrader Open API: `volume = lots × lotSize_raw`.  For XAUUSD where
        `lotSize_raw = 10000` (100 oz/lot × 100 centi/oz), 1.00 lot sends
        volume=10000, 0.01 lot sends volume=100.  Signed input accepted;
        magnitude only is returned (the API takes a separate `tradeSide`).
        """
        if self._lot_size_raw <= 0:
            return 0
        return int(round(abs(lots) * self._lot_size_raw))

    def api_volume_to_lots(self, volume: int) -> float:
        """Convert cTrader API `volume` (centi-units) back to lots."""
        if self._lot_size_raw <= 0:
            return 0.0
        return volume / self._lot_size_raw

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

        IMPORTANT: ``client.send()`` and ``Deferred.addCallback/addErrback``
        must run on the Twisted reactor thread — Deferreds are NOT thread-safe.
        We use ``reactor.callFromThread()`` to schedule the send + callback
        attachment atomically on the reactor thread.

        Raises RuntimeError if the connection is down, which is caught
        by the ``@_with_reconnect`` decorator on public API methods.
        """
        if self._client is None or not self._connected:
            raise RuntimeError("Not connected to cTrader")

        from twisted.internet import reactor

        msg_id = self._next_msg_id()

        result_container = {"value": None, "error": None, "done": False}

        def _do_send():
            """Runs on the Twisted reactor thread."""
            try:
                d = self._client.send(
                    message,
                    clientMsgId=msg_id,
                    responseTimeoutInSeconds=timeout,
                )

                def on_success(result):
                    result_container["value"] = result
                    result_container["done"] = True
                    return result

                def on_error(failure):
                    result_container["error"] = failure
                    result_container["done"] = True

                d.addCallback(on_success)
                d.addErrback(on_error)
            except Exception as exc:
                result_container["error"] = exc
                result_container["done"] = True

        reactor.callFromThread(_do_send)

        return await self._poll_result(result_container, timeout=timeout + 2)

    async def _deferred_to_future(self, deferred, timeout: float = 15.0):
        """Convert a Twisted Deferred to an asyncio-awaitable result.

        Attaches callbacks on the reactor thread for thread safety,
        then polls from the asyncio side.
        """
        from twisted.internet import reactor

        result_container = {"value": None, "error": None, "done": False}

        def _attach_callbacks():
            """Runs on the Twisted reactor thread."""
            def on_success(result):
                result_container["value"] = result
                result_container["done"] = True
                return result

            def on_error(failure):
                result_container["error"] = failure
                result_container["done"] = True

            deferred.addCallback(on_success)
            deferred.addErrback(on_error)

        reactor.callFromThread(_attach_callbacks)

        return await self._poll_result(result_container, timeout=timeout)

    async def _poll_result(self, result_container: dict, timeout: float = 15.0):
        """Poll a result container until resolved or timeout.

        Uses asyncio.wait_for as a hard safety net in case the manual
        deadline check fails to fire (e.g., callFromThread stall).
        """
        async def _poll():
            deadline = time.monotonic() + timeout
            while not result_container["done"]:
                if time.monotonic() > deadline:
                    raise asyncio.TimeoutError(
                        f"cTrader request timed out after {timeout}s"
                    )
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(_poll(), timeout=timeout + 10)
        except asyncio.TimeoutError:
            if not result_container["done"]:
                raise asyncio.TimeoutError(
                    f"cTrader request hard-timed-out after {timeout + 10}s "
                    f"(callFromThread may have stalled)"
                )

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
                        self._bid = spot.bid / _SPOT_PRICE_SCALE
                    if spot.HasField("ask"):
                        self._ask = spot.ask / _SPOT_PRICE_SCALE
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
                err_code = getattr(err, "errorCode", "")
                err_desc = getattr(err, "description", "")
                logger.error(
                    f"cTrader error: {err_code} — {err_desc}"
                    if err_desc else
                    f"cTrader error: {err_code}"
                )
                # FIX CT-07: Set flag so _reconnect() / connect() callers
                # can detect ghost-session without relying on exception
                # string matching (the Deferred times out with a generic
                # TimeoutError, not a RuntimeError containing the error code).
                if "ALREADY_LOGGED_IN" in str(err_code):
                    self._already_logged_in_seen = True
                    logger.warning(
                        "CT-07: ALREADY_LOGGED_IN detected in _on_message — "
                        "ghost session flag set"
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
                # FIX CFD-B-01: Run blocking HTTP call in thread executor
                # to avoid blocking the asyncio event loop for up to 30s.
                loop = asyncio.get_running_loop()
                token_data = await loop.run_in_executor(
                    None,
                    lambda: Auth.refreshToken(
                        refreshToken=self._refresh_token,
                        clientId=self._client_id,
                        clientSecret=self._client_secret,
                    ),
                )

                if "accessToken" in token_data:
                    self._access_token = token_data["accessToken"]
                    if "refreshToken" in token_data:
                        self._refresh_token = token_data["refreshToken"]
                    self._persist_tokens_to_state_file()
                    logger.info("Access token refreshed successfully")
                else:
                    logger.error(f"Token refresh failed: {token_data}")
                    for delay in retry_delays:
                        await asyncio.sleep(delay)
                        # FIX CFD-B-01: Retry also in executor
                        token_data = await loop.run_in_executor(
                            None,
                            lambda: Auth.refreshToken(
                                refreshToken=self._refresh_token,
                                clientId=self._client_id,
                                clientSecret=self._client_secret,
                            ),
                        )
                        if "accessToken" in token_data:
                            self._access_token = token_data["accessToken"]
                            if "refreshToken" in token_data:
                                self._refresh_token = token_data["refreshToken"]
                            self._persist_tokens_to_state_file()
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
