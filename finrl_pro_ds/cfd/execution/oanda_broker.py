"""OANDA v20 broker for EUR/USD CFD paper/live trading.

Drop-in replacement for ``CTraderBroker`` from the live engine's
perspective. Implements the same duck-typed interface as
``IBFuturesBroker`` / ``ExchangePerpBroker`` / ``CTraderBroker``:

    - connect(), close()
    - execute_position_change(asset, current_pos, target_pos, pv) -> OrderResult
    - get_single_position(asset) -> float
    - get_account_info() -> dict
    - emergency_flatten(assets) -> RebalanceResult
    - get_funding_rates(assets) -> dict
    - check_orphaned_positions() -> list[dict]
    - close_position_by_id(position_id, volume) -> bool   (NETTING stub)
    - get_deal_list(from_days_ago) -> list[dict]          (audit)

Attributes accessed by LiveTradingEngine:
    - exchange_id: str = "oanda"
    - testnet: bool   (True when env == "practice")
    - client: None    (compat stub; OandaDataLoader uses broker._session)

Uses raw ``httpx.AsyncClient`` over REST + streaming HTTP. Bearer-token
auth, no OAuth refresh, no token-state file. Practice and live use
different hosts and different tokens.

Env vars:
    OANDA_API_TOKEN     — personal access token (Bearer auth)
    OANDA_ACCOUNT_ID    — trading account ID (101-001-XXXXXXX-NNN)
    OANDA_ENV           — "practice" (default) or "live"

Per-strategy overrides honored by the compose entry:
    <STRATEGY>_OANDA_ACCOUNT_ID — overrides shared OANDA_ACCOUNT_ID

See `.agent/artifacts/oanda_broker_architecture.md` for the full design.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from typing import Any, Optional

import httpx
import numpy as np

from finrl_pro_ds.crypto.execution.exchange_perp_broker import (
    OrderResult,
    RebalanceResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Host + symbol maps
# ---------------------------------------------------------------------------

_REST_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}
_STREAM_HOSTS = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live": "https://stream-fxtrade.oanda.com",
}

# Map our config-side symbols to OANDA API symbols. Keeping config tokens
# stable (EURUSD, XAUUSD) avoids churn to bundle metadata, drift baselines,
# WandB tags, etc. — broker owns the translation boundary.
_SYMBOL_MAP = {
    "EURUSD": "EUR_USD",
    "XAUUSD": "XAU_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
}

# Streaming heartbeat ages
_HEARTBEAT_WARN_SEC = 30.0   # force-reconnect at this age
_HEARTBEAT_DEAD_SEC = 120.0  # mark broker disconnected at this age

# Per-bar position-fetch cache. OANDA's NETTING positions endpoint returns
# in <100ms but we still avoid hammering it; engine reconcile runs every 4
# bars anyway.
_POSITION_CACHE_TTL_SEC = 2.0


def _with_reconnect(method):
    """Decorator: check stream liveness, reconnect on stale heartbeat."""

    @functools.wraps(method)
    async def wrapper(self: "OandaBroker", *args, **kwargs):
        if self._connected and self._session is not None:
            try:
                return await method(self, *args, **kwargs)
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                asyncio.TimeoutError,
            ) as exc:
                logger.warning(
                    "OANDA API call %s failed (%s), attempting reconnect",
                    method.__name__,
                    exc,
                )
        reconnected = await self._reconnect()
        if not reconnected:
            raise RuntimeError(
                f"OANDA reconnect failed — cannot execute {method.__name__}"
            )
        return await method(self, *args, **kwargs)

    return wrapper


class OandaBroker:
    """OANDA v20 CFD broker for forex + metals paper and live trading."""

    _RECONNECT_COOLDOWN_SECS = 60

    def __init__(
        self,
        env: str = "practice",
        symbol: str = "EURUSD",
        lot_size: float = 100_000.0,
        min_lot: float = 0.01,
        tick_size: float = 0.00001,
        leverage: int = 1,
        order_type: str = "MKT",
        taker_fee: float = 0.00005,
        market_fallback_timeout: float = 10.0,
        max_reconnect_attempts: int = 3,
    ) -> None:
        """
        Args:
            env: "practice" (paper) or "live" (real money).
            symbol: Config-side symbol (e.g. "EURUSD"). Internally mapped
                to OANDA convention via _SYMBOL_MAP.
            lot_size: Base-currency units per standard lot (FX = 100_000).
            min_lot: Minimum lot increment (0.01 = 1000 units micro-lot).
            tick_size: Price increment (0.00001 for 5-digit pairs).
            leverage: Account leverage cap (1 = 1:1; OANDA EUR/USD
                supports up to 50:1 on practice).
            order_type: "MKT" only — OANDA MARKET orders with FOK TIF.
            taker_fee: Effective spread as fraction. EUR/USD practice =
                ~0.4 bps = 0.00005 (matches HPO trial #18 env.taker_fee).
            market_fallback_timeout: Seconds to wait for order fill.
            max_reconnect_attempts: Cap on reconnect retries before failing.
        """
        env = env.lower()
        if env not in _REST_HOSTS:
            raise ValueError(f"OANDA env must be 'practice' or 'live', got {env!r}")

        self._env = env
        self._symbol_name = symbol
        self._oanda_symbol = _SYMBOL_MAP.get(symbol)
        if self._oanda_symbol is None:
            raise ValueError(
                f"OANDA symbol map has no entry for {symbol!r}. "
                f"Add it to _SYMBOL_MAP in oanda_broker.py."
            )

        self._lot_size = float(lot_size)
        self._min_lot = float(min_lot)
        self._tick_size = float(tick_size)
        self._leverage = int(leverage)
        self._order_type = order_type
        self._taker_fee = float(taker_fee)
        self._market_fallback_timeout = float(market_fallback_timeout)
        self._max_reconnect_attempts = int(max_reconnect_attempts)

        # LiveTradingEngine contract attributes
        self.exchange_id = "oanda"
        # `testnet` semantics across the codebase: True = paper, False = live.
        # OANDA practice == paper, live == live. Map accordingly.
        self.testnet = (env == "practice")

        # Credentials (resolved from env at __init__; one OANDA token covers
        # all sub-accounts under the user identity; per-strategy isolation
        # is via OANDA_ACCOUNT_ID).
        self._token = os.environ.get("OANDA_API_TOKEN", "")
        self._account_id = os.environ.get("OANDA_ACCOUNT_ID", "")

        # Hosts
        self._rest_host = _REST_HOSTS[env]
        self._stream_host = _STREAM_HOSTS[env]

        # Connection state
        self._session: Optional[httpx.AsyncClient] = None
        self._connected: bool = False
        self._reconnect_lock = asyncio.Lock()
        self._consecutive_reconnects: int = 0
        self._reconnect_cooldown_until: float = 0.0

        # Market state (updated by streaming pricing task)
        self._bid: float = 0.0
        self._ask: float = 0.0
        self._mid_price: float = 0.0
        self._last_heartbeat: float = 0.0
        self._stream_task: Optional[asyncio.Task] = None

        # Account / position state (broker truth)
        self._position_units: int = 0  # signed integer units
        self._portfolio_value: float = 0.0
        self._position_cache_at: float = 0.0  # epoch of last position fetch

        # Instrument metadata (populated on connect)
        self._pip_location: int = -4
        self._display_precision: int = 5
        self._trade_units_precision: int = 0
        self._minimum_trade_size: int = 1
        self._margin_rate: float = 0.02  # 50:1 default; refreshed on connect

    # ------------------------------------------------------------------
    # Engine-contract compat
    # ------------------------------------------------------------------

    @property
    def client(self) -> Optional[httpx.AsyncClient]:
        """Active httpx session (None if disconnected).

        Returned for parity with CTraderBroker's ``client`` property; the
        OANDA data loader actually uses ``broker._session`` directly and
        doesn't need a separate handle.
        """
        return self._session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate, validate instrument, prime price stream."""
        if self._connected:
            return
        if not self._token:
            raise RuntimeError(
                "OANDA_API_TOKEN env var is empty. Set it in docker/live/.env "
                "and force-recreate the container."
            )
        if not self._account_id:
            raise RuntimeError(
                "OANDA_ACCOUNT_ID env var is empty. Set per-strategy "
                "OANDA_ACCOUNT_ID in compose env (e.g. "
                "SG1_EURUSD_OANDA_ACCOUNT_ID)."
            )

        logger.info(
            "Connecting to OANDA %s (env=%s, account=%s, symbol=%s -> %s)",
            self._rest_host, self._env, self._account_id,
            self._symbol_name, self._oanda_symbol,
        )

        # One client for REST + streaming. HTTP/2 reduces stream-reconnect
        # overhead but is optional; falling back if h2 isn't installed.
        try:
            self._session = httpx.AsyncClient(
                base_url=self._rest_host,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept-Datetime-Format": "RFC3339",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=10.0),
                http2=True,
            )
        except ImportError:
            self._session = httpx.AsyncClient(
                base_url=self._rest_host,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept-Datetime-Format": "RFC3339",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=10.0),
            )

        # Step 1: validate token + fetch NAV
        summary = await self._get_account_summary()
        self._portfolio_value = float(summary["NAV"])
        self._margin_rate = float(summary.get("marginRate", "0.02"))

        # Defensive guard: config leverage must fit under the broker's
        # marginRate cap. marginRate 0.02 = 50:1 ceiling; config leverage=1
        # is always safe. Guard catches mis-configured live deploys.
        max_leverage = int(round(1.0 / max(self._margin_rate, 0.001)))
        if self._leverage > max_leverage:
            raise RuntimeError(
                f"MARGIN-CFG violation: config leverage={self._leverage} "
                f"exceeds OANDA cap {max_leverage} (marginRate={self._margin_rate})"
            )

        logger.info(
            "OANDA account: NAV=$%.2f balance=$%s leverage_cap=%d:1 "
            "(config leverage=%d)",
            self._portfolio_value, summary.get("balance"), max_leverage,
            self._leverage,
        )

        # Step 2: validate instrument tradeable + cache metadata
        await self._fetch_instrument_metadata()

        # Step 3: prime current position
        await self._refresh_position_cache()

        # Step 4: launch pricing stream task
        self._stream_task = asyncio.create_task(
            self._consume_pricing_stream(),
            name=f"oanda-stream-{self._oanda_symbol}",
        )

        # Step 5: wait up to 10s for the first PRICE message — engine
        # refuses to trade without a mid_price.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._mid_price > 0:
                break
            await asyncio.sleep(0.2)
        if self._mid_price <= 0:
            logger.warning(
                "OANDA stream not yet producing prices after 10s — "
                "continuing; engine bar loop will retry."
            )

        self._connected = True
        self._consecutive_reconnects = 0
        logger.info(
            "OANDA connected: bid=%.5f ask=%.5f mid=%.5f position_units=%d",
            self._bid, self._ask, self._mid_price, self._position_units,
        )

    async def close(self) -> None:
        """Cancel stream task, close httpx session. Idempotent."""
        self._connected = False
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await asyncio.wait_for(self._stream_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._stream_task = None
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("OANDA session aclose error (non-fatal): %s", exc)
            self._session = None
        logger.info("OANDA broker closed")

    def _check_connection(self) -> bool:
        """True if session is healthy and stream heartbeat is recent."""
        if not self._connected or self._session is None:
            return False
        if self._last_heartbeat <= 0:
            return False
        age = time.time() - self._last_heartbeat
        return age < _HEARTBEAT_DEAD_SEC

    async def _reconnect(self) -> bool:
        """Re-establish session + stream. Cooldown-rate-limited."""
        async with self._reconnect_lock:
            now = time.time()
            if now < self._reconnect_cooldown_until:
                wait = self._reconnect_cooldown_until - now
                logger.info("OANDA reconnect on cooldown, waiting %.1fs", wait)
                await asyncio.sleep(wait)

            for attempt in range(1, self._max_reconnect_attempts + 1):
                try:
                    await self.close()
                    await self.connect()
                    return True
                except Exception as exc:  # noqa: BLE001
                    delay = 5.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "OANDA reconnect attempt %d/%d failed: %s (retry in %.0fs)",
                        attempt, self._max_reconnect_attempts, exc, delay,
                    )
                    if attempt < self._max_reconnect_attempts:
                        await asyncio.sleep(delay)

            self._reconnect_cooldown_until = time.time() + self._RECONNECT_COOLDOWN_SECS
            self._consecutive_reconnects += 1
            return False

    # ------------------------------------------------------------------
    # Pricing stream
    # ------------------------------------------------------------------

    async def _consume_pricing_stream(self) -> None:
        """Long-running task: consume OANDA pricing stream NDJSON.

        OANDA emits two message types:
          - PRICE: bid/ask tick. Update self._bid/_ask/_mid_price.
          - HEARTBEAT: keepalive every ~5s. Update self._last_heartbeat.

        Stream restart on any error after 1s backoff (engine bar loop is
        the ultimate arbiter — if mid_price stays stale across a bar,
        engine pauses / halts).
        """
        url = (
            f"{self._stream_host}/v3/accounts/{self._account_id}"
            f"/pricing/stream?instruments={self._oanda_symbol}"
        )
        headers = {"Authorization": f"Bearer {self._token}"}

        while True:
            try:
                async with httpx.AsyncClient(timeout=None) as stream_client:
                    async with stream_client.stream(
                        "GET", url, headers=headers,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                            except json.JSONDecodeError:
                                logger.debug("OANDA stream junk line: %r", line[:100])
                                continue
                            msg_type = msg.get("type")
                            if msg_type == "PRICE":
                                self._on_price_message(msg)
                            elif msg_type == "HEARTBEAT":
                                self._last_heartbeat = time.time()
                            else:
                                logger.debug(
                                    "OANDA stream unknown msg type: %s", msg_type,
                                )
            except asyncio.CancelledError:
                logger.info("OANDA stream task cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OANDA stream error: %s (restarting in 1s)", exc,
                )
                await asyncio.sleep(1.0)

    def _on_price_message(self, msg: dict) -> None:
        """Update bid/ask/mid from a PRICE message."""
        try:
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            if not bids or not asks:
                return
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            self._bid = bid
            self._ask = ask
            self._mid_price = (bid + ask) / 2.0
            self._last_heartbeat = time.time()
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("OANDA PRICE parse error: %s msg=%r", exc, msg)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    @_with_reconnect
    async def execute_position_change(
        self,
        asset: str,
        current_position: float,
        target_position: float,
        portfolio_value: float,
    ) -> OrderResult:
        """Execute a single-asset position change via MARKET order.

        OANDA NETTING semantics: a single signed-units order handles
        flat->long, long->short, short->flat, and partial-size changes.
        No close-then-open dance is needed (vs cTrader's CT-06 path).

        Args:
            asset: Config-side asset symbol (used for logging + OrderResult).
            current_position: Engine's view of position fraction [-1, 1].
            target_position: Agent's desired position fraction [-1, 1].
            portfolio_value: Engine's NAV snapshot at decision time.

        Returns:
            OrderResult — broker NEVER raises on rejection; returns
            ``status="failed"`` so the engine can decide what to do.
        """
        self._portfolio_value = portfolio_value
        price = self._mid_price

        if price <= 0:
            return OrderResult(
                asset=asset, symbol=self._symbol_name, side="none",
                order_type="failed", quantity=0.0, price=0.0,
                filled_quantity=0.0, avg_fill_price=0.0, fee=0.0,
                status="failed",
                error="No price available — pricing stream may not be active",
            )

        target_lots = self._position_to_lots(target_position, portfolio_value, price)
        current_lots = self._position_units / self._lot_size
        delta_lots = target_lots - current_lots

        # Dust skip
        if abs(delta_lots) < self._min_lot:
            return OrderResult(
                asset=asset, symbol=self._symbol_name, side="none",
                order_type="skipped", quantity=abs(delta_lots), price=price,
                filled_quantity=0.0, avg_fill_price=0.0, fee=0.0,
                status="skipped",
                error=f"Delta {delta_lots:.4f} lots below min {self._min_lot}",
            )

        # Round to min_lot precision; recover sign if abs() rounded to zero.
        delta_lots_rounded = round(delta_lots / self._min_lot) * self._min_lot
        if abs(delta_lots_rounded) < self._min_lot:
            delta_lots_rounded = self._min_lot * (1 if delta_lots > 0 else -1)

        side = "buy" if delta_lots_rounded > 0 else "sell"
        delta_units = self.lots_to_units_signed(delta_lots_rounded)

        if abs(delta_units) < self._minimum_trade_size:
            return OrderResult(
                asset=asset, symbol=self._symbol_name, side=side,
                order_type="skipped", quantity=abs(delta_lots_rounded),
                price=price, filled_quantity=0.0, avg_fill_price=0.0,
                fee=0.0, status="skipped",
                error=(
                    f"Units {delta_units} below OANDA minimum "
                    f"{self._minimum_trade_size}"
                ),
            )

        logger.info(
            "Placing %s %.4f lots %s @ ~%.5f (units=%+d)",
            side.upper(), abs(delta_lots_rounded), self._symbol_name,
            price, delta_units,
        )

        # Post the order. FOK = Fill-Or-Kill: rejects partials. OANDA
        # practice depth is generous (>2M units on top of book for major
        # pairs), so FOK is safe for retail sizes. Live deploy may want
        # IOC fallback — deferred.
        body = {
            "order": {
                "type": "MARKET",
                "instrument": self._oanda_symbol,
                "units": str(delta_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }

        try:
            resp = await self._session.post(
                f"/v3/accounts/{self._account_id}/orders",
                json=body,
                timeout=self._market_fallback_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("OANDA order POST failed: %s", exc)
            return OrderResult(
                asset=asset, symbol=self._symbol_name, side=side,
                order_type="market", quantity=abs(delta_lots_rounded),
                price=price, filled_quantity=0.0, avg_fill_price=0.0,
                fee=0.0, status="failed", error=str(exc),
            )

        if resp.status_code >= 400:
            err_body = self._safe_json(resp)
            err_msg = (
                err_body.get("errorMessage")
                or err_body.get("errorCode")
                or f"HTTP {resp.status_code}"
            )
            logger.error(
                "OANDA order rejected (HTTP %d): %s body=%s",
                resp.status_code, err_msg, err_body,
            )
            return OrderResult(
                asset=asset, symbol=self._symbol_name, side=side,
                order_type="market", quantity=abs(delta_lots_rounded),
                price=price, filled_quantity=0.0, avg_fill_price=0.0,
                fee=0.0, status="failed",
                error=f"OANDA rejected: {err_msg}",
            )

        payload = self._safe_json(resp)
        fill = payload.get("orderFillTransaction") or {}
        cancel = payload.get("orderCancelTransaction") or {}

        if not fill and cancel:
            reason = cancel.get("reason", "UNKNOWN_REASON")
            logger.error("OANDA order cancelled: %s", reason)
            return OrderResult(
                asset=asset, symbol=self._symbol_name, side=side,
                order_type="market", quantity=abs(delta_lots_rounded),
                price=price, filled_quantity=0.0, avg_fill_price=0.0,
                fee=0.0, status="failed",
                error=f"OANDA cancelled: {reason}",
            )

        # Fill present — parse it
        fill_price = float(fill.get("price", price))
        filled_units_str = fill.get("units", "0")
        try:
            filled_units = int(filled_units_str)
        except (ValueError, TypeError):
            filled_units = delta_units  # fallback: assume full fill
        commission = float(fill.get("commission", "0"))
        order_id = str(fill.get("orderID", ""))

        filled_lots = filled_units / self._lot_size  # signed
        filled_notional = abs(filled_lots) * self._lot_size * fill_price
        # OANDA practice EUR/USD commission is 0; cost is in the spread.
        # Model spread-cost as taker_fee × notional, plus any commission
        # OANDA reports (defensive — non-zero on some instruments/configs).
        fee = abs(commission) + filled_notional * self._taker_fee

        # Update broker-truth position
        self._position_units += filled_units
        self._position_cache_at = time.time()

        logger.info(
            "Filled %s %.4f lots @ %.5f, fee=$%.4f, position now %d units "
            "(%.4f lots)",
            side.upper(), abs(filled_lots), fill_price, fee,
            self._position_units, self._position_units / self._lot_size,
        )

        return OrderResult(
            asset=asset, symbol=self._symbol_name, side=side,
            order_type="market", quantity=abs(delta_lots_rounded),
            price=price, filled_quantity=abs(filled_lots),
            avg_fill_price=fill_price, fee=fee, status="filled",
            order_id=order_id,
        )

    # ------------------------------------------------------------------
    # Position + account queries
    # ------------------------------------------------------------------

    @_with_reconnect
    async def get_single_position(self, asset: str) -> float:
        """Fetch current position for self._symbol_name as signed fraction.

        On exception: returns LAST KNOWN position (never fake-flat 0.0,
        which would cause the engine to re-open positions = double exposure).
        Mirrors cTrader's anti-fake-flat guard.
        """
        try:
            await self._refresh_position_cache()
        except Exception as exc:  # noqa: BLE001
            logger.error("OANDA get_single_position failed: %s", exc)
            # Fall through to cached value below

        price = self._mid_price if self._mid_price > 0 else 1.0
        lots = self._position_units / self._lot_size
        return self._lots_to_position(lots, self._portfolio_value, price)

    async def _refresh_position_cache(self) -> None:
        """Update self._position_units from OANDA. NETTING-aware."""
        # 2s TTL to avoid hammering during a single engine tick that calls
        # both get_single_position and execute_position_change.
        now = time.time()
        if now - self._position_cache_at < _POSITION_CACHE_TTL_SEC:
            return

        resp = await self._session.get(
            f"/v3/accounts/{self._account_id}/positions/{self._oanda_symbol}",
        )
        if resp.status_code == 404:
            # No position ever opened on this instrument — that's flat.
            self._position_units = 0
            self._position_cache_at = now
            return
        resp.raise_for_status()
        payload = self._safe_json(resp).get("position", {})

        long_units = int(payload.get("long", {}).get("units", "0"))
        short_units = int(payload.get("short", {}).get("units", "0"))
        # In NETTING mode one of these is 0; sum is the net position.
        # short_units is reported as a negative integer by OANDA.
        net_units = long_units + short_units
        self._position_units = net_units
        self._position_cache_at = now

    @_with_reconnect
    async def get_account_info(self) -> dict:
        """Fetch NAV, balance, margin in the same shape as cTrader returns."""
        try:
            summary = await self._get_account_summary()
        except Exception as exc:  # noqa: BLE001
            logger.error("OANDA get_account_info failed: %s", exc)
            return {
                "total_equity": self._portfolio_value,
                "available_balance": self._portfolio_value,
                "used_margin": 0.0,
                "balance": 0.0,
                "unrealized_pnl": 0.0,
                "n_positions": 0,
                "mid_price": self._mid_price,
            }

        nav = float(summary["NAV"])
        balance = float(summary["balance"])
        unrealized_pnl = float(summary.get("unrealizedPL", "0"))
        used_margin = float(summary.get("marginUsed", "0"))
        margin_available = float(summary.get("marginAvailable", str(nav)))
        n_positions = int(summary.get("openPositionCount", 0))

        self._portfolio_value = nav

        return {
            "total_equity": nav,
            "available_balance": margin_available,
            "used_margin": used_margin,
            "balance": balance,
            "unrealized_pnl": unrealized_pnl,
            "n_positions": n_positions,
            "mid_price": self._mid_price,
        }

    async def get_funding_rates(self, assets: list[str]) -> dict[str, float]:
        """CFDs have no funding rates. Return zeros."""
        return {a: 0.0 for a in assets}

    @_with_reconnect
    async def check_orphaned_positions(self) -> list[dict]:
        """Enumerate positions for this symbol.

        OANDA NETTING accounts cannot generate orphans by construction —
        long.units * short.units == 0 always. If we somehow see both
        non-zero, log CRITICAL and return both entries so the engine
        halt path engages.
        """
        try:
            resp = await self._session.get(
                f"/v3/accounts/{self._account_id}/positions/{self._oanda_symbol}",
            )
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            payload = self._safe_json(resp).get("position", {})
        except Exception as exc:  # noqa: BLE001
            logger.error("OANDA check_orphaned_positions failed: %s", exc)
            return []

        long_obj = payload.get("long", {})
        short_obj = payload.get("short", {})
        long_units = int(long_obj.get("units", "0"))
        short_units = int(short_obj.get("units", "0"))

        positions: list[dict] = []
        if long_units != 0:
            positions.append({
                "position_id": f"{self._account_id}/{self._oanda_symbol}/long",
                "side": "BUY",
                "lots": long_units / self._lot_size,
                "entry_price": float(long_obj.get("averagePrice", "0")),
            })
        if short_units != 0:
            positions.append({
                "position_id": f"{self._account_id}/{self._oanda_symbol}/short",
                "side": "SELL",
                "lots": abs(short_units) / self._lot_size,
                "entry_price": float(short_obj.get("averagePrice", "0")),
            })

        if len(positions) > 1:
            logger.critical(
                "OANDA orphan-like state: NETTING account reports BOTH long "
                "and short legs non-zero (long=%d short=%d). This should be "
                "impossible in NETTING mode. Halting recommended.",
                long_units, short_units,
            )
        elif len(positions) == 1:
            p = positions[0]
            logger.info(
                "Single position confirmed: %s %.4f lots @ %.5f",
                p["side"], p["lots"], p["entry_price"],
            )
        else:
            logger.info("No open position for %s", self._symbol_name)

        return positions

    @_with_reconnect
    async def close_position_by_id(self, position_id: int, volume: int) -> bool:
        """NETTING compat stub. Delegates to emergency_flatten.

        cTrader-side callers (CT-06 close-then-open path) don't run on
        OANDA — but the engine occasionally probes via this method, so
        provide a sane response.
        """
        logger.info(
            "OANDA close_position_by_id called (NETTING delegates to flatten): "
            "position_id=%s volume=%d",
            position_id, volume,
        )
        result = await self.emergency_flatten([self._symbol_name])
        return result.n_failed == 0

    @_with_reconnect
    async def emergency_flatten(self, assets: list[str]) -> RebalanceResult:
        """Close all positions on the configured instrument via bulk close."""
        orders: list[OrderResult] = []
        n_failed = 0
        n_executed = 0

        for attempt in range(3):
            try:
                # Refresh state — we may have raced an in-flight order
                await self._refresh_position_cache()
                if self._position_units == 0:
                    logger.info(
                        "OANDA emergency_flatten: no position on %s",
                        self._symbol_name,
                    )
                    break

                # OANDA bulk close — handles long and short legs in one PUT
                close_body = {"longUnits": "ALL", "shortUnits": "ALL"}
                resp = await self._session.put(
                    f"/v3/accounts/{self._account_id}/positions/"
                    f"{self._oanda_symbol}/close",
                    json=close_body,
                    timeout=self._market_fallback_timeout,
                )

                if resp.status_code >= 400:
                    err = self._safe_json(resp).get(
                        "errorMessage", f"HTTP {resp.status_code}",
                    )
                    logger.error(
                        "OANDA flatten attempt %d failed: %s", attempt + 1, err,
                    )
                    n_failed += 1
                    orders.append(OrderResult(
                        asset=self._symbol_name, symbol=self._symbol_name,
                        side="unknown", order_type="market",
                        quantity=abs(self._position_units) / self._lot_size,
                        price=0.0, filled_quantity=0.0, avg_fill_price=0.0,
                        fee=0.0, status="failed", error=err,
                    ))
                    await asyncio.sleep(1.0)
                    continue

                payload = self._safe_json(resp)
                # Long-leg fill
                for key, side_label in (
                    ("longOrderFillTransaction", "sell"),
                    ("shortOrderFillTransaction", "buy"),
                ):
                    fill = payload.get(key)
                    if not fill:
                        continue
                    units = abs(int(fill.get("units", "0")))
                    if units == 0:
                        continue
                    fill_price = float(fill.get("price", self._mid_price))
                    commission = float(fill.get("commission", "0"))
                    lots = units / self._lot_size
                    notional = lots * self._lot_size * fill_price
                    fee = abs(commission) + notional * self._taker_fee
                    orders.append(OrderResult(
                        asset=self._symbol_name, symbol=self._symbol_name,
                        side=side_label, order_type="market",
                        quantity=lots, price=self._mid_price,
                        filled_quantity=lots, avg_fill_price=fill_price,
                        fee=fee, status="filled",
                    ))
                    n_executed += 1
                    logger.info(
                        "OANDA flatten: closed %d units (%.4f lots) @ %.5f",
                        units, lots, fill_price,
                    )

                # Confirm flat
                self._position_units = 0
                self._position_cache_at = time.time()
                break

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "OANDA emergency_flatten attempt %d exception: %s",
                    attempt + 1, exc,
                )
                n_failed += 1
                await asyncio.sleep(1.0)

        total_fees = sum(o.fee for o in orders)
        return RebalanceResult(
            orders=orders,
            total_fees=total_fees,
            n_executed=n_executed,
            n_failed=n_failed,
        )

    @_with_reconnect
    async def get_deal_list(self, from_days_ago: int = 7) -> list[dict]:
        """Fetch ORDER_FILL transactions for audit. Optional."""
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(days=from_days_ago)
        try:
            resp = await self._session.get(
                f"/v3/accounts/{self._account_id}/transactions",
                params={
                    "from": since.isoformat().replace("+00:00", "Z"),
                    "type": "ORDER_FILL",
                },
            )
            resp.raise_for_status()
            payload = self._safe_json(resp)
        except Exception as exc:  # noqa: BLE001
            logger.error("OANDA get_deal_list failed: %s", exc)
            return []

        out = []
        for txn in payload.get("transactions", []):
            try:
                units = int(txn.get("units", "0"))
                out.append({
                    "txn_id": txn.get("id"),
                    "time": txn.get("time"),
                    "instrument": txn.get("instrument"),
                    "units": units,
                    "lots": units / self._lot_size,
                    "price": float(txn.get("price", "0")),
                    "commission": float(txn.get("commission", "0")),
                    "pnl": float(txn.get("pl", "0")),
                    "reason": txn.get("reason"),
                })
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("OANDA txn parse error: %s txn=%r", exc, txn)
        return out

    # ------------------------------------------------------------------
    # Position <-> lots conversion (mirrors cTrader semantics)
    # ------------------------------------------------------------------

    def _position_to_lots(
        self, fraction: float, portfolio_value: float, price: float,
    ) -> float:
        """Convert fraction [-1, 1] to signed lots under leverage cap.

        Identical math to CTraderBroker._position_to_lots. Agent targets
        notional = |fraction| × PV; lots = notional / (price × lot_size).
        Floor-up to min_lot when conviction ≥ 0.1 and the min_lot notional
        fits under the leverage cap; otherwise skip (return 0).
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0

        leverage = max(float(self._leverage), 1.0)
        max_notional = portfolio_value * leverage
        min_lot_notional = self._min_lot * self._lot_size * price

        if min_lot_notional > max_notional:
            if abs(fraction) >= 0.1:
                logger.warning(
                    "OANDA sizing skip: min_lot %.4f @ %.5f (notional $%.0f) "
                    "exceeds leverage cap ($%.0f = PV $%.0f × %.1fx). "
                    "fraction=%.3f. Top-up account or raise leverage.",
                    self._min_lot, price, min_lot_notional,
                    max_notional, portfolio_value, leverage, fraction,
                )
            return 0.0

        notional = abs(fraction) * portfolio_value
        lots = notional / (price * self._lot_size)
        lots = round(lots / self._min_lot) * self._min_lot

        max_lots = (
            int((max_notional / (price * self._lot_size)) / self._min_lot)
            * self._min_lot
        )
        if lots > max_lots:
            lots = max_lots

        if lots < self._min_lot and abs(fraction) >= 0.1:
            lots = self._min_lot

        return lots * float(np.sign(fraction))

    def _lots_to_position(
        self, lots: float, portfolio_value: float, price: float,
    ) -> float:
        """Convert signed lots to fraction [-1, 1]."""
        if portfolio_value <= 0 or price <= 0:
            return 0.0
        notional = abs(lots) * price * self._lot_size
        fraction = notional / portfolio_value
        return float(np.clip(fraction * np.sign(lots), -1.0, 1.0))

    def lots_to_units_signed(self, lots: float) -> int:
        """Signed lots -> signed OANDA integer units (sign carries direction)."""
        return int(round(lots * self._lot_size))

    def units_to_lots(self, units: int) -> float:
        """Signed integer units -> signed lots."""
        if self._lot_size <= 0:
            return 0.0
        return float(units) / self._lot_size

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    async def _get_account_summary(self) -> dict:
        """GET /v3/accounts/{id}/summary -> .account dict."""
        resp = await self._session.get(
            f"/v3/accounts/{self._account_id}/summary",
        )
        resp.raise_for_status()
        return self._safe_json(resp).get("account", {})

    async def _fetch_instrument_metadata(self) -> None:
        """GET instrument metadata + cache pip/precision/min-trade."""
        resp = await self._session.get(
            f"/v3/accounts/{self._account_id}/instruments",
            params={"instruments": self._oanda_symbol},
        )
        resp.raise_for_status()
        instruments = self._safe_json(resp).get("instruments", [])
        if not instruments:
            raise RuntimeError(
                f"OANDA instrument {self._oanda_symbol} not tradeable on "
                f"account {self._account_id}"
            )
        meta = instruments[0]
        self._pip_location = int(meta.get("pipLocation", -4))
        self._display_precision = int(meta.get("displayPrecision", 5))
        self._trade_units_precision = int(meta.get("tradeUnitsPrecision", 0))
        try:
            self._minimum_trade_size = int(float(meta.get("minimumTradeSize", "1")))
        except (ValueError, TypeError):
            self._minimum_trade_size = 1
        logger.info(
            "OANDA instrument %s: pipLocation=%d displayPrecision=%d "
            "minTradeSize=%d",
            self._oanda_symbol, self._pip_location, self._display_precision,
            self._minimum_trade_size,
        )

    @staticmethod
    def _safe_json(resp: httpx.Response) -> dict[str, Any]:
        """Parse JSON; return empty dict on failure (avoid exception in error path)."""
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            logger.debug("OANDA non-JSON response: %r", resp.text[:200])
            return {}
