"""Close all orphaned positions on a cTrader account.

Standalone script that connects to cTrader Open API, enumerates all open
positions, prints them for review, then closes each one individually.

Usage (inside Docker container):
    python scripts/close_orphaned_positions.py [--dry-run]

Requires env vars: CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET,
                   CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# cTrader volume encoding: api `volume` is in centi-units of the base
# currency, NOT hundredths of a lot.  For a correct lots display we need
# `lotSize` from ProtoOASymbol per position.  This admin script currently
# skips that lookup — the `lots` field below is really "centi-units / 100"
# which for XAUUSD happens to print 100× larger than the real lot count.
# The CLOSE action passes td.volume through unchanged, so closes work
# correctly regardless of the display glitch.
_VOLUME_SCALE = 100
_SPOT_PRICE_SCALE = 100_000


class PositionCleaner:
    """Minimal cTrader client for position enumeration and closing."""

    def __init__(self):
        self._client_id = os.environ.get("CTRADER_CLIENT_ID", "")
        self._client_secret = os.environ.get("CTRADER_CLIENT_SECRET", "")
        self._access_token = os.environ.get("CTRADER_ACCESS_TOKEN", "")
        self._account_id = int(os.environ.get("CTRADER_ACCOUNT_ID", "0"))
        self._client = None
        self._connected = False
        self._bid = 0.0
        self._ask = 0.0
        self._mid_price = 0.0
        self._money_digits = 2

        if not all([self._client_id, self._client_secret, self._access_token, self._account_id]):
            raise ValueError(
                "Missing cTrader credentials. Set CTRADER_CLIENT_ID, "
                "CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID"
            )

    async def connect(self) -> None:
        """Connect and authenticate to cTrader demo API."""
        from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAAccountAuthReq,
            ProtoOAApplicationAuthReq,
            ProtoOATraderReq,
        )

        host = EndPoints.PROTOBUF_DEMO_HOST  # Always demo for this script
        port = EndPoints.PROTOBUF_PORT

        logger.info(f"Connecting to cTrader DEMO at {host}:{port}")

        self._client = Client(host, port, TcpProtocol)
        self._client.setMessageReceivedCallback(self._on_message)
        self._client.setDisconnectedCallback(self._on_disconnected)

        from twisted.internet import reactor
        reactor.callFromThread(self._client.startService)

        connected_deferred = self._client.whenConnected(failAfterFailures=3)
        await self._deferred_to_future(connected_deferred, timeout=15.0)
        self._connected = True
        logger.info("TCP connected")

        # Application auth
        app_auth = ProtoOAApplicationAuthReq()
        app_auth.clientId = self._client_id
        app_auth.clientSecret = self._client_secret
        app_res = await self._send_request(app_auth, timeout=10.0)
        self._check_error(Protobuf.extract(app_res), "Application auth")
        logger.info("Application authenticated")

        # Account auth
        acct_auth = ProtoOAAccountAuthReq()
        acct_auth.ctidTraderAccountId = self._account_id
        acct_auth.accessToken = self._access_token
        acct_res = await self._send_request(acct_auth, timeout=10.0)
        self._check_error(Protobuf.extract(acct_res), "Account auth")
        logger.info(f"Account {self._account_id} authenticated")

        # Get money digits
        trader_req = ProtoOATraderReq()
        trader_req.ctidTraderAccountId = self._account_id
        trader_res = await self._send_request(trader_req, timeout=10.0)
        trader_payload = Protobuf.extract(trader_res)
        trader = trader_payload.trader
        self._money_digits = trader.moneyDigits
        balance = trader.balance / (10 ** self._money_digits)
        logger.info(f"Account balance: ${balance:,.2f}")

    async def list_positions(self) -> list:
        """Enumerate all open positions with details."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAReconcileReq,
        )

        recon_req = ProtoOAReconcileReq()
        recon_req.ctidTraderAccountId = self._account_id
        recon_res = await self._send_request(recon_req, timeout=10.0)
        recon_payload = Protobuf.extract(recon_res)

        positions = []
        for pos in recon_payload.position:
            td = pos.tradeData
            side = "BUY" if td.tradeSide == 1 else "SELL"
            lots = td.volume / _VOLUME_SCALE
            entry_price = pos.price  # double, already decimal
            used_margin = pos.usedMargin / (10 ** self._money_digits) if pos.usedMargin else 0.0
            positions.append({
                "position_id": pos.positionId,
                "symbol_id": td.symbolId,
                "symbol_name": getattr(td, "symbolName", f"id={td.symbolId}"),
                "side": side,
                "lots": lots,
                "volume": td.volume,
                "entry_price": entry_price,
                "used_margin": used_margin,
            })

        return positions

    async def close_position(self, position_id: int, volume: int) -> bool:
        """Close a single position by ID. Returns True on success."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAClosePositionReq,
        )

        try:
            close_req = ProtoOAClosePositionReq()
            close_req.ctidTraderAccountId = self._account_id
            close_req.positionId = position_id
            close_req.volume = volume
            await self._send_request(close_req, timeout=15.0)
            return True
        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            return False

    async def close(self) -> None:
        """Disconnect cleanly."""
        if self._client:
            try:
                from twisted.internet import reactor
                reactor.callFromThread(self._client.stopService)
            except Exception:
                pass
        self._connected = False

    # --- Internal helpers (copied from CTraderBroker) ---

    def _on_message(self, client, message):
        """Handle incoming messages (spot prices, errors)."""
        from ctrader_open_api import Protobuf
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
            ProtoOAPayloadType,
        )
        payload_type = message.payloadType
        if payload_type == ProtoOAPayloadType.PROTO_OA_ERROR_RES:
            try:
                err = Protobuf.extract(message)
                logger.error(
                    f"cTrader error: {err.errorCode} - "
                    f"{err.description if hasattr(err, 'description') else '(no desc)'}"
                )
            except Exception:
                pass

    def _on_disconnected(self, client, reason):
        self._connected = False
        logger.warning(f"Disconnected: {reason}")

    async def _send_request(self, message, timeout: float = 10.0):
        """Send protobuf request and await response."""
        if self._client is None or not self._connected:
            raise RuntimeError("Not connected")

        from twisted.internet import reactor

        result = {"value": None, "error": None, "done": False}

        def _do_send():
            try:
                d = self._client.send(
                    message, responseTimeoutInSeconds=timeout,
                )
                def on_success(r):
                    result["value"] = r
                    result["done"] = True
                    return r
                def on_error(f):
                    result["error"] = f
                    result["done"] = True
                d.addCallback(on_success)
                d.addErrback(on_error)
            except Exception as exc:
                result["error"] = exc
                result["done"] = True

        reactor.callFromThread(_do_send)

        deadline = time.monotonic() + timeout + 5
        while not result["done"]:
            if time.monotonic() > deadline:
                raise asyncio.TimeoutError(f"Request timed out after {timeout}s")
            await asyncio.sleep(0.05)

        if result["error"] is not None:
            failure = result["error"]
            if hasattr(failure, "raiseException"):
                failure.raiseException()
            raise RuntimeError(f"Request failed: {failure}")

        return result["value"]

    async def _deferred_to_future(self, deferred, timeout: float = 15.0):
        """Convert Twisted Deferred to asyncio-awaitable."""
        from twisted.internet import reactor

        result = {"value": None, "error": None, "done": False}

        def _attach():
            def on_success(r):
                result["value"] = r
                result["done"] = True
                return r
            def on_error(f):
                result["error"] = f
                result["done"] = True
            deferred.addCallback(on_success)
            deferred.addErrback(on_error)

        reactor.callFromThread(_attach)

        deadline = time.monotonic() + timeout + 5
        while not result["done"]:
            if time.monotonic() > deadline:
                raise asyncio.TimeoutError(f"Deferred timed out after {timeout}s")
            await asyncio.sleep(0.05)

        if result["error"] is not None:
            failure = result["error"]
            if hasattr(failure, "raiseException"):
                failure.raiseException()
            raise RuntimeError(f"Deferred failed: {failure}")

        return result["value"]

    @staticmethod
    def _check_error(payload, step: str) -> None:
        if hasattr(payload, "errorCode"):
            desc = getattr(payload, "description", "unknown")
            raise RuntimeError(f"{step} failed: {payload.errorCode} - {desc}")


async def main_async(dry_run: bool = False) -> int:
    """Main async entrypoint. Returns exit code (0=success, 1=failure)."""
    cleaner = PositionCleaner()

    try:
        await cleaner.connect()
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return 1

    try:
        # Step 1: List all positions
        positions = await cleaner.list_positions()

        if not positions:
            logger.info("No open positions found. Nothing to do.")
            return 0

        # Step 2: Print position table
        print(f"\n{'='*80}")
        print(f"  OPEN POSITIONS — Account {cleaner._account_id}")
        print(f"{'='*80}")
        print(f"  {'PosID':<12} {'Side':<6} {'Lots':>8} {'Entry Price':>14} {'Margin':>12}")
        print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*14} {'-'*12}")

        total_buy_lots = 0.0
        total_sell_lots = 0.0
        for p in positions:
            print(
                f"  {p['position_id']:<12} {p['side']:<6} "
                f"{p['lots']:>8.2f} {p['entry_price']:>14.2f} "
                f"${p['used_margin']:>11,.2f}"
            )
            if p["side"] == "BUY":
                total_buy_lots += p["lots"]
            else:
                total_sell_lots += p["lots"]

        print(f"  {'-'*54}")
        print(f"  Total: {len(positions)} positions | "
              f"BUY={total_buy_lots:.2f} lots | SELL={total_sell_lots:.2f} lots | "
              f"Net={total_buy_lots - total_sell_lots:.2f} lots")
        print(f"{'='*80}\n")

        if dry_run:
            logger.info("DRY RUN — no positions will be closed.")
            return 0

        # Step 3: Close each position
        closed = 0
        failed = 0
        for p in positions:
            logger.info(
                f"Closing position {p['position_id']} "
                f"({p['side']} {p['lots']:.2f} lots @ {p['entry_price']:.2f})..."
            )
            success = await cleaner.close_position(p["position_id"], p["volume"])
            if success:
                closed += 1
                logger.info(f"  -> CLOSED position {p['position_id']}")
            else:
                failed += 1
                logger.error(f"  -> FAILED position {p['position_id']}")
            # Small delay between closes to avoid rate limiting
            await asyncio.sleep(0.3)

        # Step 4: Verify — re-list positions
        await asyncio.sleep(1.0)
        remaining = await cleaner.list_positions()

        print(f"\n{'='*80}")
        print(f"  RESULTS")
        print(f"{'='*80}")
        print(f"  Closed:    {closed}")
        print(f"  Failed:    {failed}")
        print(f"  Remaining: {len(remaining)}")
        print(f"{'='*80}\n")

        if remaining:
            logger.warning(f"{len(remaining)} positions still open!")
            for p in remaining:
                logger.warning(
                    f"  Remaining: {p['position_id']} {p['side']} "
                    f"{p['lots']:.2f} lots"
                )
            return 1

        logger.info("All positions closed successfully.")
        return 0

    finally:
        await cleaner.close()


def main():
    parser = argparse.ArgumentParser(
        description="Close all orphaned positions on cTrader account"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List positions without closing them"
    )
    args = parser.parse_args()

    # Start Twisted reactor in daemon thread (same pattern as run_live_ctrader.py)
    from twisted.internet import reactor
    reactor_thread = threading.Thread(
        target=reactor.run, args=(False,), daemon=True
    )
    reactor_thread.start()

    exit_code = asyncio.run(main_async(dry_run=args.dry_run))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
