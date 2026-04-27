"""AlphaSeek v3 — Stage 0.5 shadow fill-model logger (S500 M5 concurrent).

Calibration data collector for the M2 parametric fill model:

    P(fill) = sigmoid(α · OFI_signed + β · spread_inv + γ · age)

Submits small post-only limit orders against Binance perp DEMO (no agent —
random or BBO-tracking strategy) and logs per-order outcomes (fill or
cancel-after-`limit_max_age` bars). After 3–7 days of data, fit α/β/γ via
logistic regression and overwrite the literature defaults in
configs/alphaseek_v3_*.yaml before launching Stage 1.

Defaults match M2 contract: limit_max_age=60 bars (=60s on 1s LOB) and
BBO ± 1 tick post-only placement.

Usage:
    python scripts/stage0_5_shadow_logger.py \
        --symbol BTCUSDT --side random --interval-sec 5 \
        --max-orders 50000 --output-dir data/stage0_5_shadow

Env vars required (already populated in docker/live/.env):
    STAGE0_5_BINANCE_DEMO_API_KEY, STAGE0_5_BINANCE_DEMO_API_SECRET
    (must be a dedicated demo subaccount — sharing with gmgp1-btc/sg1-btc
    causes position_mismatch halts at reconcile; see S501-cont).

Output:
    data/stage0_5_shadow/btcusdt_shadow_<utc_iso>.parquet
    Columns: ts_submit, side, limit_px, mid_at_submit, spread_bps,
             ofi_signed, queue_age_bars (always 0 at submit), ts_outcome,
             outcome, bars_to_fill (None for cancelled).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("stage0_5_shadow")

OFI_WINDOW_BARS = 10
SPREAD_INV_FLOOR = 0.5
DEFAULT_LIMIT_MAX_AGE_BARS = 60
PARQUET_FLUSH_EVERY = 60


@dataclass
class OrderRecord:
    ts_submit: float
    side: int
    limit_px: float
    mid_at_submit: float
    spread_bps: float
    ofi_signed: float
    queue_age_bars: int
    ts_outcome: float | None = None
    outcome: str | None = None
    bars_to_fill: int | None = None
    order_id: str | None = None


class OFITracker:
    """Rolling order-flow imbalance over the last `window` book snapshots."""

    def __init__(self, window: int = OFI_WINDOW_BARS) -> None:
        self.window = int(window)
        self._bid_top: deque[float] = deque(maxlen=window + 1)
        self._ask_top: deque[float] = deque(maxlen=window + 1)
        self._bid_sz: deque[float] = deque(maxlen=window + 1)
        self._ask_sz: deque[float] = deque(maxlen=window + 1)

    def update(self, bid_px: float, bid_sz: float, ask_px: float, ask_sz: float) -> None:
        self._bid_top.append(bid_px)
        self._ask_top.append(ask_px)
        self._bid_sz.append(bid_sz)
        self._ask_sz.append(ask_sz)

    def signed_ofi(self) -> float:
        if len(self._bid_top) < 2:
            return 0.0
        # Cont, Kukanov & Stoikov (2014) signed OFI on top-of-book.
        delta_bid = 0.0
        if self._bid_top[-1] > self._bid_top[-2]:
            delta_bid = self._bid_sz[-1]
        elif self._bid_top[-1] < self._bid_top[-2]:
            delta_bid = -self._bid_sz[-2]
        else:
            delta_bid = self._bid_sz[-1] - self._bid_sz[-2]

        delta_ask = 0.0
        if self._ask_top[-1] < self._ask_top[-2]:
            delta_ask = self._ask_sz[-1]
        elif self._ask_top[-1] > self._ask_top[-2]:
            delta_ask = -self._ask_sz[-2]
        else:
            delta_ask = self._ask_sz[-1] - self._ask_sz[-2]

        return float(delta_bid - delta_ask)


def make_record(
    side: int,
    book_top: dict[str, float],
    ofi: float,
    tick_size: float,
) -> tuple[OrderRecord, float]:
    bid, ask = book_top["bid"], book_top["ask"]
    mid = 0.5 * (bid + ask)
    spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else float("inf")
    # BBO ± 1 tick post-only placement
    limit_px = bid - tick_size if side > 0 else ask + tick_size
    return (
        OrderRecord(
            ts_submit=time.time_ns(),
            side=int(side),
            limit_px=float(limit_px),
            mid_at_submit=float(mid),
            spread_bps=float(spread_bps),
            ofi_signed=float(ofi),
            queue_age_bars=0,
        ),
        limit_px,
    )


async def shadow_loop(args: argparse.Namespace) -> int:
    import ccxt.async_support as ccxt_async

    api_key = os.getenv("STAGE0_5_BINANCE_DEMO_API_KEY", "")
    api_secret = os.getenv("STAGE0_5_BINANCE_DEMO_API_SECRET", "")
    if not (api_key and api_secret):
        logger.error("STAGE0_5_BINANCE_DEMO_API_KEY / SECRET missing — set in env or "
                     "source docker/live/.env first. Must be an isolated demo "
                     "subaccount (not shared with gmgp1-btc or sg1-btc).")
        return 2

    exchange = ccxt_async.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSettle": "USDT"},
    })
    exchange.enable_demo_trading(True)
    logger.info("Binance perp DEMO connected — symbol=%s side=%s interval=%ss",
                args.symbol, args.side, args.interval_sec)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / (
        f"{args.symbol.lower()}_shadow_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.parquet"
    )

    ofi = OFITracker(OFI_WINDOW_BARS)
    pending: dict[str, OrderRecord] = {}
    completed: list[OrderRecord] = []
    running = True

    def _stop(signum: int, _frame: Any) -> None:
        nonlocal running
        logger.info("signal %d received — draining and exiting", signum)
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (OSError, ValueError):
            pass  # Windows / non-main thread

    try:
        markets = await exchange.load_markets()
        market = markets.get(args.symbol) or markets[f"{args.symbol[:-4]}/USDT:USDT"]
        tick_size = float(market["precision"]["price"])
        symbol_unified = market["symbol"]

        order_count = 0
        next_submit = time.monotonic()

        while running and order_count < args.max_orders:
            book = await exchange.fetch_order_book(symbol_unified, limit=5)
            if not (book["bids"] and book["asks"]):
                await asyncio.sleep(args.interval_sec)
                continue
            bid_px, bid_sz = book["bids"][0]
            ask_px, ask_sz = book["asks"][0]
            ofi.update(bid_px, bid_sz, ask_px, ask_sz)

            now = time.monotonic()
            if now >= next_submit:
                next_submit = now + args.interval_sec
                side = random.choice([-1, 1]) if args.side == "random" \
                    else (1 if args.side == "long" else -1)
                record, limit_px = make_record(
                    side,
                    {"bid": bid_px, "ask": ask_px},
                    ofi.signed_ofi(),
                    tick_size,
                )
                try:
                    order = await exchange.create_order(
                        symbol_unified,
                        type="limit",
                        side="buy" if side > 0 else "sell",
                        amount=args.amount,
                        price=limit_px,
                        params={"timeInForce": "GTX"},  # post-only
                    )
                    record.order_id = str(order["id"])
                    pending[record.order_id] = record
                    order_count += 1
                    logger.info("submit #%d id=%s side=%d px=%.2f spread=%.2fbps ofi=%.3f",
                                order_count, record.order_id, side, limit_px,
                                record.spread_bps, record.ofi_signed)
                except Exception as exc:
                    logger.warning("submit failed: %s", exc)

            now_ns = time.time_ns()
            for order_id in list(pending.keys()):
                rec = pending[order_id]
                age_sec = (now_ns - rec.ts_submit) / 1e9
                try:
                    status = await exchange.fetch_order(order_id, symbol_unified)
                except Exception as exc:
                    logger.warning("fetch_order %s failed: %s", order_id, exc)
                    continue
                if status.get("status") == "closed" and float(status.get("filled", 0)) > 0:
                    rec.ts_outcome = now_ns
                    rec.outcome = "filled"
                    rec.bars_to_fill = int(age_sec)
                    completed.append(rec)
                    pending.pop(order_id)
                elif age_sec >= args.limit_max_age_bars:
                    try:
                        await exchange.cancel_order(order_id, symbol_unified)
                    except Exception as exc:
                        logger.warning("cancel %s failed: %s", order_id, exc)
                    rec.ts_outcome = now_ns
                    rec.outcome = "cancelled"
                    completed.append(rec)
                    pending.pop(order_id)

            if len(completed) >= PARQUET_FLUSH_EVERY:
                _flush_parquet(parquet_path, completed)
                logger.info("flushed %d records → %s", len(completed), parquet_path.name)
                completed.clear()

            await asyncio.sleep(min(1.0, args.interval_sec))

        for order_id in list(pending.keys()):
            try:
                await exchange.cancel_order(order_id, symbol_unified)
            except Exception:
                pass
            rec = pending.pop(order_id)
            rec.ts_outcome = time.time_ns()
            rec.outcome = "cancelled_on_shutdown"
            completed.append(rec)
        if completed:
            _flush_parquet(parquet_path, completed)
            logger.info("final flush %d records → %s", len(completed), parquet_path.name)

    finally:
        await exchange.close()
    return 0


def _flush_parquet(path: Path, records: list[OrderRecord]) -> None:
    df_new = pd.DataFrame([asdict(r) for r in records])
    if path.exists():
        df_existing = pd.read_parquet(path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", choices=["random", "long", "short"], default="random")
    parser.add_argument("--interval-sec", type=float, default=5.0,
                        help="seconds between order submissions")
    parser.add_argument("--amount", type=float, default=0.001,
                        help="contract amount per order (BTC notional ~$60 at $60k)")
    parser.add_argument("--limit-max-age-bars", type=int,
                        default=DEFAULT_LIMIT_MAX_AGE_BARS,
                        help="cancel age in bars (=seconds on 1s LOB)")
    parser.add_argument("--max-orders", type=int, default=50000,
                        help="hard stop after this many submissions")
    parser.add_argument("--output-dir", default="data/stage0_5_shadow")
    args = parser.parse_args()
    return asyncio.run(shadow_loop(args))


if __name__ == "__main__":
    sys.exit(main())
