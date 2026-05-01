"""AlphaSeek v3 — Stage 0.5 shadow fill-model logger (S500 M5 concurrent).

Calibration data collector for the M2 parametric fill model:

    P(fill) = sigmoid(α · OFI_signed + β · spread_inv + γ · age)

Three operating modes (see --mode):

    demo          : Submit post-only limit orders against Binance perp DEMO.
                    Legacy default. Calibration data is contaminated by
                    demo's synthetic orderbook and phantom-wick prints
                    (S502 finding) — use only for code-path testing.

    live-passive  : Read-only book observer against PROD Binance liquidity.
                    No orders submitted, ever. Logs hypothetical post-only
                    orders at BBO ± 1 tick and resolves them via subsequent
                    book snapshots (buy fills if best_bid ≤ limit_px; sell
                    fills if best_ask ≥ limit_px). Standard top-of-book
                    passive simulation (Cartea/Jaimungal/Penalva).
                    Trade-off: assumes head-of-queue placement → over-
                    estimates fill probability vs real fills. Apply
                    shrinkage 0.5–0.8 during M2 deployment if needed.

    live-active   : Submit real orders against PROD. RESERVED — requires
                    --i-understand-this-is-real-money plus a funded sub-
                    account with futures-trade permission.

After 3–7 days of data, fit α/β/γ via logistic regression and overwrite
the literature defaults in configs/alphaseek_v3_*.yaml before launching
Stage 1.

Defaults match M2 contract: limit_max_age=60 bars (=60s on 1s LOB) and
BBO ± 1 tick post-only placement.

Usage:
    # Live-passive against prod (S502-cont, recommended):
    python scripts/stage0_5_shadow_logger.py --mode live-passive \
        --symbol BTCUSDT --side random --interval-sec 1 \
        --max-orders 50000

    # Legacy demo (do not use for M2 calibration):
    python scripts/stage0_5_shadow_logger.py --mode demo \
        --symbol BTCUSDT --side random --interval-sec 5 \
        --max-orders 50000

Env vars (resolved per mode from docker/live/.env):
    demo                       → STAGE0_5_BINANCE_DEMO_API_KEY / _SECRET
    live-passive, live-active  → STAGE0_5_BINANCE_LIVE_API_KEY / _SECRET

For demo, the subaccount must be dedicated — sharing with gmgp1-btc /
sg1-btc causes position_mismatch halts at reconcile (S501-cont). For
live-passive, a read-only key with zero capital is sufficient (and
recommended — the script will never call create_order in this mode).

Output (mode-specific default dir):
    demo, live-active : data/stage0_5_shadow/<symbol>_shadow_<utc>.parquet
    live-passive      : data/stage0_5_shadow_live_passive/<symbol>_passive_<utc>.parquet
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
    placement_depth_ticks: int = 1  # 0 = AT BBO; k = k ticks worse than BBO (default 1 = legacy demo behavior)
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


def _resolve_creds(mode: str) -> tuple[str, str, str]:
    """Return (api_key, api_secret, env_prefix) for the given mode."""
    prefix = "STAGE0_5_BINANCE_DEMO" if mode == "demo" else "STAGE0_5_BINANCE_LIVE"
    return (
        os.getenv(f"{prefix}_API_KEY", ""),
        os.getenv(f"{prefix}_API_SECRET", ""),
        prefix,
    )


async def _connect_exchange(mode: str):
    """Create and return a ccxt async Binance client; mode-aware sandbox toggle.

    Returns None on missing credentials so the caller can exit cleanly.
    enable_demo_trading is only invoked for mode=="demo"; live-passive and
    live-active connect to mainnet.
    """
    import ccxt.async_support as ccxt_async

    api_key, api_secret, prefix = _resolve_creds(mode)
    if not (api_key and api_secret):
        logger.error(
            "%s_API_KEY / %s_API_SECRET missing — set in env or "
            "source docker/live/.env first.",
            prefix, prefix,
        )
        return None

    exchange = ccxt_async.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        # Restrict load_markets to linear futures only. Without these two
        # options ccxt also hits spot-wallet (sapi/v1/capital/config/getall)
        # and cross-margin (sapi/v1/margin/allPairs) endpoints which a
        # read-only key without spot/margin scope cannot reach. Same pattern
        # used in finrl_pro_ds/crypto/data/crypto_loader.py:86.
        "options": {
            "defaultType": "swap",
            "defaultSettle": "USDT",
            "fetchCurrencies": False,
            "fetchMarkets": ["linear"],
        },
    })
    if mode == "demo":
        exchange.enable_demo_trading(True)
        logger.info("Binance perp DEMO connected (mode=demo)")
    elif mode == "live-passive":
        logger.info(
            "Binance perp LIVE-PASSIVE connected — read-only book observer, "
            "NO ORDERS will be submitted",
        )
    elif mode == "live-active":
        logger.warning(
            "Binance perp LIVE-ACTIVE connected — REAL ORDERS will be submitted "
            "against the prod book",
        )
    return exchange


def _parse_depth_spec(spec: str) -> list[int]:
    """Parse --placement-depth-ticks spec into a list of ints to sample from.

    Accepts a single integer (fixed depth) or comma-separated integers
    (uniform random sample at submission time). Repeat values to weight:
    "0,0,1" gives 2/3 weight at depth=0 and 1/3 at depth=1.

    Examples:
        "1"          → [1]                (fixed, legacy default)
        "0"          → [0]                (always at BBO, AlphaSeek v3 primary)
        "0,1,2,5,10" → [0,1,2,5,10]       (uniform random over 5 depths)
        "0,0,0,1"    → [0,0,0,1]          (75/25 weighted, via repetition)
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--placement-depth-ticks: empty spec {spec!r}")
    depths: list[int] = []
    for p in parts:
        try:
            d = int(p)
        except ValueError as exc:
            raise ValueError(
                f"--placement-depth-ticks: {p!r} is not an integer (spec={spec!r})",
            ) from exc
        if d < 0:
            raise ValueError(
                f"--placement-depth-ticks: depth must be ≥ 0, got {d} (spec={spec!r})",
            )
        depths.append(d)
    return depths


def _sample_depth(depths: list[int]) -> int:
    """Sample one depth from the configured list (uniform random)."""
    return int(random.choice(depths))


def make_record(
    side: int,
    book_top: dict[str, float],
    ofi: float,
    tick_size: float,
    depth_ticks: int = 1,
) -> tuple[OrderRecord, float]:
    """Build an OrderRecord at the requested placement depth.

    Placement convention:
        depth_ticks=0  → AT BBO (buy at best bid; sell at best ask).
                          Joins back of existing queue at the inside.
        depth_ticks=k  → k ticks WORSE than BBO (buy at best_bid - k*tick;
                          sell at best_ask + k*tick). Posts a new level
                          k ticks deeper into the book; price must move
                          through it for fill.
    """
    bid, ask = book_top["bid"], book_top["ask"]
    mid = 0.5 * (bid + ask)
    spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else float("inf")
    offset = depth_ticks * tick_size
    limit_px = (bid - offset) if side > 0 else (ask + offset)
    return (
        OrderRecord(
            ts_submit=time.time_ns(),
            side=int(side),
            limit_px=float(limit_px),
            mid_at_submit=float(mid),
            spread_bps=float(spread_bps),
            ofi_signed=float(ofi),
            queue_age_bars=0,
            placement_depth_ticks=int(depth_ticks),
        ),
        limit_px,
    )


async def _active_loop(args: argparse.Namespace, exchange) -> int:
    """Submit real (or demo) post-only limit orders and log outcomes.

    Used for mode in {demo, live-active}. The original shadow-loop body,
    extracted so the connection setup is shared with the passive observer.
    """
    depths = _parse_depth_spec(args.placement_depth_ticks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_tag = "live_active" if args.mode == "live-active" else "shadow"
    parquet_path = output_dir / (
        f"{args.symbol.lower()}_{file_tag}_"
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
                    depth_ticks=_sample_depth(depths),
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


async def _passive_loop(args: argparse.Namespace, exchange) -> int:
    """Read-only book observer with simulated post-only fills.

    No order submission anywhere in this code path — by construction, not
    by gating. On each cycle: fetch book, resolve any pending hypothetical
    orders, optionally submit a new hypothetical at the configured cadence.

    Fill criterion (top-of-book passive simulation, strict):
        buy at limit_px  → filled iff subsequent best_bid <  limit_px
                           (best bid moved BELOW our level → our queue wiped)
        sell at limit_px → filled iff subsequent best_ask >  limit_px
                           (best ask moved ABOVE our level → our queue lifted)

    Strict-inequality avoids the false-fill case where best bid merely
    touches our limit (= our level became top-of-book but no aggressor
    has yet hit us). On 1-tick-wide spreads (typical for BTCUSDT perp),
    `≤` produces ~100% spurious fill rates from normal micro-movement;
    `<` requires the price to actually move past our level.

    Bias remaining: assumes head-of-queue placement and ignores self-
    impact, so fill probability is still over-estimated vs real fills.
    For depth=0 (AT BBO) the bias is largest because we miss the case
    where aggressors consume our queue without best_bid moving down.
    Apply shrinkage 0.5–0.8 to fitted α/β/γ at M2 deployment time.

    Multi-depth: depth_ticks is sampled per hypothetical from
    args.placement_depth_ticks (see _parse_depth_spec). Each record's
    placement_depth_ticks column lets the M2 fit recover P(fill | depth).
    """
    depths = _parse_depth_spec(args.placement_depth_ticks)
    logger.info("placement depth distribution: %s (sampled per hypothetical)", depths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / (
        f"{args.symbol.lower()}_passive_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.parquet"
    )

    ofi = OFITracker(OFI_WINDOW_BARS)
    pending: list[OrderRecord] = []
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
        # In passive mode, the limit-age unit is "bars" but resolved against
        # observed snapshots rather than wall-clock seconds. With 1s polling
        # cadence the two are equivalent; with slower polling they diverge.

        while running and order_count < args.max_orders:
            try:
                book = await exchange.fetch_order_book(symbol_unified, limit=5)
            except Exception as exc:
                logger.warning("fetch_order_book failed: %s", exc)
                await asyncio.sleep(args.interval_sec)
                continue
            if not (book["bids"] and book["asks"]):
                await asyncio.sleep(args.interval_sec)
                continue
            bid_px, _bid_sz = book["bids"][0]
            ask_px, _ask_sz = book["asks"][0]
            ofi.update(bid_px, _bid_sz, ask_px, _ask_sz)
            now_ns = time.time_ns()

            # Resolve pending hypothetical orders against the new snapshot.
            # Strict inequality: best_bid must be BELOW (buy) / best_ask
            # ABOVE (sell) our limit, meaning the price moved past our
            # resting level and our queue would have been wiped. Touching
            # our level (=) only means we're now top-of-book; not yet hit.
            still_pending: list[OrderRecord] = []
            for rec in pending:
                age_sec = (now_ns - rec.ts_submit) / 1e9
                if rec.side > 0 and bid_px < rec.limit_px:
                    rec.outcome = "filled"
                    rec.ts_outcome = now_ns
                    rec.bars_to_fill = int(age_sec)
                    completed.append(rec)
                elif rec.side < 0 and ask_px > rec.limit_px:
                    rec.outcome = "filled"
                    rec.ts_outcome = now_ns
                    rec.bars_to_fill = int(age_sec)
                    completed.append(rec)
                elif age_sec >= args.limit_max_age_bars:
                    rec.outcome = "cancelled"
                    rec.ts_outcome = now_ns
                    completed.append(rec)
                else:
                    still_pending.append(rec)
            pending = still_pending

            # Submit a new hypothetical at the cadence.
            now_mono = time.monotonic()
            if now_mono >= next_submit:
                next_submit = now_mono + args.interval_sec
                side = random.choice([-1, 1]) if args.side == "random" \
                    else (1 if args.side == "long" else -1)
                record, _limit_px = make_record(
                    side,
                    {"bid": bid_px, "ask": ask_px},
                    ofi.signed_ofi(),
                    tick_size,
                    depth_ticks=_sample_depth(depths),
                )
                pending.append(record)
                order_count += 1
                if order_count == 1 or order_count % 60 == 0:
                    n_done = len(completed)
                    n_filled = sum(1 for c in completed if c.outcome == "filled")
                    fill_rate = (n_filled / n_done * 100.0) if n_done else 0.0
                    logger.info(
                        "hypothetical #%d  pending=%d  completed=%d  "
                        "fill_rate=%.1f%%  spread=%.2fbps  ofi=%.3f  depth=%d",
                        order_count, len(pending), n_done,
                        fill_rate, record.spread_bps, record.ofi_signed,
                        record.placement_depth_ticks,
                    )

            if len(completed) >= PARQUET_FLUSH_EVERY:
                _flush_parquet(parquet_path, completed)
                logger.info("flushed %d records → %s", len(completed), parquet_path.name)
                completed.clear()

            await asyncio.sleep(args.interval_sec)

        for rec in pending:
            rec.outcome = "cancelled_on_shutdown"
            rec.ts_outcome = time.time_ns()
            completed.append(rec)
        if completed:
            _flush_parquet(parquet_path, completed)
            logger.info("final flush %d records → %s", len(completed), parquet_path.name)

    finally:
        await exchange.close()
    return 0


async def shadow_loop(args: argparse.Namespace) -> int:
    """Mode-dispatching entry point.

    Routes to _active_loop (demo, live-active) or _passive_loop
    (live-passive). Validates the live-active confirm flag before
    connecting to the exchange so a typo can't reach the order book.
    """
    if args.mode == "live-active" and not getattr(
        args, "i_understand_this_is_real_money", False,
    ):
        logger.error(
            "mode=live-active requires --i-understand-this-is-real-money "
            "to confirm intent. Refusing to launch.",
        )
        return 2

    exchange = await _connect_exchange(args.mode)
    if exchange is None:
        return 2

    logger.info(
        "starting mode=%s symbol=%s side=%s interval=%ss output_dir=%s",
        args.mode, args.symbol, args.side, args.interval_sec, args.output_dir,
    )

    if args.mode == "live-passive":
        return await _passive_loop(args, exchange)
    return await _active_loop(args, exchange)


def _flush_parquet(path: Path, records: list[OrderRecord]) -> None:
    df_new = pd.DataFrame([asdict(r) for r in records])
    if path.exists():
        df_existing = pd.read_parquet(path)
        df_new = pd.concat([df_existing, df_new], ignore_index=True)
    df_new.to_parquet(path, index=False)


def main() -> int:
    # Windows compatibility: aiodns (used by aiohttp inside ccxt async) requires
    # SelectorEventLoop; the default Windows ProactorEventLoop causes
    # `aiodns.error.DNSError: Timeout while contacting DNS servers` even when
    # system DNS is healthy. No-op on Linux (Docker deploys), where Selector
    # is already the default policy.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["demo", "live-passive", "live-active"],
        default="demo",
        help=(
            "demo: post-only orders against Binance demo (legacy default; "
            "data contaminated by demo wicks per S502). "
            "live-passive: read-only book observer with simulated fills "
            "against PROD liquidity (zero capital, zero risk; recommended "
            "for M2 calibration). "
            "live-active: real orders against PROD (RESERVED — requires "
            "--i-understand-this-is-real-money + funded sub-account)."
        ),
    )
    parser.add_argument(
        "--i-understand-this-is-real-money",
        dest="i_understand_this_is_real_money",
        action="store_true",
        help="Required confirmation to launch --mode live-active.",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", choices=["random", "long", "short"], default="random")
    parser.add_argument("--interval-sec", type=float, default=5.0,
                        help="seconds between order submissions "
                             "(passive mode also uses this as poll cadence)")
    parser.add_argument("--amount", type=float, default=0.001,
                        help="contract amount per order (active modes only; "
                             "ignored in live-passive)")
    parser.add_argument("--limit-max-age-bars", type=int,
                        default=DEFAULT_LIMIT_MAX_AGE_BARS,
                        help="cancel age in bars (=seconds on 1s LOB)")
    parser.add_argument(
        "--placement-depth-ticks",
        type=str,
        default="0,1,2,5,10",
        help=(
            "Placement depth(s) in ticks worse than BBO. Single int = fixed; "
            "comma-separated = uniform random sample per hypothetical (repeat "
            "values to weight). depth=0 places AT BBO (AlphaSeek v3 primary); "
            "depth=k places k ticks deeper. Default '0,1,2,5,10' covers v3's "
            "primary (0) + retry (1) + sensitivity range for P(fill | depth)."
        ),
    )
    parser.add_argument("--max-orders", type=int, default=50000,
                        help="hard stop after this many submissions "
                             "(or hypothetical submissions in passive mode)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="parquet output directory (default depends on --mode: "
             "data/stage0_5_shadow_live_passive for live-passive, "
             "data/stage0_5_shadow otherwise)",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            "data/stage0_5_shadow_live_passive"
            if args.mode == "live-passive"
            else "data/stage0_5_shadow"
        )
    return asyncio.run(shadow_loop(args))


if __name__ == "__main__":
    sys.exit(main())
