"""
Bitfinex WebSocket LOB Recorder — Collect live order book snapshots.
====================================================================

Connects to Bitfinex WS v2 public API (no auth needed).
Subscribes to the book channel for tBTCF0:USTF0 (BTC perpetual).
Maintains a local order book from snapshot + incremental updates.
Records BBO (best bid/offer) snapshots at configurable interval.

Output: data/bitfinex/lob_snapshots_YYYYMMDD_HHMMSS.parquet
  Columns: timestamp, bid_price_1, bid_size_1, bid_count_1,
           ask_price_1, ask_size_1, ask_count_1,
           mid_price, spread, spread_bps,
           bid_depth_5, ask_depth_5 (sum of top 5 levels)

Usage:
  # Record for 24 hours (default), snapshot every 1 second
  python scripts/record_bitfinex_lob.py

  # Record for 48 hours, snapshot every 500ms
  python scripts/record_bitfinex_lob.py --duration 48 --interval 0.5

  # Record with 25-level depth (default), flush every 5 minutes
  python scripts/record_bitfinex_lob.py --depth 25 --flush-interval 300

  # Quick test: 5 minutes
  python scripts/record_bitfinex_lob.py --duration 0.083

Controls:
  Ctrl+C to stop gracefully (saves remaining data before exit).

Bitfinex WS v2 Book Protocol:
  - Subscribe → receive full snapshot (array of [PRICE, COUNT, AMOUNT])
  - Then incremental updates: [PRICE, COUNT, AMOUNT]
    - COUNT > 0: update/add price level
    - COUNT = 0: delete price level (AMOUNT=1 for bid, AMOUNT=-1 for ask)
  - AMOUNT > 0 = bid side, AMOUNT < 0 = ask side
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import websocket

# Force unbuffered output (Windows buffers stdout in non-interactive mode)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

WSS_URL = "wss://api-pub.bitfinex.com/ws/2"
SYMBOL = "tBTCF0:USTF0"
DATA_DIR = Path(__file__).parent.parent / "data" / "bitfinex"


class OrderBook:
    """Local order book maintained from Bitfinex WS snapshots + updates."""

    def __init__(self):
        self.bids: OrderedDict[float, tuple[int, float]] = OrderedDict()  # price → (count, amount)
        self.asks: OrderedDict[float, tuple[int, float]] = OrderedDict()  # price → (count, amount)
        self.initialized = False
        self.last_update = 0.0

    def process_snapshot(self, data: list):
        """Process initial book snapshot."""
        self.bids.clear()
        self.asks.clear()

        for entry in data:
            if len(entry) < 3:
                continue
            price, count, amount = float(entry[0]), int(entry[1]), float(entry[2])
            if amount > 0:
                self.bids[price] = (count, amount)
            elif amount < 0:
                self.asks[price] = (count, abs(amount))

        self._sort()
        self.initialized = True
        self.last_update = time.time()

    def process_update(self, data: list):
        """Process incremental book update."""
        if len(data) < 3:
            return

        price, count, amount = float(data[0]), int(data[1]), float(data[2])

        if count == 0:
            # Delete level
            if amount == 1:  # bid side
                self.bids.pop(price, None)
            elif amount == -1:  # ask side
                self.asks.pop(price, None)
        else:
            # Add/update level
            if amount > 0:
                self.bids[price] = (count, amount)
            elif amount < 0:
                self.asks[price] = (count, abs(amount))

        self._sort()
        self.last_update = time.time()

    def _sort(self):
        """Sort bids descending, asks ascending."""
        self.bids = OrderedDict(sorted(self.bids.items(), key=lambda x: -x[0]))
        self.asks = OrderedDict(sorted(self.asks.items(), key=lambda x: x[0]))

    def get_bbo(self) -> dict | None:
        """Get best bid/offer snapshot with depth metrics."""
        if not self.bids or not self.asks:
            return None

        bid_prices = list(self.bids.keys())
        ask_prices = list(self.asks.keys())

        best_bid = bid_prices[0]
        best_ask = ask_prices[0]

        if best_ask <= best_bid:
            return None  # Crossed book — skip

        bid_count, bid_size = self.bids[best_bid]
        ask_count, ask_size = self.asks[best_ask]

        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000

        # Depth: sum of top N levels
        def level_depth(book: OrderedDict, n: int) -> float:
            total = 0.0
            for i, (price, (count, amount)) in enumerate(book.items()):
                if i >= n:
                    break
                total += amount
            return total

        return {
            'timestamp': datetime.now(timezone.utc),
            'bid_price_1': best_bid,
            'bid_size_1': bid_size,
            'bid_count_1': bid_count,
            'ask_price_1': best_ask,
            'ask_size_1': ask_size,
            'ask_count_1': ask_count,
            'mid_price': mid,
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_depth_5': level_depth(self.bids, 5),
            'ask_depth_5': level_depth(self.asks, 5),
            'bid_depth_25': level_depth(self.bids, 25),
            'ask_depth_25': level_depth(self.asks, 25),
            'n_bid_levels': len(self.bids),
            'n_ask_levels': len(self.asks),
        }


class BitfinexLOBRecorder:
    """WebSocket recorder that collects and persists LOB snapshots."""

    def __init__(self, symbol: str = SYMBOL, depth: int = 25,
                 snapshot_interval: float = 1.0, flush_interval: float = 300.0,
                 duration_hours: float = 24.0):
        self.symbol = symbol
        self.depth = depth
        self.snapshot_interval = snapshot_interval
        self.flush_interval = flush_interval
        self.duration_hours = duration_hours

        self.book = OrderBook()
        self.snapshots: list[dict] = []
        self.channel_id: int | None = None

        self.start_time = time.time()
        self.last_snapshot_time = 0.0
        self.last_flush_time = time.time()
        self.total_updates = 0
        self.total_snapshots = 0
        self.reconnect_count = 0
        self.running = True

        # Stats for live display
        self.spread_values: list[float] = []
        self.stats_window = 300  # 5 min rolling window for display

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = DATA_DIR / f"lob_snapshots_{ts}.parquet"
        self.all_flushed_path = DATA_DIR / f"lob_snapshots_{ts}_all.parquet"

    def on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Event messages (subscribe confirmation, info, etc.)
        if isinstance(data, dict):
            event = data.get('event')
            if event == 'subscribed':
                self.channel_id = data.get('chanId')
                print(f"[WS] Subscribed to book channel {self.channel_id} "
                      f"({data.get('symbol')}, prec={data.get('prec')}, len={data.get('len')})")
            elif event == 'info':
                print(f"[WS] Info: platform status={data.get('platform', {}).get('status')}")
            elif event == 'error':
                print(f"[WS] ERROR: {data.get('msg')} (code={data.get('code')})")
            return

        # Data messages: [CHANNEL_ID, DATA]
        if not isinstance(data, list) or len(data) < 2:
            return

        chan_id = data[0]
        if chan_id != self.channel_id:
            return

        payload = data[1]

        # Heartbeat
        if payload == 'hb':
            return

        # Snapshot: [[PRICE, COUNT, AMOUNT], [PRICE, COUNT, AMOUNT], ...]
        if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], list):
            self.book.process_snapshot(payload)
            self.total_updates += len(payload)
            return

        # Incremental update: [PRICE, COUNT, AMOUNT]
        if isinstance(payload, list) and len(payload) >= 3:
            self.book.process_update(payload)
            self.total_updates += 1

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"[WS] Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"[WS] Connection closed (status={close_status_code}, msg={close_msg})")
        if self.running:
            self.reconnect_count += 1
            print(f"[WS] Will reconnect (attempt #{self.reconnect_count})...")

    def on_open(self, ws):
        """Subscribe to book channel on connection."""
        subscribe_msg = json.dumps({
            "event": "subscribe",
            "channel": "book",
            "symbol": self.symbol,
            "prec": "P0",       # Tightest aggregation (5 sig figs)
            "freq": "F0",       # Real-time updates
            "len": str(self.depth),
        })
        ws.send(subscribe_msg)
        print(f"[WS] Connected. Subscribing to {self.symbol} book "
              f"(P0, F0, depth={self.depth})...")

    def snapshot_loop(self):
        """Background thread: record BBO snapshots at regular intervals."""
        while self.running:
            now = time.time()

            # Check duration
            elapsed_hours = (now - self.start_time) / 3600
            if elapsed_hours >= self.duration_hours:
                print(f"\n[REC] Duration reached ({self.duration_hours}h). Stopping...")
                self.running = False
                break

            # Record snapshot
            if self.book.initialized and (now - self.last_snapshot_time) >= self.snapshot_interval:
                bbo = self.book.get_bbo()
                if bbo is not None:
                    self.snapshots.append(bbo)
                    self.total_snapshots += 1
                    self.spread_values.append(bbo['spread_bps'])

                    # Trim rolling window for display
                    max_window = int(self.stats_window / self.snapshot_interval)
                    if len(self.spread_values) > max_window:
                        self.spread_values = self.spread_values[-max_window:]

                self.last_snapshot_time = now

            # Periodic flush to disk
            if (now - self.last_flush_time) >= self.flush_interval and self.snapshots:
                self._flush()

            # Live stats display (every 10 seconds)
            if self.total_snapshots > 0 and self.total_snapshots % max(1, int(10 / self.snapshot_interval)) == 0:
                self._print_stats()

            time.sleep(self.snapshot_interval * 0.9)  # Slight undersleep to stay on cadence

    def _flush(self):
        """Flush accumulated snapshots to parquet."""
        if not self.snapshots:
            return

        df = pd.DataFrame(self.snapshots)
        # Append to existing file
        if self.all_flushed_path.exists():
            existing = pd.read_parquet(self.all_flushed_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(self.all_flushed_path, index=False, engine='pyarrow')
        n_flushed = len(self.snapshots)
        self.snapshots = []
        self.last_flush_time = time.time()
        elapsed = (time.time() - self.start_time) / 3600
        print(f"[FLUSH] {n_flushed} snapshots → {self.all_flushed_path.name} "
              f"(total={len(df):,}, {elapsed:.1f}h elapsed)")

    def _print_stats(self):
        """Print live spread statistics."""
        if not self.spread_values:
            return

        arr = np.array(self.spread_values)
        elapsed = (time.time() - self.start_time) / 3600
        remaining = max(0, self.duration_hours - elapsed)

        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        print(f"[{now_str}] {elapsed:.2f}h/{self.duration_hours}h | "
              f"snaps={self.total_snapshots:,} | updates={self.total_updates:,} | "
              f"spread: mean={arr.mean():.3f} med={np.median(arr):.3f} "
              f"p95={np.percentile(arr, 95):.3f} p99={np.percentile(arr, 99):.3f} bps | "
              f"bid_depth={self.snapshots[-1]['bid_depth_5']:.3f} "
              f"ask_depth={self.snapshots[-1]['ask_depth_5']:.3f} BTC" if self.snapshots else "")

    def run(self):
        """Main entry: connect, record, reconnect on failure."""
        print(f"[REC] Bitfinex LOB Recorder")
        print(f"[REC] Symbol: {self.symbol}")
        print(f"[REC] Duration: {self.duration_hours}h")
        print(f"[REC] Snapshot interval: {self.snapshot_interval}s")
        print(f"[REC] Flush interval: {self.flush_interval}s")
        print(f"[REC] Output: {self.all_flushed_path}")
        print(f"[REC] Press Ctrl+C to stop gracefully")
        print()

        # Start snapshot thread
        snap_thread = threading.Thread(target=self.snapshot_loop, daemon=True)
        snap_thread.start()

        # WebSocket loop with auto-reconnect
        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    WSS_URL,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)

                if self.running:
                    backoff = min(30, 2 ** self.reconnect_count)
                    print(f"[WS] Reconnecting in {backoff}s...")
                    time.sleep(backoff)

            except KeyboardInterrupt:
                print("\n[REC] Ctrl+C received. Shutting down...")
                self.running = False
                break
            except Exception as e:
                print(f"[WS] Unexpected error: {e}")
                if self.running:
                    time.sleep(5)

        # Final flush
        self._flush()
        self._print_final_report()

    def _print_final_report(self):
        """Print summary of recorded data."""
        elapsed = (time.time() - self.start_time) / 3600

        print(f"\n{'='*70}")
        print(f"RECORDING COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration: {elapsed:.2f} hours")
        print(f"  Total snapshots: {self.total_snapshots:,}")
        print(f"  Total book updates: {self.total_updates:,}")
        print(f"  Reconnections: {self.reconnect_count}")
        print(f"  Output: {self.all_flushed_path}")

        # Load and analyze
        if self.all_flushed_path.exists():
            df = pd.read_parquet(self.all_flushed_path)
            if len(df) > 0 and 'spread_bps' in df.columns:
                s = df['spread_bps']
                print(f"\n  Spread Statistics ({len(df):,} observations):")
                print(f"    Mean:    {s.mean():.4f} bps")
                print(f"    Median:  {s.median():.4f} bps")
                print(f"    Std:     {s.std():.4f} bps")
                print(f"    P5:      {s.quantile(0.05):.4f} bps")
                print(f"    P25:     {s.quantile(0.25):.4f} bps")
                print(f"    P75:     {s.quantile(0.75):.4f} bps")
                print(f"    P95:     {s.quantile(0.95):.4f} bps")
                print(f"    P99:     {s.quantile(0.99):.4f} bps")
                print(f"    Min:     {s.min():.4f} bps")
                print(f"    Max:     {s.max():.4f} bps")

                # Viability verdict
                median_spread = s.median()
                if median_spread < 0.5:
                    print(f"\n  VERDICT: SPREAD < 0.5 bps → BTC Bitfinex VIABLE (strong)")
                elif median_spread < 1.0:
                    print(f"\n  VERDICT: SPREAD < 1.0 bps → BTC Bitfinex MARGINAL")
                elif median_spread < 2.0:
                    print(f"\n  VERDICT: SPREAD < 2.0 bps → BTC Bitfinex TIGHT (needs LOB features)")
                else:
                    print(f"\n  VERDICT: SPREAD >= 2.0 bps → BTC Bitfinex FAIL (hidden cost too high)")

        print(f"\n  Next: python scripts/analyze_bitfinex_spread.py")
        print(f"        (detailed analysis with hourly breakdown)")


def main():
    parser = argparse.ArgumentParser(description="Record Bitfinex LOB snapshots")
    parser.add_argument("--symbol", default=SYMBOL, help="Bitfinex symbol")
    parser.add_argument("--depth", type=int, default=25,
                        help="Book depth (levels per side)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Snapshot interval in seconds")
    parser.add_argument("--flush-interval", type=float, default=300.0,
                        help="Flush to disk interval in seconds")
    parser.add_argument("--duration", type=float, default=24.0,
                        help="Recording duration in hours")
    args = parser.parse_args()

    recorder = BitfinexLOBRecorder(
        symbol=args.symbol,
        depth=args.depth,
        snapshot_interval=args.interval,
        flush_interval=args.flush_interval,
        duration_hours=args.duration,
    )

    # Handle signals for graceful shutdown
    def handle_signal(signum, frame):
        print(f"\n[REC] Signal {signum} received. Stopping...")
        recorder.running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    recorder.run()


if __name__ == "__main__":
    main()
