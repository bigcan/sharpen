"""
Hyperliquid WebSocket LOB Recorder — Collect live order book snapshots.
======================================================================

Connects to Hyperliquid public WS (no auth).
Subscribes to `l2Book` (full-depth snapshots) + `trades` for a given coin.
Persists 1-per-second BBO + 10-level-depth snapshots to parquet (daily-rotated).

Purpose: Phase 1 data collection for MM venue reframe validation.
Gate: A-S baseline @ 1.5 bps maker cost, PF >= 1.05 required (else MM closed).

Output: data/hyperliquid/<coin>/lob_YYYYMMDD.parquet  (one file per UTC day)
  Columns: timestamp_ms, coin, bid_price_1, bid_size_1, bid_count_1,
           ask_price_1, ask_size_1, ask_count_1, mid_price, spread, spread_bps,
           bid_depth_5, ask_depth_5, bid_depth_10, ask_depth_10,
           bid_prices (JSON list, 10 levels), bid_quantities (JSON list),
           ask_prices, ask_quantities, n_bid_levels, n_ask_levels

Trades (separate file): data/hyperliquid/<coin>/trades_YYYYMMDD.parquet
  Columns: timestamp_ms, coin, side, px, sz, tid, users (JSON list)

Usage:
  python scripts/record_hyperliquid_lob.py --coin BTC --duration-hours 168
  python scripts/record_hyperliquid_lob.py --coin BTC --duration-hours 0.1  # smoke

Hyperliquid WS protocol:
  URL: wss://api.hyperliquid.xyz/ws
  Subscribe: {"method":"subscribe","subscription":{"type":"l2Book","coin":"BTC"}}
  Message:   {"channel":"l2Book","data":{"coin":"BTC","time":...,
              "levels":[[{"px":"...","sz":"...","n":...}], [asks]]}}
  Trades:    {"channel":"trades","data":[{"coin":"BTC","side":"B"|"A",
              "px":"...","sz":"...","time":...,"tid":...,"users":[...]}]}
  Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websockets

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

WSS_URL = "wss://api.hyperliquid.xyz/ws"
DATA_DIR = Path(__file__).parent.parent / "data" / "hyperliquid"


class HLRecorder:
    def __init__(self, coin: str, duration_hours: float,
                 snapshot_interval: float = 1.0,
                 flush_interval: float = 300.0,
                 depth: int = 10):
        self.coin = coin
        self.duration_hours = duration_hours
        self.snapshot_interval = snapshot_interval
        self.flush_interval = flush_interval
        self.depth = depth

        self.latest_book: dict | None = None
        self.snapshots: list[dict] = []
        self.trade_buffer: list[dict] = []

        self.start_time = 0.0
        self.last_snapshot_time = 0.0
        self.last_flush_time = 0.0
        self.last_stats_time = 0.0
        self.total_snapshots = 0
        self.total_book_msgs = 0
        self.total_trades = 0
        self.reconnect_count = 0
        self.running = True

        out_root = DATA_DIR / coin.lower()
        out_root.mkdir(parents=True, exist_ok=True)
        self.out_root = out_root

    def _current_lob_path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.out_root / f"lob_{day}.parquet"

    def _current_trades_path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.out_root / f"trades_{day}.parquet"

    def _handle_l2book(self, data: dict) -> None:
        # data = {"coin": "BTC", "time": int_ms, "levels": [[bids], [asks]]}
        try:
            levels = data.get("levels") or []
            if len(levels) < 2:
                return
            bids_raw, asks_raw = levels[0], levels[1]
            if not bids_raw or not asks_raw:
                return

            bids = [(float(lvl["px"]), float(lvl["sz"]), int(lvl.get("n", 0)))
                    for lvl in bids_raw[:self.depth]]
            asks = [(float(lvl["px"]), float(lvl["sz"]), int(lvl.get("n", 0)))
                    for lvl in asks_raw[:self.depth]]

            self.latest_book = {
                "exchange_time_ms": int(data.get("time", 0)),
                "bids": bids,
                "asks": asks,
            }
            self.total_book_msgs += 1
        except (KeyError, ValueError, TypeError) as e:
            print(f"[WS] Malformed l2Book: {e}")

    def _handle_trades(self, data: list) -> None:
        # data = list of {"coin","side","px","sz","time","tid","users"}
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        for t in data or []:
            try:
                self.trade_buffer.append({
                    "timestamp_ms": now_ms,
                    "exchange_time_ms": int(t.get("time", 0)),
                    "coin": t.get("coin", self.coin),
                    "side": t.get("side", ""),
                    "px": float(t["px"]),
                    "sz": float(t["sz"]),
                    "tid": int(t.get("tid", 0)),
                    "users": json.dumps(t.get("users", [])),
                })
                self.total_trades += 1
            except (KeyError, ValueError, TypeError) as e:
                print(f"[WS] Malformed trade: {e}")

    def _build_snapshot(self) -> dict | None:
        if not self.latest_book:
            return None
        bids = self.latest_book["bids"]
        asks = self.latest_book["asks"]
        if not bids or not asks:
            return None

        best_bid_px, best_bid_sz, best_bid_n = bids[0]
        best_ask_px, best_ask_sz, best_ask_n = asks[0]
        if best_ask_px <= best_bid_px:
            return None  # Crossed — skip

        mid = (best_bid_px + best_ask_px) / 2
        spread = best_ask_px - best_bid_px
        spread_bps = (spread / best_bid_px) * 10000

        bid_depth_5 = sum(sz for _, sz, _ in bids[:5])
        ask_depth_5 = sum(sz for _, sz, _ in asks[:5])
        bid_depth_10 = sum(sz for _, sz, _ in bids[:10])
        ask_depth_10 = sum(sz for _, sz, _ in asks[:10])

        now = datetime.now(timezone.utc)
        return {
            "timestamp_ms": int(now.timestamp() * 1000),
            "exchange_time_ms": self.latest_book["exchange_time_ms"],
            "coin": self.coin,
            "bid_price_1": best_bid_px,
            "bid_size_1": best_bid_sz,
            "bid_count_1": best_bid_n,
            "ask_price_1": best_ask_px,
            "ask_size_1": best_ask_sz,
            "ask_count_1": best_ask_n,
            "mid_price": mid,
            "spread": spread,
            "spread_bps": spread_bps,
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "bid_depth_10": bid_depth_10,
            "ask_depth_10": ask_depth_10,
            "bid_prices": json.dumps([p for p, _, _ in bids]),
            "bid_quantities": json.dumps([s for _, s, _ in bids]),
            "ask_prices": json.dumps([p for p, _, _ in asks]),
            "ask_quantities": json.dumps([s for _, s, _ in asks]),
            "n_bid_levels": len(bids),
            "n_ask_levels": len(asks),
        }

    def _flush(self) -> None:
        if self.snapshots:
            path = self._current_lob_path()
            df = pd.DataFrame(self.snapshots)
            if path.exists():
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            df.to_parquet(path, index=False, engine="pyarrow")
            n = len(self.snapshots)
            self.snapshots = []
            elapsed = (asyncio.get_event_loop().time() - self.start_time) / 3600
            print(f"[FLUSH] {n} snaps -> {path.name} "
                  f"(file_total={len(df):,}, {elapsed:.2f}h elapsed)")

        if self.trade_buffer:
            path = self._current_trades_path()
            df = pd.DataFrame(self.trade_buffer)
            if path.exists():
                df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            df.to_parquet(path, index=False, engine="pyarrow")
            n = len(self.trade_buffer)
            self.trade_buffer = []
            print(f"[FLUSH] {n} trades -> {path.name} (file_total={len(df):,})")

    async def _snapshot_loop(self, loop_start: float) -> None:
        while self.running:
            now = asyncio.get_event_loop().time()

            elapsed_h = (now - loop_start) / 3600
            if elapsed_h >= self.duration_hours:
                print(f"\n[REC] Duration reached ({self.duration_hours}h). Stopping.")
                self.running = False
                break

            if (now - self.last_snapshot_time) >= self.snapshot_interval:
                snap = self._build_snapshot()
                if snap:
                    self.snapshots.append(snap)
                    self.total_snapshots += 1
                self.last_snapshot_time = now

            if (now - self.last_flush_time) >= self.flush_interval:
                self._flush()
                self.last_flush_time = now

            if (now - self.last_stats_time) >= 30.0:
                self._print_stats(loop_start)
                self.last_stats_time = now

            await asyncio.sleep(self.snapshot_interval * 0.5)

    def _print_stats(self, loop_start: float) -> None:
        elapsed = (asyncio.get_event_loop().time() - loop_start) / 3600
        remaining = max(0, self.duration_hours - elapsed)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if self.latest_book:
            snap = self._build_snapshot()
            spread_bps = snap["spread_bps"] if snap else float("nan")
            mid = snap["mid_price"] if snap else float("nan")
        else:
            spread_bps = float("nan")
            mid = float("nan")
        print(f"[{now_str}] {self.coin} | {elapsed:.2f}/{self.duration_hours}h "
              f"(remain {remaining:.2f}h) | snaps={self.total_snapshots:,} "
              f"books={self.total_book_msgs:,} trades={self.total_trades:,} "
              f"reconnects={self.reconnect_count} | "
              f"mid={mid:.2f} spread={spread_bps:.3f} bps")

    async def _ws_loop(self) -> None:
        while self.running:
            try:
                async with websockets.connect(
                    WSS_URL, ping_interval=30, ping_timeout=20,
                    max_size=2**22,  # 4 MB, books can be big
                    close_timeout=5,
                ) as ws:
                    print(f"[WS] Connected. Subscribing to l2Book + trades ({self.coin})")
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "l2Book", "coin": self.coin},
                    }))
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "trades", "coin": self.coin},
                    }))

                    async for raw in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        ch = msg.get("channel")
                        data = msg.get("data")
                        if ch == "l2Book":
                            self._handle_l2book(data)
                        elif ch == "trades":
                            self._handle_trades(data)
                        elif ch == "subscriptionResponse":
                            print(f"[WS] Sub confirmed: {data}")
                        elif ch == "error":
                            print(f"[WS] Error msg: {data}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.running:
                    break
                self.reconnect_count += 1
                backoff = min(30, 2 ** min(self.reconnect_count, 5))
                print(f"[WS] Disconnected ({type(e).__name__}: {e}). "
                      f"Reconnect #{self.reconnect_count} in {backoff}s...")
                await asyncio.sleep(backoff)

    async def run(self) -> None:
        print("[REC] Hyperliquid LOB Recorder")
        print(f"[REC] Coin: {self.coin} | Duration: {self.duration_hours}h "
              f"| Snap: {self.snapshot_interval}s | Flush: {self.flush_interval}s")
        print(f"[REC] Output: {self.out_root}")

        loop_start = asyncio.get_event_loop().time()
        self.start_time = loop_start
        self.last_flush_time = loop_start
        self.last_stats_time = loop_start

        ws_task = asyncio.create_task(self._ws_loop())
        snap_task = asyncio.create_task(self._snapshot_loop(loop_start))

        done, pending = await asyncio.wait(
            [ws_task, snap_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        self.running = False
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        self._flush()
        self._print_final()

    def _print_final(self) -> None:
        print(f"\n{'=' * 70}\nRECORDING COMPLETE ({self.coin})\n{'=' * 70}")
        print(f"  Snapshots: {self.total_snapshots:,}")
        print(f"  Book msgs: {self.total_book_msgs:,}")
        print(f"  Trades:    {self.total_trades:,}")
        print(f"  Reconnects: {self.reconnect_count}")
        files = sorted(self.out_root.glob("lob_*.parquet"))
        if files:
            total_rows = 0
            for f in files:
                try:
                    total_rows += len(pd.read_parquet(f, columns=["timestamp_ms"]))
                except Exception:
                    pass
            print(f"  Files: {len(files)} daily files, {total_rows:,} total rows")
            # Spread summary across all recorded data
            try:
                dfs = [pd.read_parquet(f, columns=["spread_bps"]) for f in files]
                s = pd.concat(dfs, ignore_index=True)["spread_bps"]
                print(f"\n  Spread (bps): mean={s.mean():.3f} median={s.median():.3f} "
                      f"p5={s.quantile(0.05):.3f} p95={s.quantile(0.95):.3f} "
                      f"p99={s.quantile(0.99):.3f}")
                med = s.median()
                if med >= 1.0:
                    verdict = f"NOT tick-pinned (median {med:.2f} bps) — A-S baseline justified"
                else:
                    verdict = f"tight median {med:.2f} bps — may be tick-pinned on this pair"
                print(f"  Verdict: {verdict}")
            except Exception as e:
                print(f"  (spread summary skipped: {e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Hyperliquid LOB + trades")
    parser.add_argument("--coin", default="BTC", help="Coin symbol (BTC, ETH, SOL, ...)")
    parser.add_argument("--duration-hours", type=float, default=168.0,
                        help="Recording duration in hours (default 168 = 7 days)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Snapshot sampling interval in seconds")
    parser.add_argument("--flush-interval", type=float, default=300.0,
                        help="Parquet flush interval in seconds")
    parser.add_argument("--depth", type=int, default=10,
                        help="Book depth per side (Hyperliquid serves up to 20)")
    args = parser.parse_args()

    recorder = HLRecorder(
        coin=args.coin,
        duration_hours=args.duration_hours,
        snapshot_interval=args.interval,
        flush_interval=args.flush_interval,
        depth=args.depth,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(signum, _frame):
        print(f"\n[REC] Signal {signum} received. Stopping...")
        recorder.running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        loop.run_until_complete(recorder.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
