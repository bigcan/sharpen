"""Forward collector: Polymarket 5-min up/down L2 book + resolution-relevant price streams.

Stage-B prerequisite for the pre-registered pm-updown-mm-v0 paper test
(configs/polymarket_updown_mm_paper.gates.yaml). Historical order-book depth for
Polymarket does not exist anywhere (official endpoint dead since 2026-02-20), so
the shadow-quoting fill simulation can only run on data collected FORWARD from
the day this starts. Both feeds are public and unauthenticated (read-only market
data; no wallet, no orders).

Collects, as hourly-rotated gzip JSONL under --out:
  clob_YYYYMMDD_HH.jsonl.gz    CLOB market channel (book snapshots, price_change
                               deltas, last_trade_price, tick_size_change) for the
                               current + next N 5-min windows per asset
  rtds_YYYYMMDD_HH.jsonl.gz    RTDS crypto prices: Chainlink stream (the exact
                               resolution source) + Binance spot
  markets_YYYYMMDD.jsonl       catalog of discovered windows (slug, condition_id,
                               token ids, start/end) for later interpretation

Each JSONL line: {"t": <local recv epoch ms>, "ch": <channel>, "msg": <payload>}.

Market discovery uses deterministic slugs ({asset}-updown-5m-{window_start_epoch},
epoch 300-aligned; verified against markets.parquet: end_date - slug_epoch = 300s
for 99.3% of the family) resolved through the public Gamma API. The CLOB
websocket takes its asset list at connect time, so the collector reconnects
whenever the tracked window set rolls (every 5 minutes) — each reconnect also
yields a fresh full book snapshot, which the downstream sim wants anyway.

Usage:
    python scripts/data/collect_polymarket_updown_l2.py --selftest
    python scripts/data/collect_polymarket_updown_l2.py \
        --assets btc,eth --out data/polymarket_updown/l2
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import time
from pathlib import Path

import aiohttp
import websockets

log = logging.getLogger("pm_l2_collector")

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
WINDOW_S = 300

BINANCE_SYMBOLS = {"btc": "btcusdt", "eth": "ethusdt", "sol": "solusdt", "xrp": "xrpusdt"}
CHAINLINK_SYMBOLS = {"btc": "btc/usd", "eth": "eth/usd", "sol": "sol/usd", "xrp": "xrp/usd"}


class HourlyGzipWriter:
    """Append JSONL lines to gzip files rotated on the wall-clock hour."""

    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir = out_dir
        self.prefix = prefix
        self._fh: gzip.GzipFile | None = None
        self._hour_key = ""

    def write(self, channel: str, msg: object) -> None:
        now_ms = int(time.time() * 1000)
        hour_key = time.strftime("%Y%m%d_%H", time.gmtime(now_ms / 1000))
        if hour_key != self._hour_key:
            self.close()
            path = self.out_dir / f"{self.prefix}_{hour_key}.jsonl.gz"
            self._fh = gzip.open(path, "at", encoding="utf-8")
            self._hour_key = hour_key
            log.info("writer %s -> %s", self.prefix, path.name)
        line = json.dumps({"t": now_ms, "ch": channel, "msg": msg}, separators=(",", ":"))
        assert self._fh is not None
        self._fh.write(line + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


async def discover_windows(
    session: aiohttp.ClientSession, assets: list[str], n_ahead: int, catalog: Path
) -> dict[str, dict]:
    """Resolve current + next n_ahead windows per asset via deterministic slugs.

    Returns {token_id: market_info}; appends newly seen markets to the catalog file.
    """
    now = int(time.time())
    cur_start = now // WINDOW_S * WINDOW_S
    tokens: dict[str, dict] = {}
    seen: set[str] = set()
    if catalog.exists():
        with open(catalog, encoding="utf-8") as fh:
            seen = {json.loads(ln)["slug"] for ln in fh if ln.strip()}
    for asset in assets:
        for k in range(n_ahead + 1):
            slug = f"{asset}-updown-5m-{cur_start + k * WINDOW_S}"
            try:
                async with session.get(
                    GAMMA_MARKETS_URL, params={"slug": slug}, timeout=aiohttp.ClientTimeout(10)
                ) as r:
                    if r.status != 200:
                        log.warning("gamma %s -> HTTP %d", slug, r.status)
                        continue
                    payload = await r.json()
            except Exception as e:  # noqa: BLE001 - network fetch, log and continue
                log.warning("gamma %s failed: %s", slug, type(e).__name__)
                continue
            if not payload:
                log.warning("gamma: no market for %s (not created yet?)", slug)
                continue
            mkt = payload[0]
            try:
                token_ids = json.loads(mkt["clobTokenIds"])
            except (KeyError, json.JSONDecodeError):
                log.warning("gamma %s: missing/bad clobTokenIds", slug)
                continue
            info = {
                "slug": slug,
                "asset": asset,
                "window_start": cur_start + k * WINDOW_S,
                "window_end": cur_start + (k + 1) * WINDOW_S,
                "condition_id": mkt.get("conditionId"),
                "token_ids": token_ids,
            }
            for tid in token_ids:
                tokens[tid] = info
            if slug not in seen:
                with open(catalog, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(info, separators=(",", ":")) + "\n")
    return tokens


async def clob_loop(
    assets: list[str], n_ahead: int, writer: HourlyGzipWriter, catalog: Path, stats: dict
) -> None:
    """Subscribe to the CLOB market channel for tracked windows; roll every 5 min."""
    # Force gzip/deflate only: some remote envs ship a brotli decompressor whose .process()
    # signature doesn't match what this aiohttp version calls (TypeError inside aiohttp's own
    # br handling -> surfaces as ClientPayloadError on every request). gzip is universally
    # supported server-side (confirmed via `Vary: Accept-Encoding`) and sidesteps the whole
    # class of bug regardless of what compression libs happen to be installed on the host.
    async with aiohttp.ClientSession(headers={"Accept-Encoding": "gzip, deflate"}) as session:
        backoff = 1.0
        while True:
            try:
                tokens = await discover_windows(session, assets, n_ahead, catalog)
                if not tokens:
                    log.warning("clob: no tokens discovered, retrying in 10s")
                    await asyncio.sleep(10)
                    continue
                # reconnect at the next window roll for a fresh set + snapshot
                roll_at = (int(time.time()) // WINDOW_S + 1) * WINDOW_S + 2
                sub = {"assets_ids": list(tokens), "type": "market"}
                async with websockets.connect(CLOB_WS_URL, ping_interval=10) as ws:
                    await ws.send(json.dumps(sub))
                    log.info("clob: subscribed %d tokens (%d windows)",
                             len(tokens), len(tokens) // 2)
                    backoff = 1.0
                    while time.time() < roll_at:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except TimeoutError:
                            await ws.send("PING")  # app-level keepalive
                            continue
                        if raw == "PONG":
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            msg = {"unparsed": str(raw)[:2000]}
                        writer.write("clob", msg)
                        stats["clob"] += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - reconnect loop
                log.warning("clob loop error: %s (reconnect in %.0fs)", type(e).__name__, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)


async def rtds_loop(assets: list[str], writer: HourlyGzipWriter, stats: dict) -> None:
    """Subscribe to RTDS Chainlink (resolution source) + Binance spot prices.

    NOTE: we deliberately subscribe WITHOUT a ``filters`` field. Empirically (probe
    2026-07-04 against wss://ws-live-data.polymarket.com), supplying ``filters`` on the
    crypto_prices / crypto_prices_chainlink topics flips the server into a one-shot
    HISTORICAL-snapshot mode -- it returns a single ``{"payload":{"data":[...]}}`` batch
    and then streams nothing (the "rtds stuck at 3 messages" bug). Subscribing with no
    filter yields the LIVE per-second update stream for every symbol (~1 msg/symbol/s on
    each topic). We collect all symbols (tiny vs CLOB volume); downstream filters on the
    per-message ``payload.symbol`` field. ``assets`` is retained for signature parity only.
    """
    _ = assets  # all symbols streamed; server-side filtering is broken (see docstring)
    subs = [
        {"topic": "crypto_prices", "type": "update"},
        {"topic": "crypto_prices_chainlink", "type": "*"},
    ]
    payload = {"action": "subscribe", "subscriptions": subs}
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(RTDS_WS_URL, ping_interval=10) as ws:
                await ws.send(json.dumps(payload))
                log.info("rtds: subscribed crypto_prices + crypto_prices_chainlink (all symbols, no filter)")
                backoff = 1.0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        msg = {"unparsed": str(raw)[:2000]}
                    writer.write("rtds", msg)
                    stats["rtds"] += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - reconnect loop
            log.warning("rtds loop error: %s (reconnect in %.0fs)", type(e).__name__, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    catalog = out / f"markets_{time.strftime('%Y%m%d', time.gmtime())}.jsonl"
    clob_writer = HourlyGzipWriter(out, "clob")
    rtds_writer = HourlyGzipWriter(out, "rtds")
    stats = {"clob": 0, "rtds": 0}
    tasks = [
        asyncio.create_task(clob_loop(assets, args.windows_ahead, clob_writer, catalog, stats)),
        asyncio.create_task(rtds_loop(assets, rtds_writer, stats)),
    ]
    try:
        if args.selftest:
            await asyncio.sleep(args.selftest_seconds)
        else:
            while True:
                await asyncio.sleep(600)
                log.info("heartbeat: clob=%d rtds=%d msgs", stats["clob"], stats["rtds"])
    finally:
        for t in tasks:
            t.cancel()
        # The inner clob_loop receive-wait (asyncio.wait_for(ws.recv(), timeout=5.0) in a
        # tight while-loop) does not reliably honor an external task.cancel() every
        # iteration -- observed empirically to keep the process alive indefinitely past
        # --selftest-seconds (verified: one run over 12h, one run 11+ min, both required a
        # manual kill). Bound the shutdown wait itself so the process always exits within a
        # known time regardless of that cancellation-propagation subtlety.
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
        except TimeoutError:
            log.warning("shutdown: tasks did not honor cancellation within 10s, forcing exit")
        clob_writer.close()
        rtds_writer.close()
    if args.selftest:
        ok = stats["clob"] >= 1 and stats["rtds"] >= 1
        log.info("selftest: clob=%d rtds=%d -> %s", stats["clob"], stats["rtds"],
                 "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="btc,eth")
    ap.add_argument("--out", default="data/polymarket_updown/l2")
    ap.add_argument("--windows-ahead", type=int, default=2)
    ap.add_argument("--selftest", action="store_true",
                    help="connect both feeds, require >=1 message each, then exit")
    ap.add_argument("--selftest-seconds", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
