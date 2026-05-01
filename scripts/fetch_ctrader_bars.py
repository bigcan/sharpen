#!/usr/bin/env python3
"""Standalone cTrader trendbar fetcher.

Authenticates against cTrader Open API using credentials from docker/live/.env
and fetches 1-min OHLCV trendbars for XAUUSD over a date range.

Designed for the SG-1 XAUUSD drift diagnostic (audit option (f) — comparing
live cTrader bars to OANDA over the overlap window 2026-04-22 → 2026-04-30).

Concurrent with gmgp1-xauusd live container is OK: cTrader OAuth allows
multiple sessions per access token.
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ctrader-fetch")

_TRENDBAR_PRICE_SCALE = 100_000


def _load_env(env_path: Path) -> dict:
    out = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def fetch_bars(account_id: int, symbol: str, start: datetime, end: datetime,
               client_id: str, client_secret: str, access_token: str,
               host: str = "demo.ctraderapi.com", port: int = 5035) -> pd.DataFrame:
    """Drive a Twisted reactor to completion fetching trendbars; return DataFrame."""
    from ctrader_open_api import Client, Protobuf, TcpProtocol
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq,
        ProtoOASymbolsListReq, ProtoOAGetTrendbarsReq,
    )
    from twisted.internet import reactor

    client = Client(host, port, TcpProtocol)
    bars: list[dict] = []
    state = {"symbol_id": None, "remaining_chunks": []}

    def on_connected(client_):
        log.info("TCP connected; sending app auth")
        req = ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        client_.send(req)

    def on_disconnected(client_, reason):
        log.warning(f"disconnected: {reason}")

    def on_message(client_, message):
        try:
            payload = Protobuf.extract(message)
        except Exception as e:  # noqa: BLE001
            log.warning(f"decode failed: {e}")
            return
        cls = type(payload).__name__
        if cls == "ProtoOAApplicationAuthRes":
            log.info("app authenticated")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = account_id
            req.accessToken = access_token
            client_.send(req)
        elif cls == "ProtoOAAccountAuthRes":
            log.info("account authenticated")
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = account_id
            client_.send(req)
        elif cls == "ProtoOASymbolsListRes":
            for s in payload.symbol:
                if s.symbolName == symbol:
                    state["symbol_id"] = s.symbolId
                    log.info(f"{symbol} id={s.symbolId}")
                    break
            if state["symbol_id"] is None:
                log.error(f"{symbol} not found in symbol list")
                reactor.callLater(0.1, reactor.stop)
                return
            cur = start
            while cur < end:
                ce = min(cur + timedelta(days=3), end)
                state["remaining_chunks"].append((cur, ce))
                cur = ce
            log.info(f"queued {len(state['remaining_chunks'])} chunks")
            _send_next(client_)
        elif cls == "ProtoOAGetTrendbarsRes":
            for bar in payload.trendbar:
                low = bar.low / _TRENDBAR_PRICE_SCALE
                ts = datetime.fromtimestamp(bar.utcTimestampInMinutes * 60,
                                            tz=timezone.utc)
                bars.append({
                    "timestamp": ts, "ticker": symbol,
                    "open": round(low + bar.deltaOpen / _TRENDBAR_PRICE_SCALE, 2),
                    "high": round(low + bar.deltaHigh / _TRENDBAR_PRICE_SCALE, 2),
                    "low": round(low, 2),
                    "close": round(low + bar.deltaClose / _TRENDBAR_PRICE_SCALE, 2),
                    "volume": bar.volume,
                })
            log.info(f"+ {len(payload.trendbar)} bars  total={len(bars)}")
            _send_next(client_)
        elif cls == "ProtoOAErrorRes":
            log.error(f"cTrader error: {payload.errorCode} {payload.description}")
            reactor.callLater(0.1, reactor.stop)

    def _send_next(client_):
        if not state["remaining_chunks"]:
            log.info("all chunks done")
            reactor.callLater(0.5, reactor.stop)
            return
        c_start, c_end = state["remaining_chunks"].pop(0)
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = account_id
        req.symbolId = state["symbol_id"]
        req.period = 1  # M1
        req.fromTimestamp = int(c_start.timestamp() * 1000)
        req.toTimestamp = int(c_end.timestamp() * 1000)
        log.info(f"req {c_start} -> {c_end}")
        client_.send(req)

    client.setConnectedCallback(on_connected)
    client.setMessageReceivedCallback(on_message)
    client.setDisconnectedCallback(on_disconnected)
    client.startService()
    reactor.callLater(300, lambda: reactor.stop() if reactor.running else None)
    reactor.run(installSignalHandlers=False)
    return pd.DataFrame(bars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-21")
    ap.add_argument("--end", default="2026-04-30")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--account_env", default="CTRADER_ACCOUNT_ID")
    ap.add_argument("--out", default="data/ctrader_demo/xauusd_m1_apr2026.parquet")
    ap.add_argument("--env_file", default="docker/live/.env")
    args = ap.parse_args()

    env = _load_env(ROOT / args.env_file)
    client_id = env.get("CTRADER_CLIENT_ID", "")
    client_secret = env.get("CTRADER_CLIENT_SECRET", "")
    access_token = env.get("CTRADER_ACCESS_TOKEN", "")
    account_id = int(env.get(args.account_env, "0"))
    if not (client_id and client_secret and access_token and account_id):
        log.error("missing creds; check docker/live/.env")
        sys.exit(1)

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    log.info(f"fetching {args.symbol} 1m {start_dt} -> {end_dt} for account {account_id}")
    df = fetch_bars(account_id, args.symbol, start_dt, end_dt,
                    client_id, client_secret, access_token)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"saved {len(df)} bars to {out}")
    if len(df):
        log.info(f"first ts {df['timestamp'].min()}  last ts {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
