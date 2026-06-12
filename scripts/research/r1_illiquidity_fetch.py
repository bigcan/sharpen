"""R1-illiquidity probe — data fetcher (crypto + OANDA branches).

Spec: docs/research/r1_illiquidity_probe_spec_2026-06-12.md (pre-registered).

Fetches 1-min OHLCV for the 23-asset liquidity-spectrum basket into
data/r1_illiquidity/. Resumable: appends from the last stored timestamp.
Unfetchable instruments are recorded as UNTESTABLE in fetch_manifest.json
(no silent drops — spec Gate D / no-silent-caps rule).

Usage:
  python scripts/research/r1_illiquidity_fetch.py --branch crypto
  python scripts/research/r1_illiquidity_fetch.py --branch oanda
  python scripts/research/r1_illiquidity_fetch.py --branch crypto --symbols BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "r1_illiquidity"
# One manifest per branch — the two branches run as concurrent processes and
# must not read-modify-write the same file.
MANIFESTS = {b: OUT_DIR / f"fetch_manifest_{b}.json" for b in ("crypto", "oanda")}

START = "2024-09-01"
END = "2026-06-01"

CRYPTO_TIERS = {
    "T0": ["BTCUSDT", "ETHUSDT"],
    "T1": ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"],
    "T2": ["NEARUSDT", "ATOMUSDT", "FILUSDT", "ARBUSDT", "OPUSDT", "INJUSDT"],
    "T3": ["GALAUSDT", "CHZUSDT", "SANDUSDT", "ALGOUSDT"],
}
CRYPTO_SYMBOLS = [s for tier in CRYPTO_TIERS.values() for s in tier]

OANDA_INSTRUMENTS = [
    "EUR_USD",   # control
    "USD_MXN", "USD_ZAR", "USD_TRY",          # exotic FX
    "XCU_USD", "JP225_USD", "WTICO_USD",      # CFD copper / Nikkei / WTI
]
OANDA_BASE = "https://api-fxpractice.oanda.com/v3"


def _load_manifest(branch: str) -> dict:
    path = MANIFESTS[branch]
    if path.exists():
        return json.loads(path.read_text())
    return {branch: {}, "spec": "r1_illiquidity_probe_spec_2026-06-12"}


def _save_manifest(branch: str, m: dict) -> None:
    m["updated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFESTS[branch].write_text(json.dumps(m, indent=2))


def _finalize(df: pd.DataFrame, path: Path) -> dict:
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False)
    ts = pd.to_datetime(df["timestamp"])
    return {
        "rows": int(len(df)),
        "t0": str(ts.iloc[0]),
        "t1": str(ts.iloc[-1]),
        "file": str(path.relative_to(ROOT)),
        "status": "OK",
    }


# ---------------------------------------------------------------- crypto ----
# Source: Binance Vision public dumps (data.binance.vision) — USDT-M futures
# monthly 1m kline zips. Chosen because api.binance.com and api.bybit.com are
# geo-blocked (HTTP 451/403) from this workstation, while the dump CDN is not
# (verified 2026-06-12). Monthly zips are also faster + deterministic vs
# paginated REST. Falls back to daily zips for months without a monthly file.

VISION = "https://data.binance.vision/data/futures/um"


def _parse_kline_zip(content: bytes) -> pd.DataFrame:
    import io
    import zipfile
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, header=None, usecols=range(6), names=cols)
    # Newer dumps carry a header row — drop it if open_time isn't numeric.
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df = df.astype({"open_time": "int64", "open": "float64", "high": "float64",
                    "low": "float64", "close": "float64", "volume": "float64"})
    # 2025+ some Binance dumps switched open_time to microseconds — auto-detect.
    unit = "us" if df["open_time"].iloc[0] > 10**14 else "ms"
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, utc=True).dt.tz_localize(None)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _download(url: str, session, retries: int = 4) -> bytes | None:
    """GET with retry; returns None on 404 (file genuinely absent)."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"download failed after {retries} tries: {url}: {exc}")
            time.sleep(2.0 * (attempt + 1))
    return None


def fetch_crypto_symbol_vision(session, symbol: str) -> pd.DataFrame:
    months = pd.period_range(pd.Timestamp(START), pd.Timestamp(END) - pd.Timedelta(days=1), freq="M")
    frames: list[pd.DataFrame] = []
    for m in months:
        url = f"{VISION}/monthly/klines/{symbol}/1m/{symbol}-1m-{m}.zip"
        content = _download(url, session)
        if content is None:
            # monthly missing -> try daily zips for that month
            got_daily = 0
            for d in pd.date_range(m.start_time, m.end_time, freq="D"):
                durl = f"{VISION}/daily/klines/{symbol}/1m/{symbol}-1m-{d:%Y-%m-%d}.zip"
                dcontent = _download(durl, session)
                if dcontent is not None:
                    frames.append(_parse_kline_zip(dcontent))
                    got_daily += 1
            print(f"  {symbol} {m}: monthly missing, {got_daily} daily files", flush=True)
            continue
        frames.append(_parse_kline_zip(content))
    if not frames:
        raise RuntimeError("no files found on Binance Vision")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["timestamp"] >= pd.Timestamp(START)) & (df["timestamp"] < pd.Timestamp(END))]
    return df


def run_crypto(symbols: list[str]) -> None:
    import requests
    session = requests.Session()
    manifest = _load_manifest("crypto")
    for sym in symbols:
        path = OUT_DIR / f"crypto_{sym}.parquet"
        entry = manifest["crypto"].get(sym, {})
        if entry.get("status") == "OK" and entry.get("complete"):
            print(f"{sym}: already complete, skip", flush=True)
            continue
        print(f"{sym}: fetching from Binance Vision", flush=True)
        try:
            df = fetch_crypto_symbol_vision(session, sym)
            if len(df) == 0:
                raise RuntimeError("zero rows fetched")
            info = _finalize(df, path)
            info["complete"] = pd.to_datetime(info["t1"]) >= pd.Timestamp(END) - pd.Timedelta(days=2)
            tier = next(t for t, ss in CRYPTO_TIERS.items() if sym in ss)
            info["tier"] = tier
            info["source"] = "binance_vision_um_futures"
            manifest["crypto"][sym] = info
            print(f"{sym}: OK rows={info['rows']} t0={info['t0']} t1={info['t1']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            manifest["crypto"][sym] = {"status": "UNTESTABLE", "reason": str(exc)[:300]}
            print(f"{sym}: UNTESTABLE ({exc})", flush=True)
        _save_manifest("crypto", manifest)


# ----------------------------------------------------------------- oanda ----

def _oanda_token() -> str:
    env_file = ROOT / ".env"
    for line in env_file.read_text().splitlines():
        if line.startswith("OANDA_TOKEN"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OANDA_TOKEN not found in .env")


def fetch_oanda_instrument(token: str, instrument: str,
                           start: pd.Timestamp, end: pd.Timestamp,
                           existing: pd.DataFrame | None) -> pd.DataFrame:
    import requests
    cursor = start
    if existing is not None and len(existing) > 0:
        cursor = max(cursor, pd.Timestamp(existing["timestamp"].iloc[-1]) + pd.Timedelta(minutes=1))
    headers = {"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"}
    all_rows: list = []
    req = 0
    while cursor < end:
        params = {
            "granularity": "M1", "price": "M", "count": 5000,
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        }
        r = requests.get(f"{OANDA_BASE}/instruments/{instrument}/candles",
                         headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"OANDA {r.status_code}: {r.text[:200]}")
        candles = r.json().get("candles", [])
        req += 1
        if not candles:
            break
        for c in candles:
            if not c.get("complete"):
                continue
            t = pd.Timestamp(c["time"]).tz_convert("UTC").tz_localize(None)
            if t >= end:  # end is tz-naive UTC by construction
                continue
            p = c["mid"]
            all_rows.append({
                "timestamp": t,
                "open": float(p["o"]), "high": float(p["h"]),
                "low": float(p["l"]), "close": float(p["c"]),
                "volume": int(c.get("volume", 0)),
            })
        last_t = pd.Timestamp(candles[-1]["time"]).tz_convert("UTC").tz_localize(None)
        if last_t <= cursor:
            break
        cursor = last_t + pd.Timedelta(minutes=1)
        if req % 25 == 0:
            print(f"  {instrument}: req {req}, up to {last_t:%Y-%m-%d}", flush=True)
        time.sleep(0.15)
    new = pd.DataFrame(all_rows)
    if existing is not None and len(existing) > 0 and len(new) > 0:
        new = pd.concat([existing, new], ignore_index=True)
    elif existing is not None and len(new) == 0:
        new = existing
    return new


def run_oanda(instruments: list[str]) -> None:
    token = _oanda_token()
    start = pd.Timestamp(START)
    end = pd.Timestamp(END)
    manifest = _load_manifest("oanda")
    for inst in instruments:
        path = OUT_DIR / f"oanda_{inst}.parquet"
        entry = manifest["oanda"].get(inst, {})
        if entry.get("status") == "OK" and entry.get("complete"):
            print(f"{inst}: already complete, skip", flush=True)
            continue
        existing = pd.read_parquet(path) if path.exists() else None
        print(f"{inst}: fetching (resume={existing is not None})", flush=True)
        try:
            df = fetch_oanda_instrument(token, inst, start, end, existing)
            if df is None or len(df) == 0:
                raise RuntimeError("zero rows fetched")
            info = _finalize(df, path)
            # FX closes weekends; complete = within 4 days of END
            info["complete"] = pd.to_datetime(info["t1"]) >= end - pd.Timedelta(days=4)
            manifest["oanda"][inst] = info
            print(f"{inst}: OK rows={info['rows']} t1={info['t1']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            manifest["oanda"][inst] = {"status": "UNTESTABLE", "reason": str(exc)[:300]}
            print(f"{inst}: UNTESTABLE ({exc})", flush=True)
        _save_manifest("oanda", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["crypto", "oanda"], required=True)
    parser.add_argument("--symbols", default=None, help="comma-separated subset")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.branch == "crypto":
        syms = args.symbols.split(",") if args.symbols else CRYPTO_SYMBOLS
        run_crypto(syms)
    else:
        insts = args.symbols.split(",") if args.symbols else OANDA_INSTRUMENTS
        run_oanda(insts)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
