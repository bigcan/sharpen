"""Taiwan INTRADAY 1-min ingestion via FinMind (Sponsor tier) → clean → resample to 3m/15m.

Produces the bar files sg-1 (3m) and gmgp1 (15m) consume, for a Taiwan-asset test of the
intraday multiscale directional RL (the S553-cont-85 less-efficient-market thesis: the
directional signal absent in efficient BTC/gold may exist in a retail-driven Taiwan market).

Data routes (FinMind, verified schemas — finmind.github.io/llms-full.txt):
  - STOCK 1-min: dataset ``TaiwanStockKBar`` (Sponsor), cols date/minute/stock_id/open/high/
    low/close/volume, 2019-01-01→, ONE DAY PER REQUEST.
  - FUTURES (TXF/TX): NO minute dataset exists → ``TaiwanFuturesTick`` (Sponsor, 2011→, ONE
    DAY PER REQUEST, **tick volume is DOUBLE-SIDED** → ÷2) aggregated client-side to 1-min.

Pipeline parity with the rest of the project:
  - Runs the canonical ``scripts.clean_ohlcv`` detect/repair + stale-print scan (the cleaner
    is timeframe-aware; at 1-min it is the strict R1 gate) on the 1-min series, THEN resamples.
  - **LEAK-2**: resample is causal (OHLCV aggregated within each closed window; the env acts on
    the CLOSED bar). Timestamps are tz-naive Asia/Taipei local (GMT+8), matching gmgp1/sg-1.
  - Earned-status manifest per interval (mirrors fetch_taiwan_finmind / cross_asset_loader).

Needs a FinMind **Sponsor** token (the free tier returns HTTP 400 "Please update your user
level" on intraday datasets — handled explicitly below). Set FINMIND_TOKEN or pass --token.

Usage:
    # validate the clean/resample/manifest core with NO token (synthetic 1-min):
    python scripts/data/fetch_taiwan_intraday_finmind.py --selftest

    # real fetch once you have a Sponsor token:
    python scripts/data/fetch_taiwan_intraday_finmind.py --mode stock --id 2330 \
        --start 2019-01-01 --end 2026-06-27 --resample 3min,15min --out data/taiwan_intraday
    python scripts/data/fetch_taiwan_intraday_finmind.py --mode futures --id TX \
        --start 2019-01-01 --resample 3min,15min --out data/taiwan_intraday
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("fetch_taiwan_intraday")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
LOADER_MANIFEST_VERSION = 2
_OHLCV = ["open", "high", "low", "close", "volume"]
# Resample aggregation — open=first, high=max, low=min, close=last, volume=sum (same as
# scripts.clean_ohlcv.rederive_downstream; causal within each window).
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# --------------------------------------------------------------------------- #
# FinMind single-day intraday fetch (explicit paywall handling)
# --------------------------------------------------------------------------- #
def _finmind_day(dataset: str, data_id: str, date: str, token: str) -> pd.DataFrame:
    """One (dataset, data_id, single day) pull. Surfaces the free-tier paywall + quota clearly."""
    params = {"dataset": dataset, "data_id": data_id, "start_date": date, "end_date": date}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(FINMIND_URL, headers=headers, params=params, timeout=90)
    if resp.status_code == 400 and "level" in resp.text.lower():
        raise PermissionError(
            f"{dataset} requires a FinMind SPONSOR tier (HTTP 400: {resp.json().get('msg')}). "
            "Subscribe + set FINMIND_TOKEN; the free tier does not serve intraday data."
        )
    if resp.status_code == 402:
        raise RuntimeError(f"FinMind quota exhausted (HTTP 402) on {dataset}/{data_id} {date}.")
    if resp.status_code != 200:
        raise RuntimeError(f"FinMind HTTP {resp.status_code} on {dataset}/{data_id} {date}: "
                           f"{resp.text[:160]}")
    return pd.DataFrame(resp.json().get("data", []))


def _trading_days(start: str, end: str) -> list[str]:
    """Weekday calendar (Mon-Fri) as ISO strings — TWSE/TAIFEX holidays simply return empty."""
    days = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in days]


# --------------------------------------------------------------------------- #
# Normalizers → tidy 1-min [timestamp, open, high, low, close, volume]
# --------------------------------------------------------------------------- #
def _stock_kbar_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """TaiwanStockKBar day frame → 1-min OHLCV with a tz-naive Asia/Taipei timestamp."""
    if df.empty:
        return df
    need = {"date", "minute", *_OHLCV}
    if not need.issubset(df.columns):
        raise RuntimeError(f"TaiwanStockKBar missing {need - set(df.columns)} (got {list(df.columns)})")
    ts = pd.to_datetime(df["date"].astype(str) + " " + df["minute"].astype(str),
                        errors="coerce")
    out = df[_OHLCV].apply(pd.to_numeric, errors="coerce")
    out.insert(0, "timestamp", ts)
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _futures_tick_to_1min(df: pd.DataFrame) -> pd.DataFrame:
    """TaiwanFuturesTick day frame → 1-min OHLCV. Front contract only; volume ÷2 (double-sided)."""
    if df.empty:
        return df
    if "futures_id" in df.columns and df["futures_id"].nunique() > 1:
        df = df[df["futures_id"] == df["futures_id"].mode().iloc[0]]
    # Front month = nearest contract_date (earliest, since only >= current list on a trade day).
    if "contract_date" in df.columns and df["contract_date"].nunique() > 1:
        df = df[df["contract_date"] == sorted(df["contract_date"].astype(str).unique())[0]]
    # Time column name varies across FinMind revisions — accept the first time-like column.
    tcol = next((c for c in ("Time", "time", "deal_time") if c in df.columns), None)
    base = df["date"].astype(str)
    ts = pd.to_datetime(base + " " + df[tcol].astype(str), errors="coerce") if tcol \
        else pd.to_datetime(base, errors="coerce")
    px = pd.to_numeric(df["price"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce") / 2.0  # double-sided convention
    tk = pd.DataFrame({"timestamp": ts, "price": px, "volume": vol}).dropna(
        subset=["timestamp", "price"]).set_index("timestamp").sort_index()
    bars = tk["price"].resample("1min").ohlc()
    bars["volume"] = tk["volume"].resample("1min").sum()
    return bars.dropna(subset=["open", "high", "low", "close"]).reset_index()


# --------------------------------------------------------------------------- #
# Clean (canonical) + resample + manifest
# --------------------------------------------------------------------------- #
def _clean_1min(min_df: pd.DataFrame, threshold: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Run the project cleaner (detect/repair + stale-print scan) on the 1-min series."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.clean_ohlcv import detect_outliers, detect_stale_runs, repair_outliers

    df = min_df.set_index("timestamp")[_OHLCV].copy()
    det = detect_outliers(df, threshold=threshold)
    n_bad = int((det["bad_high"] | det["bad_low"]).sum())
    if n_bad:
        df = repair_outliers(df, det, threshold=threshold)
    stale = detect_stale_runs(df)
    rep = {"rows": int(len(df)), "outliers_repaired": n_bad,
           "stale_suspect_frac": stale["stale_suspect_frac"],
           "stale_pnl_share": stale["stale_pnl_share"],
           "flat_ohlc_spikes": stale["flat_ohlc_spikes"], "stale_flagged": bool(stale["flagged"])}
    return df.reset_index(), rep


def _resample(min_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Causal OHLCV resample of a clean 1-min frame to `rule` (e.g. '3min','15min')."""
    d = min_df.set_index("timestamp").sort_index()
    out = d.resample(rule, label="left", closed="left").agg(_AGG)
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _write(df: pd.DataFrame, rep: dict, out_dir: Path, name: str, interval: str,
           source: str, instrument: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "FAIL" if rep["stale_pnl_share"] >= 0.02 else ("WARN" if rep["stale_flagged"] else "PASS")
    payload = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
    manifest = {
        "stage": "data-prep", "status": status, "loader_manifest_version": LOADER_MANIFEST_VERSION,
        "source": source, "instrument": instrument, "adjusted": False,
        "timezone": "Asia/Taipei (GMT+8), tz-naive local", "frequency": interval,
        "n_rows": int(len(df)),
        "date_min": str(df["timestamp"].min()), "date_max": str(df["timestamp"].max()),
        "stale_scan": {k: rep[k] for k in ("stale_suspect_frac", "stale_pnl_share",
                                           "flat_ohlc_spikes", "stale_flagged", "outliers_repaired")},
        "content_sha256_16": payload,
    }
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %s (%d %s bars) [%s]", out_dir / f"{name}.parquet", len(df), interval, status)
    return status


def run(min_df: pd.DataFrame, intervals: list[str], out_dir: Path, *, source: str,
        instrument: str) -> None:
    """Clean the 1-min series, write it, then write each resampled interval."""
    clean1, rep1 = _clean_1min(min_df)
    _write(clean1, rep1, out_dir, f"{instrument}_1min", "1min", source, instrument)
    for rule in intervals:
        res = _resample(clean1, rule)
        _, repr_ = _clean_1min(res) if len(res) else (res, rep1)
        _write(res, repr_ if len(res) else rep1, out_dir, f"{instrument}_{rule}", rule,
               source, instrument)


# --------------------------------------------------------------------------- #
# Self-test (no token): proves clean + resample + manifest on synthetic 1-min
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    idx = pd.date_range("2026-06-01 09:00", "2026-06-01 13:30", freq="1min")
    rng = np.random.default_rng(0)
    price = 600 + np.cumsum(rng.standard_normal(len(idx)) * 0.2)
    df = pd.DataFrame({"timestamp": idx, "open": price, "high": price + 0.3,
                       "low": price - 0.3, "close": price + rng.standard_normal(len(idx)) * 0.1,
                       "volume": rng.integers(1, 500, len(idx))})
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    out = Path(os.environ.get("TMPDIR", ".")) / "_tw_intraday_selftest"
    run(df, ["3min", "15min"], out, source="selftest_synthetic", instrument="SELF")
    n1 = len(pd.read_parquet(out / "SELF_1min.parquet"))
    n3 = len(pd.read_parquet(out / "SELF_3min.parquet"))
    n15 = len(pd.read_parquet(out / "SELF_15min.parquet"))
    ok = (n1 == len(df)) and (abs(n3 - n1 / 3) <= 1) and (abs(n15 - n1 / 15) <= 1)
    log.info("SELFTEST 1min=%d 3min=%d 15min=%d → %s", n1, n3, n15, "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Taiwan intraday 1-min ingest (FinMind Sponsor) + resample")
    ap.add_argument("--mode", choices=["stock", "futures"], default="stock")
    ap.add_argument("--id", default="2330", help="stock_id (e.g. 2330) or futures_id (e.g. TX)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--resample", default="3min,15min", help="comma intervals → sg-1=3min, gmgp1=15min")
    ap.add_argument("--out", default="data/taiwan_intraday")
    ap.add_argument("--token", default=os.environ.get("FINMIND_TOKEN", ""))
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--selftest", action="store_true", help="validate core on synthetic 1-min, no token")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.token:
        log.error("No FINMIND_TOKEN — intraday datasets need a Sponsor token. Run --selftest to "
                  "validate the pipeline, or set the token once subscribed.")
        return 2

    intervals = [s.strip() for s in args.resample.split(",") if s.strip()]
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    dataset = "TaiwanStockKBar" if args.mode == "stock" else "TaiwanFuturesTick"
    norm = _stock_kbar_to_1min if args.mode == "stock" else _futures_tick_to_1min
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    frames: list[pd.DataFrame] = []
    days = _trading_days(args.start, end)
    log.info("Fetching %s %s over %d trading days (%s..%s)", dataset, args.id, len(days),
             args.start, end)
    for i, day in enumerate(days):
        try:
            raw = _finmind_day(dataset, args.id, day, args.token)
        except PermissionError as e:
            log.error("%s", e)
            return 3
        if not raw.empty:
            frames.append(norm(raw))
        if i % 50 == 0 and i:
            log.info("  ...%d/%d days, %d bars so far", i, len(days),
                     sum(len(f) for f in frames))
        time.sleep(args.sleep)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        log.error("No intraday data returned for %s %s.", dataset, args.id)
        return 2
    min_df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values(
        "timestamp").reset_index(drop=True)
    log.info("Assembled %d 1-min bars (%s..%s)", len(min_df), min_df["timestamp"].min(),
             min_df["timestamp"].max())
    run(min_df, intervals, out_dir, source=f"finmind_sponsor_{dataset}", instrument=str(args.id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
