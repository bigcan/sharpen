"""Deribit live option-chain logger — a FREE forward real-chain dataset builder.

Snapshots the REAL (mainnet) Deribit inverse option chain to parquet, in the SAME
schema as the historical Tardis chain (`data/processed/deribit_chain/`), so the
existing `options_vrp_skew_falsification` backtest can later consume it WITHOUT paying
for historical data. Read-only public market data — NO auth, NO orders.

Why this exists: the options-VRP sleeve's only real-execution evidence is the (paid)
historical Tardis chain. Rather than buy it (~$700-2700), run this daily; after 6-12
months you own a real DAILY inverse-chain dataset for $0 and can run the true 21d-roll
real-chain re-validation the de-contamination re-audit demanded
(docs/research/options_vrp_decontamination_reaudit_2026-06-23.md).

Greeks: only `delta` is read by the backtest (25-delta strangle / wing selection); it is
computed from `mark_iv` via the project's single-source-of-truth BS module
(`finrl_pro_ds.crypto.options_pricing`). gamma/vega/theta are filled for schema parity;
bid_iv/ask_iv are left NaN (the live book summary does not publish them — only a
diagnostic in the backtest). INVERSE vs USDC-LINEAR is preserved in `symbol` so the
backtest's `INVERSE_ONLY` filter still applies (the contamination lesson).

Run (daily):
    python scripts/research/deribit_chain_logger.py                 # one BTC+ETH snapshot
    python scripts/research/deribit_chain_logger.py --loop 86400     # self-scheduling daily
    python scripts/research/deribit_chain_logger.py --currencies BTC --testnet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve finrl_pro_ds when run as a script (script dir shadows cwd on sys.path).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402  (after sys.path bootstrap so scheduled runs resolve finrl_pro_ds)
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from finrl_pro_ds.crypto.options_pricing import (  # noqa: E402
    ANN,
    leg_delta,
    leg_gamma,
    leg_vega,
    straddle_theta,
)

logger = logging.getLogger("deribit_chain_logger")

MAINNET = "https://www.deribit.com/api/v2"
TESTNET = "https://test.deribit.com/api/v2"
OUT_DIR = _ROOT / "data" / "processed" / "deribit_chain_live"

# Historical Tardis chain schema (data/processed/deribit_chain/*.parquet) — match it exactly.
SCHEMA = ["symbol", "timestamp", "type", "strike_price", "expiration", "underlying_price",
          "bid_price", "bid_iv", "ask_price", "ask_iv", "mark_price", "mark_iv",
          "open_interest", "delta", "gamma", "vega", "theta", "snapshot_date", "expiry_dt"]

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _get(base_url: str, method: str, params: dict, *, retries: int = 4, timeout: int = 25) -> list | dict:
    """GET a Deribit public endpoint with simple exponential backoff (rate-limit safe)."""
    url = f"{base_url}/public/{method}"
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("result", [])
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:  # noqa: PERF203
            last = repr(e)
        time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Deribit {method} failed after {retries} tries: {last}")


def parse_instrument(name: str) -> tuple[str, float, pd.Timestamp] | None:
    """`BTC-26DEC25-90000-C` / `BTC_USDC-26DEC25-90000-P` -> (type, strike, expiry_dt UTC 08:00).

    Returns None for non-standard (combo/future-style) names. Deribit options expire 08:00 UTC.
    The base token (with or without `_USDC`) is preserved in the caller's `symbol`.
    """
    parts = name.split("-")
    if len(parts) != 4:
        return None
    _base, date_s, strike_s, cp = parts
    if cp not in ("C", "P") or len(date_s) < 7:
        return None
    try:
        day = int(date_s[:-5])
        mon = _MONTHS[date_s[-5:-2]]
        yr = 2000 + int(date_s[-2:])
        strike = float(strike_s)
    except (KeyError, ValueError):
        return None
    expiry = pd.Timestamp(datetime(yr, mon, day, 8, 0, tzinfo=timezone.utc))
    return ("call" if cp == "C" else "put"), strike, expiry


def _greeks(opt_type: str, S: float, K: float, mark_iv_pct: float, tau: float) -> tuple:
    """delta/gamma/vega/theta from mark_iv (single-source BS). `delta` is the one the
    backtest reads (call +, put -); the rest are schema-parity. theta_per_leg = straddle/2
    (call==put at r=0). Degenerate (tau<=0 / iv<=0) -> 0.0."""
    sig = mark_iv_pct / 100.0
    if not (np.isfinite(S) and np.isfinite(K) and S > 0 and K > 0 and sig > 0 and tau > 0):
        return np.nan, np.nan, np.nan, np.nan
    d = leg_delta(opt_type, S, K, sig, tau)
    g = leg_gamma(opt_type, S, K, sig, tau)
    v = leg_vega(opt_type, S, K, sig, tau)
    th = straddle_theta(S, K, sig, tau) / 2.0  # per-leg theta (call==put at r=0)
    return d, g, v, th


def snapshot_currency(currency: str, base_url: str, snap_dt: pd.Timestamp) -> pd.DataFrame:
    """One full option-chain snapshot for `currency` as a schema-matched DataFrame."""
    book = _get(base_url, "get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    rows = []
    for it in book:
        parsed = parse_instrument(it["instrument_name"])
        if parsed is None:
            continue
        otype, strike, expiry = parsed
        tau = max((expiry - snap_dt).total_seconds() / 86400.0 / ANN, 0.0)
        S = it.get("underlying_price")
        mark_iv = it.get("mark_iv")
        S = float(S) if S is not None else np.nan
        mark_iv = float(mark_iv) if mark_iv is not None else np.nan
        dlt, gam, veg, tht = _greeks(otype, S, strike, mark_iv, tau) if np.isfinite(mark_iv) \
            else (np.nan, np.nan, np.nan, np.nan)
        rows.append({
            "symbol": it["instrument_name"],
            "timestamp": it.get("creation_timestamp"),
            "type": otype,
            "strike_price": strike,
            "expiration": int(expiry.timestamp() * 1000),
            "underlying_price": S,
            "bid_price": it.get("bid_price"),
            "bid_iv": np.nan,                 # not published in the live book summary
            "ask_price": it.get("ask_price"),
            "ask_iv": np.nan,
            "mark_price": it.get("mark_price"),
            "mark_iv": mark_iv,               # PERCENT (e.g. 35.26), matches Tardis schema
            "open_interest": it.get("open_interest"),
            "delta": dlt, "gamma": gam, "vega": veg, "theta": tht,
            "snapshot_date": snap_dt.normalize(),
            "expiry_dt": expiry,
        })
    df = pd.DataFrame(rows, columns=SCHEMA)
    return df


def run_once(currencies: list[str], out_dir: Path, base_url: str) -> Path:
    """Snapshot all `currencies` once -> a single daily parquet. Idempotent per UTC day."""
    snap_dt = pd.Timestamp.now(tz="UTC")
    frames = []
    for ccy in currencies:
        df = snapshot_currency(ccy, base_url, snap_dt)
        n_inv = int((~df["symbol"].str.contains("_", regex=False)).sum())
        logger.info("%s: %d instruments (%d inverse, %d linear)", ccy, len(df), n_inv, len(df) - n_inv)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"chain_{snap_dt.strftime('%Y-%m-%d')}.parquet"
    out.to_parquet(fpath, index=False)
    logger.info("wrote %s (%d rows, %d expiries, underlying BTC=%.0f)",
                fpath, len(out), out["expiry_dt"].nunique(),
                float(out.loc[out.symbol.str.startswith("BTC"), "underlying_price"].median()))
    return fpath


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Free Deribit live option-chain logger")
    ap.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--testnet", action="store_true", help="log test.deribit.com instead of mainnet (NOT real market data)")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="self-schedule: snapshot every N seconds (e.g. 86400 = daily); 0 = once and exit")
    args = ap.parse_args()
    base_url = TESTNET if args.testnet else MAINNET
    out_dir = Path(args.out_dir)
    logger.info("Deribit chain logger | %s | currencies=%s | out=%s",
                "TESTNET" if args.testnet else "MAINNET", args.currencies, out_dir)
    while True:
        try:
            run_once(args.currencies, out_dir, base_url)
        except Exception:
            logger.exception("snapshot failed (will retry next cycle)" if args.loop else "snapshot failed")
            if not args.loop:
                raise
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
