#!/usr/bin/env python3
"""Measure real per-side trading cost on cTrader XAUUSD paper account.

Used to pick the steady-state ``taker_fee`` for GMGP1-XAUUSD configs
(S461/S463 fee-schedule audit) — pulls recent fills via
``CTraderBroker.get_deal_list``, converts to bps per side, and prints
summary stats so the value in ``configs/gmgp1_xauusd_ftmo_hpo.yaml``
can match broker reality.

Usage:
    # Load DEMO credentials from .env.ctrader (default)
    python scripts/analyze_xauusd_fill_costs.py

    # Custom lookback
    python scripts/analyze_xauusd_fill_costs.py --days 14
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from finrl_pro_ds.cfd.execution.ctrader_broker import CTraderBroker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_xauusd_fill_costs")


async def analyze(days: int) -> int:
    broker = CTraderBroker(testnet=True)

    logger.info(
        "Connecting to cTrader (account=%s, testnet=%s)…",
        broker._account_id, broker._testnet,
    )
    await broker.connect()

    try:
        deals = await broker.get_deal_list(from_days_ago=days)
    finally:
        await broker.close()

    n = len(deals)
    if n == 0:
        logger.warning(
            "No deals found in last %d days on account %s. "
            "Run paper fills first or pass --days <larger>.",
            days, broker._account_id,
        )
        return 0

    # Per-side commission in bps: commission_usd / notional_usd * 10000
    # notional = lots × lot_size × executionPrice
    per_side_bps: list[float] = []
    skipped = 0
    for d in deals:
        notional = d["lots"] * broker._lot_size * d["executionPrice"]
        if notional <= 0:
            skipped += 1
            continue
        bps = d["commission"] / notional * 10_000
        per_side_bps.append(bps)

    if not per_side_bps:
        logger.warning("All %d deals had zero notional; cannot compute bps.", n)
        return 0

    arr = np.array(per_side_bps)
    mean_bps = float(arr.mean())
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    # Effective spread (optional): pair adjacent BUY/SELL deals.  Skip silently
    # if too noisy — the commission number is what drives the config edit.
    buys = [d for d in deals if d["tradeSide"] == "BUY"]
    sells = [d for d in deals if d["tradeSide"] == "SELL"]
    spread_bps_est: float | None = None
    if buys and sells:
        mean_buy = mean(d["executionPrice"] for d in buys)
        mean_sell = mean(d["executionPrice"] for d in sells)
        mid_est = 0.5 * (mean_buy + mean_sell)
        if mid_est > 0:
            spread_bps_est = abs(mean_buy - mean_sell) / mid_est * 10_000 / 2.0

    print("=" * 60)
    print(f"XAUUSD fill-cost analysis — account {broker._account_id}")
    print(f"  lookback:           {days} days")
    print(f"  deals analyzed:     {len(per_side_bps)} (skipped {skipped})")
    print(f"  mean commission:    {mean_bps:.3f} bps/side")
    print(f"  p50 commission:     {p50:.3f} bps/side")
    print(f"  p95 commission:     {p95:.3f} bps/side")
    print(f"  p99 commission:     {p99:.3f} bps/side")
    if spread_bps_est is not None:
        print(f"  est spread (noisy): {spread_bps_est:.3f} bps/side")
        print(
            f"  combined est:       "
            f"{mean_bps + spread_bps_est:.3f} bps/side"
        )
    else:
        print("  est spread:         n/a (need both BUY and SELL fills)")
    print("=" * 60)
    print(
        "Suggested config value: taker_fee = "
        f"{(mean_bps + (spread_bps_est or 0.6)) / 10_000:.6f}"
    )
    print(
        "  (mean commission bps + spread estimate or 0.6 bps fallback, "
        "expressed as fraction)"
    )
    print("=" * 60)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="Lookback window (days).")
    ap.add_argument(
        "--env-file",
        default=".env.ctrader",
        help="Path to .env file with CTRADER_* vars (default: .env.ctrader).",
    )
    args = ap.parse_args()

    env_path = Path(args.env_file)
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=False)
    elif not os.environ.get("CTRADER_CLIENT_ID"):
        logger.error(
            "No CTRADER_* env vars set and %s not found. "
            "Install python-dotenv or export the vars manually.",
            env_path,
        )
        return 1

    return asyncio.run(analyze(args.days))


if __name__ == "__main__":
    sys.exit(main())
