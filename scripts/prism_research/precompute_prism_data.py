"""Phase 0A: Pre-compute PRISM features via API for GC=F 2025.

Calls the PRISM API (remote desktop Docker stack) to generate:
  - GAHMM dual-regime labels (price + vol regime probabilities)
  - Chronos-2 quantile forecasts (p10, p30, p50, p70, p90, spread)

Caches results as parquet for all subsequent PRISM research phases.
No SAFFS local imports required — everything goes through the API.

Usage:
    # With SSH tunnel active (ssh -L 8001:localhost:8001 user@<TAILSCALE_HOST>):
    python scripts/prism_research/precompute_prism_data.py

    # Direct on remote desktop:
    python scripts/prism_research/precompute_prism_data.py --api-url http://localhost:8001

    # Resume from partial run:
    python scripts/prism_research/precompute_prism_data.py --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root and PRISM SDK to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "docker" / "live" / "prism_sdk"))

from prism_client import PRISMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prism_research.precompute")

# ---------- constants ----------
DATA_FILE = PROJECT_ROOT / "data" / "cme" / "gc_2025_lob1_1min_stitched.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "prism_research"
OUTPUT_FILE = OUTPUT_DIR / "prism_features_gc_2025.parquet"
PARTIAL_FILE = OUTPUT_DIR / "prism_features_gc_2025_partial.csv"

TICKER = "GC=F"
TIMEFRAME = "daily"
CHRONOS_CONTEXT = 512  # Number of daily closes to send for forecast
FORECAST_HORIZON = 10


def load_daily_ohlcv() -> pd.DataFrame:
    """Load 1-min GC parquet and resample to daily OHLCV."""
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)

    # Ensure timestamp column
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        else:
            raise ValueError("No timestamp/date column found in parquet")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.set_index("timestamp").sort_index()

    # Resample to daily
    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    logger.info(f"Daily OHLCV: {len(daily)} trading days "
                f"({daily.index[0].date()} → {daily.index[-1].date()})")
    return daily


def check_api_health(client: PRISMClient) -> bool:
    """Verify PRISM API is operational."""
    try:
        health = client.health()
        logger.info(f"PRISM API health: status={health.status}, "
                    f"chronos={health.chronos_loaded}, hmm={health.hmm_service}")
        if health.status != "ok":
            logger.error(f"PRISM API unhealthy: {health.status}")
            return False
        if not health.chronos_loaded:
            logger.warning("Chronos-2 not loaded — forecast features will be unavailable")
        if health.hmm_service != "operational":
            logger.warning("HMM service not operational — regime features will be unavailable")
        return True
    except Exception as e:
        logger.error(f"Cannot reach PRISM API: {e}")
        return False


def refit_gahmm(client: PRISMClient) -> None:
    """Trigger GAHMM refit on GC=F with full 2024-2025 lookback."""
    logger.info(f"Triggering GAHMM refit for {TICKER} (lookback=730 days)...")
    try:
        result = client.refit(
            ticker=TICKER,
            timeframe=TIMEFRAME,
            model_type="both",
            warm_start=True,
            lookback_days=730,
        )
        logger.info(f"Refit complete: status={result.status}, versions={result.versions}")
    except Exception as e:
        logger.error(f"Refit failed: {e}")
        raise


def try_regime_history(client: PRISMClient, start: str, end: str) -> pd.DataFrame | None:
    """Try to get historical regime predictions from PRISM DB."""
    logger.info(f"Attempting regime history query: {start} → {end}")
    try:
        history = client.get_regime_history(
            ticker=TICKER,
            timeframe=TIMEFRAME,
            start=start,
            end=end,
            limit=400,
        )
        if not history:
            logger.warning("No regime history found in PRISM DB")
            return None

        rows = []
        for h in history:
            rows.append({
                "date": pd.Timestamp(h.timestamp).normalize(),
                "price_regime": h.price_regime,
                "vol_regime": h.vol_regime,
                "composite_code": h.composite_code,
                "confidence": h.confidence,
                "price_bear_prob": h.price_probs.get("BEARISH", 1 / 3),
                "price_neutral_prob": h.price_probs.get("NEUTRAL", 1 / 3),
                "price_bull_prob": h.price_probs.get("BULLISH", 1 / 3),
                "vol_low_prob": h.vol_probs.get("LOW_VOL", 1 / 3),
                "vol_normal_prob": h.vol_probs.get("NORMAL_VOL", 1 / 3),
                "vol_high_prob": h.vol_probs.get("HIGH_VOL", 1 / 3),
            })

        df = pd.DataFrame(rows).set_index("date").sort_index()
        logger.info(f"Got {len(df)} regime history entries from PRISM DB")
        return df
    except Exception as e:
        logger.warning(f"Regime history query failed: {e}")
        return None


def compute_regime_via_predict(
    client: PRISMClient,
) -> pd.DataFrame | None:
    """Get current regime prediction (only works for latest date)."""
    try:
        regime = client.get_regime(ticker=TICKER, timeframe=TIMEFRAME)
        row = {
            "date": pd.Timestamp(regime.timestamp).normalize(),
            "price_regime": {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}.get(
                regime.price_regime, 1
            ),
            "vol_regime": {"LOW_VOL": 0, "NORMAL_VOL": 1, "HIGH_VOL": 2}.get(
                regime.vol_regime, 1
            ),
            "composite_code": regime.composite_code,
            "confidence": regime.confidence,
            "price_bear_prob": regime.price_probabilities.get("BEARISH", 1 / 3),
            "price_neutral_prob": regime.price_probabilities.get("NEUTRAL", 1 / 3),
            "price_bull_prob": regime.price_probabilities.get("BULLISH", 1 / 3),
            "vol_low_prob": regime.vol_probabilities.get("LOW_VOL", 1 / 3),
            "vol_normal_prob": regime.vol_probabilities.get("NORMAL_VOL", 1 / 3),
            "vol_high_prob": regime.vol_probabilities.get("HIGH_VOL", 1 / 3),
        }
        return pd.DataFrame([row]).set_index("date")
    except Exception as e:
        logger.warning(f"Regime predict failed: {e}")
        return None


def compute_chronos_forecasts(
    client: PRISMClient,
    daily: pd.DataFrame,
    resume_from: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute Chronos-2 forecasts for each trading day via API.

    For each date t, sends the last CHRONOS_CONTEXT daily closes ending at t
    to the /predict endpoint and caches the quantile outputs.
    """
    closes = daily["close"].values
    dates = daily.index

    # Determine which dates to process
    already_done = set()
    if resume_from is not None and "chronos_p50" in resume_from.columns:
        already_done = set(resume_from.index[resume_from["chronos_p50"].notna()])

    rows = []
    total = len(dates)
    skipped = 0

    for i, date in enumerate(dates):
        if date in already_done:
            skipped += 1
            continue

        # Need at least 30 prices for the API
        if i < 30:
            # Not enough context — fill with defaults
            rows.append({
                "date": date,
                "chronos_p10": 0.0,
                "chronos_p30": 0.0,
                "chronos_p50": 0.0,
                "chronos_p70": 0.0,
                "chronos_p90": 0.0,
                "chronos_spread": 0.0,
            })
            continue

        # Rolling window of prices up to date t (walk-forward safe)
        start_idx = max(0, i - CHRONOS_CONTEXT + 1)
        price_window = closes[start_idx:i + 1].tolist()

        try:
            forecast = client.get_forecast(
                ticker=TICKER,
                prices=price_window,
                horizon=FORECAST_HORIZON,
            )
            rows.append({
                "date": date,
                "chronos_p10": forecast.p10[0] if forecast.p10 else 0.0,
                "chronos_p30": forecast.p30[0] if forecast.p30 else 0.0,
                "chronos_p50": forecast.p50[0] if forecast.p50 else 0.0,
                "chronos_p70": forecast.p70[0] if forecast.p70 else 0.0,
                "chronos_p90": forecast.p90[0] if forecast.p90 else 0.0,
                "chronos_spread": forecast.quantile_spread,
            })
        except Exception as e:
            logger.warning(f"Chronos forecast failed for {date.date()}: {e}")
            rows.append({
                "date": date,
                "chronos_p10": 0.0,
                "chronos_p30": 0.0,
                "chronos_p50": 0.0,
                "chronos_p70": 0.0,
                "chronos_p90": 0.0,
                "chronos_spread": 0.0,
            })

        # Rate limit + progress
        if (i - skipped) % 10 == 0:
            logger.info(f"Chronos progress: {i + 1}/{total} dates "
                        f"({skipped} skipped from resume)")
        time.sleep(0.5)  # Gentle rate limiting

    if not rows:
        logger.info(f"All {skipped} Chronos forecasts already cached")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")
    logger.info(f"Computed {len(df)} Chronos forecasts ({skipped} skipped)")
    return df


def merge_and_save(
    regime_df: pd.DataFrame,
    chronos_df: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Merge regime labels + Chronos forecasts, fill gaps, save as parquet."""
    # Start with all trading dates
    all_dates = pd.DataFrame(index=daily.index)
    all_dates.index.name = "date"

    # Merge regime data
    if regime_df is not None and not regime_df.empty:
        all_dates = all_dates.join(regime_df, how="left")
    else:
        # Fill with uniform priors
        logger.warning("No regime data — filling with uniform priors")
        for col in ["price_regime", "vol_regime", "composite_code", "confidence"]:
            all_dates[col] = 1 if col in ("price_regime", "vol_regime") else (4 if col == "composite_code" else 0.33)
        for col in ["price_bear_prob", "price_neutral_prob", "price_bull_prob",
                     "vol_low_prob", "vol_normal_prob", "vol_high_prob"]:
            all_dates[col] = 1 / 3

    # Merge Chronos data
    if not chronos_df.empty:
        all_dates = all_dates.join(chronos_df, how="left")

    # Fill NaN with semantic defaults
    regime_prob_cols = [
        "price_bear_prob", "price_neutral_prob", "price_bull_prob",
        "vol_low_prob", "vol_normal_prob", "vol_high_prob",
    ]
    for col in regime_prob_cols:
        if col in all_dates.columns:
            all_dates[col] = all_dates[col].fillna(1 / 3)

    chronos_cols = [
        "chronos_p10", "chronos_p30", "chronos_p50",
        "chronos_p70", "chronos_p90", "chronos_spread",
    ]
    for col in chronos_cols:
        if col in all_dates.columns:
            all_dates[col] = all_dates[col].fillna(0.0)

    # Forward-fill regime labels (regime persists until next prediction)
    for col in ["price_regime", "vol_regime", "composite_code", "confidence"]:
        if col in all_dates.columns:
            all_dates[col] = all_dates[col].ffill()
            # Back-fill any remaining NaN at the start
            all_dates[col] = all_dates[col].bfill()

    # Add close price for reference
    all_dates["close"] = daily["close"]

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dates.to_parquet(OUTPUT_FILE, engine="pyarrow")
    logger.info(f"Saved PRISM features: {OUTPUT_FILE} ({len(all_dates)} rows)")

    # Summary stats
    if "vol_regime" in all_dates.columns:
        vol_counts = all_dates["vol_regime"].value_counts()
        logger.info(f"Vol regime distribution:\n{vol_counts.to_string()}")
    if "composite_code" in all_dates.columns:
        comp_counts = all_dates["composite_code"].value_counts().sort_index()
        logger.info(f"Composite code distribution:\n{comp_counts.to_string()}")

    return all_dates


def main():
    parser = argparse.ArgumentParser(description="Pre-compute PRISM features via API")
    parser.add_argument("--api-url", default="http://localhost:8001",
                        help="PRISM API URL (default: localhost:8001 via SSH tunnel)")
    parser.add_argument("--api-key", default="", help="PRISM API key (if required)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial run (skip already-computed dates)")
    parser.add_argument("--skip-refit", action="store_true",
                        help="Skip GAHMM refit (use existing model)")
    parser.add_argument("--skip-chronos", action="store_true",
                        help="Skip Chronos forecasts (regime labels only)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="API request timeout in seconds")
    args = parser.parse_args()

    # Initialize client
    client = PRISMClient(
        base_url=args.api_url,
        api_key=args.api_key or None,
        timeout=args.timeout,
    )

    # Health check
    if not check_api_health(client):
        logger.error("PRISM API health check failed. Ensure SSH tunnel is active:\n"
                      "  ssh -L 8001:localhost:8001 user@<TAILSCALE_HOST>")
        sys.exit(1)

    # Load daily OHLCV
    daily = load_daily_ohlcv()

    # Step 1: Refit GAHMM on GC=F
    if not args.skip_refit:
        refit_gahmm(client)
    else:
        logger.info("Skipping GAHMM refit (--skip-refit)")

    # Step 2: Get regime labels
    # First try history endpoint (if PRISM DB has stored predictions after refit)
    start_date = daily.index[0].strftime("%Y-%m-%d")
    end_date = daily.index[-1].strftime("%Y-%m-%d")
    regime_df = try_regime_history(client, start_date, end_date)

    if regime_df is None or len(regime_df) < 10:
        logger.warning("Insufficient regime history — using current regime predict as fallback")
        regime_df = compute_regime_via_predict(client)

    # Step 3: Compute Chronos forecasts
    resume_data = None
    if args.resume and PARTIAL_FILE.exists():
        resume_data = pd.read_csv(PARTIAL_FILE, index_col="date", parse_dates=True)
        logger.info(f"Resuming from {len(resume_data)} cached entries")

    if args.skip_chronos:
        logger.info("Skipping Chronos forecasts (--skip-chronos)")
        chronos_df = pd.DataFrame()
    else:
        chronos_df = compute_chronos_forecasts(client, daily, resume_from=resume_data)

        # Save partial progress for resume
        if not chronos_df.empty:
            if resume_data is not None:
                chronos_df = pd.concat([resume_data[chronos_df.columns], chronos_df])
                chronos_df = chronos_df[~chronos_df.index.duplicated(keep="last")]
            chronos_df.to_csv(PARTIAL_FILE)

    # Step 4: Merge and save
    result = merge_and_save(regime_df, chronos_df, daily)

    # Clean up partial file on success
    if PARTIAL_FILE.exists():
        PARTIAL_FILE.unlink()
        logger.info("Cleaned up partial progress file")

    logger.info(f"\nDone! PRISM features saved to: {OUTPUT_FILE}")
    logger.info(f"Columns: {list(result.columns)}")
    logger.info(f"Date range: {result.index[0].date()} → {result.index[-1].date()}")
    logger.info(f"Shape: {result.shape}")


if __name__ == "__main__":
    main()
