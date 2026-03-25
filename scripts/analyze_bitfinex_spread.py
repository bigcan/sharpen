"""
Analyze Bitfinex LOB Spread Data — Go/No-Go Verdict for BTC Trading.
=====================================================================

Reads recorded LOB snapshots from record_bitfinex_lob.py and produces:
  1. Overall spread statistics (mean, median, percentiles)
  2. Hourly spread profile (UTC) — identify best/worst trading hours
  3. Spread stability analysis (std, autocorrelation)
  4. Depth analysis (liquidity at top 5/25 levels)
  5. Breakeven recalculation using MEASURED spread (not estimated)
  6. Go/No-Go verdict for Bitfinex BTC at zero fees

Usage:
  python scripts/analyze_bitfinex_spread.py
  python scripts/analyze_bitfinex_spread.py --file data/bitfinex/lob_snapshots_20260223_120000_all.parquet
  python scripts/analyze_bitfinex_spread.py --latest   # Auto-find latest recording
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "bitfinex"


def find_latest_recording() -> Path | None:
    """Find the most recent LOB snapshot file."""
    if not DATA_DIR.exists():
        return None
    files = sorted(DATA_DIR.glob("lob_snapshots_*_all.parquet"), reverse=True)
    return files[0] if files else None


def load_snapshots(path: Path) -> pd.DataFrame:
    """Load LOB snapshot data."""
    print(f"[SPREAD] Loading {path}")
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    print(f"[SPREAD] Loaded {len(df):,} snapshots")
    print(f"[SPREAD] Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    duration_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    print(f"[SPREAD] Duration: {duration_hours:.2f} hours")
    return df


def overall_stats(df: pd.DataFrame):
    """Print comprehensive spread statistics."""
    s = df['spread_bps']

    print(f"\n{'='*70}")
    print(f"SPREAD STATISTICS ({len(df):,} observations)")
    print(f"{'='*70}")

    print("\n  Distribution:")
    print(f"    Mean:    {s.mean():.4f} bps")
    print(f"    Median:  {s.median():.4f} bps")
    print(f"    Mode:    {s.mode().iloc[0] if len(s.mode()) > 0 else 'N/A':.4f} bps")
    print(f"    Std:     {s.std():.4f} bps")
    print(f"    Skew:    {s.skew():.2f}")
    print(f"    Kurt:    {s.kurtosis():.2f}")

    print("\n  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = s.quantile(p / 100)
        print(f"    P{p:>2}: {val:.4f} bps")

    print("\n  Extremes:")
    print(f"    Min:     {s.min():.4f} bps")
    print(f"    Max:     {s.max():.4f} bps")

    # Fraction of time at different spread levels
    print("\n  Time at Spread Level:")
    for threshold in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        frac = (s <= threshold).mean() * 100
        print(f"    <= {threshold:>4.2f} bps: {frac:>6.1f}% of time")


def hourly_profile(df: pd.DataFrame):
    """Spread profile by hour (UTC)."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    print(f"\n{'='*70}")
    print("HOURLY SPREAD PROFILE (UTC)")
    print(f"{'='*70}")
    print(f"  {'Hour':>4} | {'Mean':>7} | {'Median':>7} | {'P95':>7} | {'Count':>7} | "
          f"{'BidDepth5':>9} | {'AskDepth5':>9}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}")

    hourly = df.groupby('hour').agg({
        'spread_bps': ['mean', 'median', lambda x: x.quantile(0.95), 'count'],
        'bid_depth_5': 'mean',
        'ask_depth_5': 'mean',
    })

    for hour in range(24):
        if hour not in hourly.index:
            continue
        row = hourly.loc[hour]
        mean = row[('spread_bps', 'mean')]
        median = row[('spread_bps', 'median')]
        p95 = row[('spread_bps', '<lambda_0>')]
        count = int(row[('spread_bps', 'count')])
        bd5 = row[('bid_depth_5', 'mean')]
        ad5 = row[('ask_depth_5', 'mean')]
        print(f"  {hour:>4} | {mean:>6.3f} | {median:>6.3f} | {p95:>6.3f} | {count:>7,} | "
              f"{bd5:>8.3f} | {ad5:>8.3f}")

    # Best/worst hours
    best_hour = hourly[('spread_bps', 'median')].idxmin()
    worst_hour = hourly[('spread_bps', 'median')].idxmax()
    print(f"\n  Best hour:  {best_hour:02d}:00 UTC "
          f"(median {hourly.loc[best_hour, ('spread_bps', 'median')]:.3f} bps)")
    print(f"  Worst hour: {worst_hour:02d}:00 UTC "
          f"(median {hourly.loc[worst_hour, ('spread_bps', 'median')]:.3f} bps)")


def depth_analysis(df: pd.DataFrame):
    """Analyze order book depth (liquidity)."""
    print(f"\n{'='*70}")
    print("DEPTH ANALYSIS (Liquidity)")
    print(f"{'='*70}")

    for col, label in [('bid_depth_5', 'Bid Top-5'),
                        ('ask_depth_5', 'Ask Top-5'),
                        ('bid_depth_25', 'Bid Top-25'),
                        ('ask_depth_25', 'Ask Top-25')]:
        if col not in df.columns:
            continue
        vals = df[col]
        print(f"\n  {label} (BTC):")
        print(f"    Mean:   {vals.mean():.3f}")
        print(f"    Median: {vals.median():.3f}")
        print(f"    P5:     {vals.quantile(0.05):.3f}")
        print(f"    P95:    {vals.quantile(0.95):.3f}")

    # Bid-ask depth imbalance
    if 'bid_depth_5' in df.columns and 'ask_depth_5' in df.columns:
        imbalance = (df['bid_depth_5'] - df['ask_depth_5']) / (df['bid_depth_5'] + df['ask_depth_5'] + 1e-9)
        print("\n  Depth Imbalance (bid-ask / total):")
        print(f"    Mean:   {imbalance.mean():+.4f}")
        print(f"    Std:    {imbalance.std():.4f}")
        print(f"    Skew:   {imbalance.skew():.2f}")


def breakeven_with_measured_spread(df: pd.DataFrame):
    """Recalculate breakeven using MEASURED spread instead of estimated."""
    print(f"\n{'='*70}")
    print("BREAKEVEN RECALCULATION WITH MEASURED SPREAD")
    print(f"{'='*70}")

    measured_median = df['spread_bps'].median()
    measured_p75 = df['spread_bps'].quantile(0.75)
    measured_p95 = df['spread_bps'].quantile(0.95)

    # Load BTC 5-min returns for mu calculation
    btc_5min_path = PROJECT_ROOT / "data" / "processed" / "btc_bitfinex_2025_full_year_5min.parquet"
    if btc_5min_path.exists():
        btc_df = pd.read_parquet(btc_5min_path)
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
        # November returns (val split)
        nov_mask = btc_df['timestamp'].dt.month == 11
        closes = btc_df.loc[nov_mask, 'close'].values
        returns_bps = np.abs(np.diff(closes) / closes[:-1]) * 10000
        mu = returns_bps.mean()
        mu_source = "Nov 2025 val split"
    else:
        mu = 11.0  # Fallback from earlier analysis
        mu_source = "estimated (no data file)"

    rf_auc = 0.528  # Our BTC RF test AUC

    print("\n  Inputs:")
    print(f"    5-min avg |return| (μ): {mu:.2f} bps ({mu_source})")
    print(f"    Our RF AUC:             {rf_auc:.3f}")
    print("    Trading fee:            0.00 bps (Bitfinex zero-fee)")
    print(f"    Measured median spread:  {measured_median:.4f} bps")
    print(f"    Measured P75 spread:     {measured_p75:.4f} bps")
    print(f"    Measured P95 spread:     {measured_p95:.4f} bps")

    print(f"\n  {'Spread Assumption':<25} | {'RT Cost':>8} | {'p_break':>7} | "
          f"{'Margin':>7} | {'Sim PF':>7} | Verdict")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")

    for label, spread in [
        ("Zero spread (ideal)", 0.0),
        (f"Median ({measured_median:.3f} bps)", measured_median),
        (f"P75 ({measured_p75:.3f} bps)", measured_p75),
        (f"P95 ({measured_p95:.3f} bps)", measured_p95),
        ("Old estimate (1.0 bps)", 1.0),
    ]:
        c = spread  # Zero fees → cost = spread only
        p_break = (c / mu + 1) / 2 if mu > 0 else 1.0
        margin = rf_auc - p_break

        # Simulated PF
        win_pnl = mu - c
        loss_pnl = mu + c
        gross_win = rf_auc * win_pnl
        gross_loss = (1 - rf_auc) * loss_pnl
        if gross_loss > 0 and gross_win > 0:
            pf = gross_win / gross_loss
            pf_str = f"{pf:.3f}"
        elif gross_win <= 0:
            pf_str = "NEG"
        else:
            pf_str = "INF"

        if margin > 0.02:
            verdict = "PROFITABLE"
        elif margin > 0:
            verdict = "MARGINAL"
        elif margin > -0.02:
            verdict = "TIGHT"
        else:
            verdict = "FAIL"

        print(f"  {label:<25} | {c:>7.3f} | {p_break:>6.4f} | "
              f"{margin:>+6.4f} | {pf_str:>7} | {verdict}")

    # Also compare to Gold
    print("\n  For comparison — Gold CME:")
    print("    Measured spread: ~0.7 bps (from LOB data)")
    print("    RF AUC: 0.554 (with LOB features)")
    print("    μ (Nov): 10.52 bps")
    gc_c = 0.7
    gc_mu = 10.52
    gc_p_break = (gc_c / gc_mu + 1) / 2
    gc_margin = 0.554 - gc_p_break
    print(f"    p_break: {gc_p_break:.4f}, margin: {gc_margin:+.4f} → PROFITABLE")


def go_nogo_verdict(df: pd.DataFrame):
    """Final go/no-go verdict."""
    median_spread = df['spread_bps'].median()
    p75_spread = df['spread_bps'].quantile(0.75)

    print(f"\n{'='*70}")
    print("GO / NO-GO VERDICT")
    print(f"{'='*70}")

    if median_spread < 0.3:
        verdict = "STRONG GO"
        detail = (f"Median spread {median_spread:.3f} bps is negligible. "
                  f"Bitfinex BTC at zero fees has comparable economics to Gold CME. "
                  f"Proceed to oracle gate through env.")
    elif median_spread < 0.5:
        verdict = "GO"
        detail = (f"Median spread {median_spread:.3f} bps is low. "
                  f"BTC is viable at zero fees. Run oracle gate to confirm. "
                  f"Gold still has stronger signal but BTC has 24/7 uptime advantage.")
    elif median_spread < 1.0:
        verdict = "CONDITIONAL GO"
        detail = (f"Median spread {median_spread:.3f} bps is moderate. "
                  f"BTC is marginal at H1 but viable at H3+. "
                  f"Need LOB features to boost AUC above breakeven. "
                  f"Gold remains the safer primary path.")
    elif median_spread < 2.0:
        verdict = "WEAK NO-GO"
        detail = (f"Median spread {median_spread:.3f} bps makes H1 unprofitable. "
                  f"Only viable at H6+ with LOB feature boost. "
                  f"Gold is clearly superior. BTC deprioritized.")
    else:
        verdict = "NO-GO"
        detail = (f"Median spread {median_spread:.3f} bps is too high. "
                  f"Despite zero trading fees, spread cost makes BTC unviable. "
                  f"Focus entirely on Gold.")

    print(f"\n  Measured median spread: {median_spread:.4f} bps")
    print(f"  Measured P75 spread:   {p75_spread:.4f} bps")
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  {detail}")

    # Action items
    print("\n  NEXT STEPS:")
    if "GO" in verdict:
        print("    1. Run oracle gate: python scripts/oracle_gate_gc.py "
              "--config configs/phase_h_bitfinex_btc_dev.yaml --split val")
        print("    2. If oracle PF > 1.5 → deploy BDQ training")
        print("    3. Start long-term LOB recording for feature engineering")
    else:
        print("    1. Focus on Gold G5 BDQ deployment")
        print("    2. Consider Bitfinex only if LOB features significantly boost BTC AUC")
        print("    3. Investigate whether spread narrows at high-volume hours")


def main():
    parser = argparse.ArgumentParser(description="Analyze Bitfinex LOB spread data")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to LOB snapshot parquet")
    parser.add_argument("--latest", action="store_true",
                        help="Auto-find latest recording")
    args = parser.parse_args()

    # Find data file
    if args.file:
        path = Path(args.file)
    elif args.latest:
        path = find_latest_recording()
    else:
        path = find_latest_recording()

    if path is None or not path.exists():
        print("[SPREAD] ERROR: No LOB snapshot data found.")
        print("[SPREAD] Run: python scripts/record_bitfinex_lob.py --duration 0.5")
        print("[SPREAD] (Records for 30 minutes as a quick test)")
        sys.exit(1)

    df = load_snapshots(path)

    if len(df) < 10:
        print(f"[SPREAD] ERROR: Only {len(df)} snapshots — need more data (at least 1 hour).")
        sys.exit(1)

    overall_stats(df)
    hourly_profile(df)
    depth_analysis(df)
    breakeven_with_measured_spread(df)
    go_nogo_verdict(df)


if __name__ == "__main__":
    main()
