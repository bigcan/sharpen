"""
Breakeven & Oracle Comparison: Bitfinex BTC (0 fees) vs Gold CME (0.35 bps)
============================================================================

Side-by-side analysis of:
  1. Return statistics (5-min bars, val/test splits)
  2. Breakeven accuracy at each venue's fee level
  3. Simulated profit factor at different accuracy levels
  4. Effective spread analysis (Bitfinex has no LOB data — estimated)
  5. RF signal quality comparison (using OHLCV-only features)

This script requires:
  - data/processed/btc_bitfinex_2025_full_year_5min.parquet (from prepare_bitfinex_5min.py)
  - data/processed/gc_2025_full_year_5min.parquet (from prepare_gc_5min.py)

Usage:
  python scripts/breakeven_bitfinex_vs_gold.py
  python scripts/breakeven_bitfinex_vs_gold.py --rf  # Also run RF signal comparison
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

VENUES = {
    "BTC Bitfinex (0bps)": {
        "file": "data/processed/btc_bitfinex_2025_full_year_5min.parquet",
        "taker_fee_bps": 0.0,   # Zero-fee model
        "maker_fee_bps": 0.0,
        "spread_cost_bps": 1.0, # Estimated effective spread (TBD from LOB)
        "symbol": "BTC",
        "val_month": 11,
        "test_month": 12,
    },
    "BTC Binance (5bps)": {
        "file": None,  # Use same data, just different fees
        "taker_fee_bps": 5.0,
        "maker_fee_bps": 2.0,
        "spread_cost_bps": 1.0,
        "symbol": "BTC",
        "val_month": 11,
        "test_month": 12,
    },
    "BTC Hyperliquid (2.5bps)": {
        "file": None,
        "taker_fee_bps": 2.5,
        "maker_fee_bps": 0.2,
        "spread_cost_bps": 1.0,
        "symbol": "BTC",
        "val_month": 11,
        "test_month": 12,
    },
    "Gold CME (0.35bps)": {
        "file": "data/processed/gc_2025_full_year_5min.parquet",
        "taker_fee_bps": 0.35,
        "maker_fee_bps": 0.10,
        "spread_cost_bps": 0.0,  # Included in LOB data
        "symbol": "GC",
        "val_month": 11,
        "test_month": 12,
    },
}


def load_data(file_path: str) -> pd.DataFrame | None:
    """Load processed 5-min parquet."""
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        return None
    df = pd.read_parquet(full_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def compute_return_stats(df: pd.DataFrame, month: int, horizons: list[int]) -> dict:
    """Compute return statistics for a given month split."""
    months = df['timestamp'].dt.month.values
    mask = months == month
    closes = df['close'].values

    stats = {}
    for h in horizons:
        rets_bps = np.full(len(closes), np.nan)
        for i in range(len(closes) - h):
            rets_bps[i] = (closes[i + h] - closes[i]) / closes[i] * 10000

        split_rets = rets_bps[mask[:len(rets_bps)]]
        split_rets = split_rets[~np.isnan(split_rets)]

        if len(split_rets) == 0:
            continue

        stats[f"H{h}"] = {
            "count": len(split_rets),
            "mean_abs_ret": np.abs(split_rets).mean(),
            "median_abs_ret": np.median(np.abs(split_rets)),
            "std": split_rets.std(),
            "skew": float(pd.Series(split_rets).skew()),
            "pct_positive": (split_rets > 0).mean() * 100,
        }
    return stats


def breakeven_table(venues: dict, btc_df: pd.DataFrame | None, gc_df: pd.DataFrame | None):
    """Print breakeven accuracy table for all venues."""
    print("\n" + "=" * 100)
    print("BREAKEVEN ACCURACY ANALYSIS — p_break = (c/μ + 1) / 2")
    print("=" * 100)
    print(f"  {'Venue':<30} | {'Split':>5} | {'H':>2} | {'μ (bps)':>8} | "
          f"{'RT Cost':>8} | {'p_break':>7} | {'RF AUC':>7} | {'Margin':>7} | Verdict")
    print("-" * 100)

    for venue_name, info in venues.items():
        # Select data source
        if "Gold" in venue_name:
            df = gc_df
            rf_auc = 0.554  # Our best Gold RF AUC (test)
        else:
            df = btc_df
            rf_auc = 0.528  # Our best BTC RF AUC (test)

        if df is None:
            print(f"  {venue_name:<30} | {'N/A — data not found':>60}")
            continue

        # Taker round-trip cost
        rt_cost_bps = info['taker_fee_bps'] * 2 + info['spread_cost_bps']

        for split_name, month in [("val", info['val_month']), ("test", info['test_month'])]:
            for horizon in [1, 3, 6]:
                stats = compute_return_stats(df, month, [horizon])
                key = f"H{horizon}"
                if key not in stats:
                    continue

                mu = stats[key]['mean_abs_ret']
                c = rt_cost_bps
                p_break = (c / mu + 1) / 2 if mu > 0 else 1.0
                margin = rf_auc - p_break

                if margin > 0.02:
                    verdict = "PROFITABLE"
                elif margin > 0:
                    verdict = "MARGINAL"
                elif margin > -0.02:
                    verdict = "TIGHT"
                else:
                    verdict = "FAIL"

                print(f"  {venue_name:<30} | {split_name:>5} | {horizon:>2} | {mu:>7.2f} | "
                      f"{c:>7.2f} | {p_break:>6.3f} | {rf_auc:>6.3f} | {margin:>+6.3f} | {verdict}")


def simulated_pf_table(btc_df: pd.DataFrame | None, gc_df: pd.DataFrame | None):
    """Print simulated PF at different accuracy levels for key venues."""
    print("\n" + "=" * 100)
    print("SIMULATED PROFIT FACTOR at Different Accuracy Levels (H1, Nov val)")
    print("=" * 100)

    scenarios = []
    for df, label, rt_cost in [
        (btc_df, "BTC Bitfinex 0bps", 1.0),   # spread only
        (btc_df, "BTC Binance 5bps", 11.0),    # 5*2 + 1 spread
        (btc_df, "BTC Hyperliq 2.5bps", 6.0),  # 2.5*2 + 1 spread
        (gc_df, "Gold CME 0.35bps", 0.7),       # 0.35*2, spread in data
    ]:
        if df is None:
            continue
        stats = compute_return_stats(df, 11, [1])
        if "H1" not in stats:
            continue
        mu = stats["H1"]["mean_abs_ret"]
        scenarios.append((label, mu, rt_cost))

    if not scenarios:
        print("  No data available for simulation")
        return

    header = f"  {'Accuracy':>10} |"
    for label, _, _ in scenarios:
        header += f" {label:>22} |"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for acc in [0.505, 0.51, 0.515, 0.52, 0.527, 0.53, 0.54, 0.55, 0.60]:
        row = f"  {acc*100:>9.1f}% |"
        for label, mu, c in scenarios:
            win_pnl = mu - c
            loss_pnl = mu + c
            gross_win = acc * win_pnl
            gross_loss = (1 - acc) * loss_pnl
            if gross_loss > 0 and gross_win > 0:
                pf = gross_win / gross_loss
                marker = " **" if pf >= 1.0 else ""
                row += f" {'PF':>4} {pf:>5.3f}{marker:>10} |"
            elif gross_win <= 0:
                row += f" {'NEGATIVE':>22} |"
            else:
                row += f" {'INF':>22} |"
        print(row)


def spread_sensitivity(btc_df: pd.DataFrame | None):
    """Show how Bitfinex BTC profitability depends on effective spread."""
    if btc_df is None:
        return

    print("\n" + "=" * 100)
    print("SPREAD SENSITIVITY — Bitfinex BTC (0 commission) at different effective spreads")
    print("=" * 100)
    print("  NOTE: Trading fee is 0%, but effective cost = spread traversal per trade.")
    print("  Need LOB data to measure actual spread. This shows sensitivity to spread assumption.")
    print()

    stats = compute_return_stats(btc_df, 11, [1])
    if "H1" not in stats:
        print("  No H1 data available")
        return

    mu = stats["H1"]["mean_abs_ret"]
    rf_auc = 0.528

    print(f"  BTC 5-min avg |return| (Nov val): {mu:.2f} bps")
    print(f"  Our RF AUC: {rf_auc:.3f}")
    print()
    print(f"  {'Eff Spread':>12} | {'RT Cost':>8} | {'p_break':>7} | {'Margin':>7} | {'Sim PF':>7} | Verdict")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")

    for spread_bps in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        c = spread_bps  # Zero commission, cost = spread only
        p_break = (c / mu + 1) / 2 if mu > 0 else 1.0
        margin = rf_auc - p_break

        # Simulated PF at our RF accuracy
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

        print(f"  {spread_bps:>11.1f} | {c:>7.2f} | {p_break:>6.3f} | "
              f"{margin:>+6.3f} | {pf_str:>7} | {verdict}")


def rf_signal_comparison(btc_df: pd.DataFrame | None, gc_df: pd.DataFrame | None):
    """Run quick RF signal test on OHLCV-only features (no LOB).

    This gives a lower bound on signal quality — LOB features boost AUC by ~11%.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("\n[RF] sklearn not available — skipping RF comparison")
        return

    print("\n" + "=" * 100)
    print("RF SIGNAL COMPARISON — OHLCV-only features (no LOB)")
    print("=" * 100)
    print("  NOTE: This is a lower bound. LOB features add ~11% AUC on Gold.")

    for label, df, val_month, test_month in [
        ("BTC (Bitfinex data)", btc_df, 11, 12),
        ("Gold (GC CME)", gc_df, 11, 12),
    ]:
        if df is None:
            print(f"\n  {label}: DATA NOT FOUND — skipping")
            continue

        print(f"\n  {label}:")
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        months = df['timestamp'].dt.month.values

        # Build simple OHLCV features
        features = []
        for i in range(20, len(closes)):
            f = []
            # Returns at various lags
            for lag in [1, 2, 3, 5, 10, 20]:
                if i - lag >= 0:
                    f.append((closes[i] - closes[i-lag]) / closes[i-lag] * 10000)
                else:
                    f.append(0)
            # Volatility (std of 1-bar returns over window)
            if i >= 20:
                rets = np.diff(closes[i-20:i+1]) / closes[i-19:i+1]
                f.append(rets.std() * 10000)
            else:
                f.append(0)
            # Volume features
            f.append(np.log1p(volumes[i]))
            vol_ma = volumes[max(0,i-20):i+1].mean()
            f.append(volumes[i] / (vol_ma + 1e-9))
            # Bar range
            f.append((highs[i] - lows[i]) / closes[i] * 10000)
            features.append(f)

        features = np.array(features)
        offset = 20

        # Target: direction at H1
        target = np.zeros(len(features))
        for i in range(len(features) - 1):
            idx = i + offset
            if idx + 1 < len(closes):
                target[i] = 1 if closes[idx + 1] > closes[idx] else 0

        # Split
        feature_months = months[offset:offset + len(features)]
        train_mask = feature_months <= 10
        val_mask = feature_months == val_month
        test_mask = feature_months == test_month

        X_train, y_train = features[train_mask], target[train_mask]
        X_val, y_val = features[val_mask], target[val_mask]
        X_test, y_test = features[test_mask], target[test_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"    Insufficient data for train/val split")
            continue

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Evaluate
        val_proba = rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)

        test_proba = rf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

        print(f"    Train samples: {len(X_train):,}")
        print(f"    Val samples:   {len(X_val):,}")
        print(f"    Test samples:  {len(X_test):,}")
        print(f"    Val AUC:  {val_auc:.4f}")
        print(f"    Test AUC: {test_auc:.4f}")

        # With LOB features, Gold AUC was 0.6147/0.5542
        # This OHLCV-only baseline shows how much LOB adds
        print(f"    (For reference: Gold with LOB features achieved 0.6147/0.5542)")


def main():
    parser = argparse.ArgumentParser(
        description="Breakeven comparison: Bitfinex BTC vs Gold CME")
    parser.add_argument("--rf", action="store_true",
                        help="Also run RF signal comparison (slower)")
    args = parser.parse_args()

    print("=" * 100)
    print("VENUE COMPARISON: Bitfinex BTC (0 fees) vs Gold CME (0.35 bps)")
    print("=" * 100)

    # Load data
    btc_df = load_data("data/processed/btc_bitfinex_2025_full_year_5min.parquet")
    gc_df = load_data("data/processed/gc_2025_full_year_5min.parquet")

    if btc_df is not None:
        print(f"\n  BTC (Bitfinex): {len(btc_df):,} bars, "
              f"{btc_df['timestamp'].min()} → {btc_df['timestamp'].max()}")
    else:
        print("\n  BTC (Bitfinex): NOT FOUND — run fetch_bitfinex_btc_perp.py + prepare_bitfinex_5min.py first")

    if gc_df is not None:
        print(f"  Gold (CME):     {len(gc_df):,} bars, "
              f"{gc_df['timestamp'].min()} → {gc_df['timestamp'].max()}")
    else:
        print("  Gold (CME):     NOT FOUND — need gc_2025_full_year_5min.parquet")

    if btc_df is None and gc_df is None:
        print("\nERROR: No data available. Run data preparation scripts first.")
        sys.exit(1)

    # Analysis
    breakeven_table(VENUES, btc_df, gc_df)
    simulated_pf_table(btc_df, gc_df)
    spread_sensitivity(btc_df)

    if args.rf:
        rf_signal_comparison(btc_df, gc_df)

    # Strategic summary
    print("\n" + "=" * 100)
    print("STRATEGIC SUMMARY")
    print("=" * 100)
    print("""
  KEY INSIGHT: At zero fees, the only trading cost is SPREAD.

  If Bitfinex BTC perp effective spread < 2 bps:
    → BTC becomes viable with our existing 52.8% AUC signal
    → p_break drops to ~50.0-50.7% (vs 83.3% at Binance, 66.7% at Hyperliquid)

  But Gold STILL has advantages:
    → Stronger LOB signal: 55.4% AUC (with LOB) vs 52.8% (BTC, OHLCV+LOB)
    → Truly negligible spread (0.7 bps on even months)
    → Known profitable oracle gate: PF 1.949/2.147

  RECOMMENDED STRATEGY:
    1. Gold (G5 BDQ) — EXECUTE NOW (validated pipeline, strong signal)
    2. Bitfinex BTC — PARALLEL INVESTIGATION:
       a. Collect 24h of LOB snapshots via WebSocket (measure real spread)
       b. If spread < 2 bps → run oracle gate through env
       c. If oracle PF > 1.5 → deploy BDQ
    3. Lighter DEX — DEPRIORITIZED (zero fees but 300ms latency, no LOB history)
""")


if __name__ == "__main__":
    main()
