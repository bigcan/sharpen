"""
Breakeven accuracy analysis — the KEY decision metric.

For a binary directional predictor with accuracy p, average absolute return μ,
and round-trip cost c:
    Expected PnL per trade = μ * (2p - 1) - c
    Breakeven accuracy: p_break = (c/μ + 1) / 2

This tells us: how good does our model need to be on each asset?
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "cme"

ASSETS = {
    "CL": {
        "name": "Crude Oil",
        "file": "cl_2025_ohlcv_1min.parquet",
        "tick_size": 0.01, "multiplier": 1000,
        "commission_per_side": 2.50, "typical_spread_ticks": 1,
    },
    "GC": {
        "name": "Gold (active)",
        "file": "gc_2025_ohlcv_1min.parquet",
        "tick_size": 0.10, "multiplier": 100,
        "commission_per_side": 2.50, "typical_spread_ticks": 1,
    },
    "ES": {
        "name": "E-mini S&P 500",
        "file": "es_2025_ohlcv_1min.parquet",
        "tick_size": 0.25, "multiplier": 50,
        "commission_per_side": 2.50, "typical_spread_ticks": 1,
    },
}


def resample_5min(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    r = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['close'])
    r = r[r['volume'] > 0]
    return r.reset_index()


def compute_rt_cost_bps(price, info):
    notional = price * info['multiplier']
    spread_bps = (info['typical_spread_ticks'] * info['tick_size'] / price) * 10000
    commission_bps = (info['commission_per_side'] / notional) * 10000
    return spread_bps + 2 * commission_bps  # Full spread + 2x commission


def main():
    print("=" * 80)
    print("BREAKEVEN ACCURACY ANALYSIS — How Good Must Our Model Be?")
    print("=" * 80)
    print()
    print("Formula: p_break = (c/μ + 1) / 2")
    print("  where c = round-trip cost (bps), μ = avg absolute 5-min return (bps)")
    print("  p = directional accuracy (0.5 = random, 1.0 = perfect)")
    print()

    # Also compute for BTC reference
    print("=" * 80)
    print(f"{'Asset':>20s} | {'Avg|Ret|':>8s} | {'RT Cost':>8s} | {'c/μ':>6s} | "
          f"{'p_break':>7s} | {'Our RF':>7s} | {'Margin':>7s} | Verdict")
    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")

    # BTC reference (approximate from our data)
    # BTC 5-min avg absolute return ≈ 15 bps (from our experience)
    btc_mu = 15.0  # approximate
    for venue, cost_per_side in [("BTC Binance", 5.0), ("BTC Hyperliq", 2.5)]:
        c = cost_per_side * 2  # RT
        p_break = (c / btc_mu + 1) / 2
        rf_acc = 0.527  # Our best RF AUC on BTC
        margin = rf_acc - p_break
        verdict = "PROFITABLE" if margin > 0 else "FAIL" if margin < -0.02 else "MARGINAL"
        print(f"  {venue:>18s} | {btc_mu:>7.1f} | {c:>7.2f} | {c/btc_mu:>5.3f} | "
              f"{p_break:>6.3f} | {rf_acc:>6.3f} | {margin:>+6.3f} | {verdict}")

    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*20}")

    results = {}
    for sym, info in ASSETS.items():
        data_path = DATA_DIR / info['file']
        if not data_path.exists():
            continue

        df = pd.read_parquet(data_path)
        df_5min = resample_5min(df)

        closes = df_5min['close'].values

        # Compute returns for different horizons
        for horizon, h_label in [(1, "H1"), (3, "H3"), (6, "H6")]:
            returns_bps = []
            for i in range(len(closes) - horizon):
                ret = (closes[i + horizon] - closes[i]) / closes[i] * 10000
                returns_bps.append(ret)

            returns_bps = np.array(returns_bps)
            abs_returns = np.abs(returns_bps)

            # Use November (val) data for analysis
            months = pd.to_datetime(df_5min['timestamp']).dt.month.values
            val_mask = months[:len(returns_bps)] == 11
            test_mask = months[:len(returns_bps)] == 12

            for split_name, mask in [("val", val_mask), ("test", test_mask)]:
                split_abs = abs_returns[mask]
                split_closes = closes[:len(returns_bps)][mask]

                if len(split_abs) == 0:
                    continue

                mu = split_abs.mean()
                avg_price = split_closes.mean()
                c = compute_rt_cost_bps(avg_price, info)
                p_break = (c / mu + 1) / 2

                # Assume our RF could achieve similar AUC on this data (0.527 baseline)
                rf_acc = 0.527
                margin = rf_acc - p_break

                if split_name == "val" and horizon == 1:
                    verdict = "PROFITABLE" if margin > 0.01 else "MARGINAL" if margin > -0.01 else "FAIL"
                    results[sym] = {"mu": mu, "c": c, "p_break": p_break, "margin": margin}
                    print(f"  {info['name']+' '+h_label:>18s} | {mu:>7.1f} | {c:>7.2f} | {c/mu:>5.3f} | "
                          f"{p_break:>6.3f} | {rf_acc:>6.3f} | {margin:>+6.3f} | {verdict}")

            # Also show H3 and H6 for val only
            if horizon > 1:
                split_abs = abs_returns[val_mask]
                split_closes = closes[:len(returns_bps)][val_mask]
                if len(split_abs) == 0:
                    continue
                mu = split_abs.mean()
                avg_price = split_closes.mean()
                c = compute_rt_cost_bps(avg_price, info)
                p_break = (c / mu + 1) / 2
                margin = 0.527 - p_break
                verdict = "PROFITABLE" if margin > 0.01 else "MARGINAL" if margin > -0.01 else "FAIL"
                print(f"  {info['name']+' '+h_label:>18s} | {mu:>7.1f} | {c:>7.2f} | {c/mu:>5.3f} | "
                      f"{p_break:>6.3f} | {0.527:>6.3f} | {margin:>+6.3f} | {verdict}")

    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("""
  p_break = breakeven accuracy (0.50 = random, lower is better)
  Our RF  = best AUC achieved on BTC (0.527) — used as baseline assumption
  Margin  = Our RF - p_break (positive = profitable if same AUC transfers)

  CRITICAL FINDING:
  - BTC @ Binance needs 83.3% accuracy → IMPOSSIBLE with current signal (AUC 0.527)
  - BTC @ Hyperliquid needs 66.7% accuracy → still far above our capability
""")

    for sym, r in results.items():
        info = ASSETS[sym]
        print(f"  - {info['name']} needs {r['p_break']*100:.1f}% accuracy → "
              f"margin = {r['margin']*100:+.1f}pp from our baseline")

    print("""
  RECOMMENDATION:
  - Gold (GC) has the lowest breakeven threshold due to negligible fees
  - Even a modest directional signal (>50.4% at H1) would be profitable on Gold
  - CL requires more accuracy but is still dramatically easier than BTC
  - The question now: does Gold's MICROSTRUCTURE have predictable signal?
    → Need to fetch MBP-10 book data and run RF signal test
""")

    # Simulated PF at different accuracy levels
    print("=" * 80)
    print("SIMULATED PROFIT FACTOR at Different Accuracy Levels (H1, Val)")
    print("=" * 80)
    print(f"\n  {'Accuracy':>10s} | ", end="")
    for sym in ["CL", "GC", "ES"]:
        print(f"{'  '+ASSETS[sym]['name']:>20s} | ", end="")
    print(f"{'BTC Binance':>12s} | {'BTC HyperLiq':>12s}")
    print(f"  {'-'*10}-+-", end="")
    for _ in range(3):
        print(f"{'-'*20}-+-", end="")
    print(f"{'-'*12}-+-{'-'*12}")

    for acc in [0.51, 0.52, 0.527, 0.53, 0.54, 0.55, 0.60]:
        print(f"  {acc*100:>9.1f}% | ", end="")

        for sym in ["CL", "GC", "ES"]:
            info = ASSETS[sym]
            r = results.get(sym)
            if r:
                mu = r['mu']
                c = r['c']
                win_pnl = mu - c  # Avg winner PnL
                loss_pnl = mu + c  # Avg loser |PnL|
                gross_win = acc * win_pnl
                gross_loss = (1 - acc) * loss_pnl
                pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
                if pf > 0:
                    pf_str = f"PF {pf:.3f}"
                else:
                    pf_str = "NEGATIVE"
                print(f"{pf_str:>20s} | ", end="")
            else:
                print(f"{'N/A':>20s} | ", end="")

        # BTC
        btc_mu = 15.0
        for btc_c in [10.0, 5.0]:
            win_pnl = btc_mu - btc_c
            loss_pnl = btc_mu + btc_c
            gross_win = acc * win_pnl
            gross_loss = (1 - acc) * loss_pnl
            pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
            pf_str = f"PF {pf:.3f}" if pf > 0 else "NEGATIVE"
            print(f"{pf_str:>12s} | ", end="")
        print()


if __name__ == "__main__":
    main()
