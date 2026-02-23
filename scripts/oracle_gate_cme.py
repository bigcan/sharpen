"""
Oracle Gate for CME Futures — compares signal ceiling across assets.

Methodology (matches BTC oracle gate):
- Resample to 5-min bars
- Perfect foresight: knows direction of next H bars
- Taker execution: pays spread + commission on entry and exit
- Metrics: Profit Factor, Sharpe, trades, max drawdown
- Split: Jan-Oct train, Nov val, Dec test (matches BTC)

Usage:
    python scripts/oracle_gate_cme.py
    python scripts/oracle_gate_cme.py --horizon 1 3 6 12
    python scripts/oracle_gate_cme.py --assets CL GC
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass


DATA_DIR = Path(__file__).parent.parent / "data" / "cme"

# Asset specifications
ASSETS = {
    "CL": {
        "name": "Crude Oil",
        "file": "cl_2025_ohlcv_1min.parquet",
        "tick_size": 0.01,
        "multiplier": 1000,
        "commission_per_side": 2.50,
        "typical_spread_ticks": 1,
    },
    "GC": {
        "name": "Gold (active)",
        "file": "gc_2025_ohlcv_1min.parquet",
        "tick_size": 0.10,
        "multiplier": 100,
        "commission_per_side": 2.50,
        "typical_spread_ticks": 1,
    },
    "ES": {
        "name": "E-mini S&P 500",
        "file": "es_2025_ohlcv_1min.parquet",
        "tick_size": 0.25,
        "multiplier": 50,
        "commission_per_side": 2.50,
        "typical_spread_ticks": 1,
    },
}

# BTC reference numbers from core.md
BTC_REFERENCE = {
    "oracle_pf_5min_h1_val": 1.329,
    "oracle_pf_5min_h1_test": 1.319,
    "best_agent_pf_val": 0.97,   # B15 best
    "best_agent_pf_test": 0.94,
    "best_supervised_pf_val": 1.010,  # RF-RAW Hyperliquid
    "best_supervised_pf_test": 1.017,
    "taker_cost_binance_bps": 5.0,
    "taker_cost_hyperliquid_bps": 2.5,
}


@dataclass
class OracleResult:
    asset: str
    horizon: int
    split: str
    pf: float
    sharpe: float
    trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_profit_bps: float
    avg_loss_bps: float
    cost_per_trade_bps: float
    bars: int


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 5-min bars."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['close'])

    # Drop bars with zero volume (no trading)
    resampled = resampled[resampled['volume'] > 0]

    return resampled.reset_index()


def compute_taker_cost_bps(price: float, info: dict) -> float:
    """Compute one-way taker cost in bps."""
    notional = price * info['multiplier']
    spread_bps = (info['typical_spread_ticks'] * info['tick_size'] / price) * 10000
    commission_bps = (info['commission_per_side'] / notional) * 10000
    return spread_bps / 2 + commission_bps  # Half spread + commission per side


def run_oracle(df: pd.DataFrame, info: dict, horizon: int = 1) -> list:
    """
    Run oracle gate on 5-min data.

    Oracle strategy:
    - Look H bars ahead
    - If future_close > current_close: go long (taker entry + taker exit)
    - If future_close < current_close: go short
    - PnL = |price_change| - round_trip_cost (for correct direction)
    - PnL = -|price_change| - round_trip_cost (would never happen with oracle, but included for completeness)
    """
    closes = df['close'].values
    timestamps = df['timestamp'].values
    n = len(closes)

    # Compute future returns
    future_returns_bps = np.full(n, np.nan)
    for i in range(n - horizon):
        future_returns_bps[i] = (closes[i + horizon] - closes[i]) / closes[i] * 10000

    # Split data: Jan-Oct train, Nov val, Dec test
    months = pd.to_datetime(df['timestamp']).dt.month.values

    splits = {
        'train': (months >= 1) & (months <= 10),
        'val': months == 11,
        'test': months == 12,
    }

    results = []
    for split_name, mask in splits.items():
        split_returns = future_returns_bps[mask]
        split_closes = closes[mask]

        # Remove NaN (last H bars of each split)
        valid = ~np.isnan(split_returns)
        split_returns = split_returns[valid]
        split_closes = split_closes[valid]

        if len(split_returns) == 0:
            continue

        # Oracle knows direction, only trades when profitable after costs
        trade_pnls = []
        for i in range(len(split_returns)):
            ret = split_returns[i]
            price = split_closes[i]

            # Round-trip taker cost (entry + exit)
            rt_cost_bps = 2 * compute_taker_cost_bps(price, info)

            abs_ret = abs(ret)

            if abs_ret > rt_cost_bps:
                # Oracle trades: direction is always correct, net PnL = |return| - cost
                pnl = abs_ret - rt_cost_bps
                trade_pnls.append(pnl)
            elif abs_ret > 0:
                # Oracle skips: move is too small to cover costs
                pass
            # else: no move, skip

        trade_pnls = np.array(trade_pnls)

        if len(trade_pnls) == 0:
            results.append(OracleResult(
                asset=info['name'], horizon=horizon, split=split_name,
                pf=0.0, sharpe=0.0, trades=0, win_rate=0.0,
                total_return_pct=0.0, max_drawdown_pct=0.0,
                avg_profit_bps=0.0, avg_loss_bps=0.0,
                cost_per_trade_bps=0.0, bars=int(mask.sum()),
            ))
            continue

        # All oracle trades are profitable by construction (it only trades when |ret| > cost)
        # But let's also compute what happens WITHOUT the cost filter (always trade)
        # to see the raw directional alpha

        # With cost filter: all trades are winners
        gross_profit = trade_pnls[trade_pnls > 0].sum()
        gross_loss = abs(trade_pnls[trade_pnls < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Compute cumulative returns for drawdown
        cum_returns = np.cumsum(trade_pnls) / 10000  # Convert bps to fraction
        peak = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - peak
        max_dd = drawdown.min() * 100  # As percentage

        # Sharpe (annualized, assuming ~230 trading days * ~46 five-min bars per day)
        bars_per_year = 230 * (1380 // 5)  # Approx
        if trade_pnls.std() > 0:
            sharpe = (trade_pnls.mean() / trade_pnls.std()) * np.sqrt(bars_per_year)
        else:
            sharpe = float('inf')

        avg_cost = 2 * compute_taker_cost_bps(split_closes.mean(), info)

        results.append(OracleResult(
            asset=info['name'],
            horizon=horizon,
            split=split_name,
            pf=pf,
            sharpe=round(sharpe, 2),
            trades=len(trade_pnls),
            win_rate=100.0,  # Oracle with cost filter = 100% win
            total_return_pct=round(cum_returns[-1] * 100, 2),
            max_drawdown_pct=round(max_dd, 2),
            avg_profit_bps=round(trade_pnls.mean(), 2),
            avg_loss_bps=0.0,
            cost_per_trade_bps=round(avg_cost, 2),
            bars=int(mask.sum()),
        ))

    return results


def run_realistic_oracle(df: pd.DataFrame, info: dict, horizon: int = 1) -> list:
    """
    More realistic oracle: trades EVERY bar (like BTC oracle gate).

    - Always takes a position based on future direction
    - Pays taker cost regardless
    - PnL = signed_return - round_trip_cost
    - This gives Profit Factor comparable to BTC oracle gate numbers
    """
    closes = df['close'].values
    n = len(closes)

    future_returns_bps = np.full(n, np.nan)
    for i in range(n - horizon):
        future_returns_bps[i] = (closes[i + horizon] - closes[i]) / closes[i] * 10000

    months = pd.to_datetime(df['timestamp']).dt.month.values
    splits = {
        'val': months == 11,
        'test': months == 12,
    }

    results = []
    for split_name, mask in splits.items():
        split_returns = future_returns_bps[mask]
        split_closes = closes[mask]

        valid = ~np.isnan(split_returns)
        split_returns = split_returns[valid]
        split_closes = split_closes[valid]

        if len(split_returns) == 0:
            continue

        # Oracle trades every bar: direction always correct
        trade_pnls = []
        for i in range(len(split_returns)):
            ret = split_returns[i]
            price = split_closes[i]
            rt_cost_bps = 2 * compute_taker_cost_bps(price, info)

            if abs(ret) < 1e-10:
                # No movement, skip
                continue

            # Oracle picks correct direction, but pays cost
            pnl = abs(ret) - rt_cost_bps
            trade_pnls.append(pnl)

        trade_pnls = np.array(trade_pnls)
        if len(trade_pnls) == 0:
            continue

        winners = trade_pnls[trade_pnls > 0]
        losers = trade_pnls[trade_pnls < 0]

        gross_profit = winners.sum() if len(winners) > 0 else 0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-10
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        win_rate = len(winners) / len(trade_pnls) * 100

        # Cumulative returns
        cum_returns = np.cumsum(trade_pnls) / 10000
        peak = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - peak
        max_dd = drawdown.min() * 100

        bars_per_year = 230 * (1380 // 5)
        if trade_pnls.std() > 0:
            sharpe = (trade_pnls.mean() / trade_pnls.std()) * np.sqrt(bars_per_year)
        else:
            sharpe = float('inf')

        avg_cost = 2 * compute_taker_cost_bps(split_closes.mean(), info)

        results.append(OracleResult(
            asset=info['name'],
            horizon=horizon,
            split=split_name,
            pf=round(pf, 3),
            sharpe=round(sharpe, 2),
            trades=len(trade_pnls),
            win_rate=round(win_rate, 2),
            total_return_pct=round(cum_returns[-1] * 100, 2),
            max_drawdown_pct=round(max_dd, 2),
            avg_profit_bps=round(winners.mean(), 2) if len(winners) > 0 else 0.0,
            avg_loss_bps=round(losers.mean(), 2) if len(losers) > 0 else 0.0,
            cost_per_trade_bps=round(avg_cost, 2),
            bars=int(mask.sum()),
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Oracle Gate for CME Futures")
    parser.add_argument("--assets", nargs="+", default=["CL", "GC", "ES"],
                        choices=list(ASSETS.keys()))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12],
                        help="Lookahead horizons in 5-min bars (H1=5min, H3=15min, etc.)")
    args = parser.parse_args()

    print("=" * 80)
    print("ORACLE GATE — CME Futures vs BTC Benchmark")
    print("Methodology: Perfect foresight, taker execution, 5-min bars")
    print("Split: Jan-Oct train, Nov val, Dec test")
    print("=" * 80)

    # BTC reference
    print(f"\n{'='*80}")
    print("BTC REFERENCE (from existing research)")
    print(f"{'='*80}")
    print(f"  Oracle H1 5-min:  Val PF={BTC_REFERENCE['oracle_pf_5min_h1_val']:.3f}  |  "
          f"Test PF={BTC_REFERENCE['oracle_pf_5min_h1_test']:.3f}")
    print(f"  Taker cost:       Binance={BTC_REFERENCE['taker_cost_binance_bps']:.1f} bps  |  "
          f"Hyperliquid={BTC_REFERENCE['taker_cost_hyperliquid_bps']:.1f} bps")
    print(f"  Best RL agent:    Val PF={BTC_REFERENCE['best_agent_pf_val']:.3f}  |  "
          f"Test PF={BTC_REFERENCE['best_agent_pf_test']:.3f}")
    print(f"  Best supervised:  Val PF={BTC_REFERENCE['best_supervised_pf_val']:.3f}  |  "
          f"Test PF={BTC_REFERENCE['best_supervised_pf_test']:.3f}")

    all_results = []

    for asset_key in args.assets:
        info = ASSETS[asset_key]
        data_path = DATA_DIR / info['file']

        if not data_path.exists():
            print(f"\n  SKIP {asset_key}: {data_path} not found")
            continue

        print(f"\n{'='*80}")
        print(f"{info['name']} ({asset_key})")
        print(f"{'='*80}")

        # Load and resample
        df = pd.read_parquet(data_path)
        df_5min = resample_to_5min(df)
        print(f"  1-min bars: {len(df):,}  →  5-min bars: {len(df_5min):,}")

        months = pd.to_datetime(df_5min['timestamp']).dt.month.values
        val_bars = (months == 11).sum()
        test_bars = (months == 12).sum()
        print(f"  Val (Nov): {val_bars:,} bars  |  Test (Dec): {test_bars:,} bars")

        # Compute average cost
        avg_price = df_5min['close'].mean()
        one_way_cost = compute_taker_cost_bps(avg_price, info)
        print(f"  Avg price: ${avg_price:.2f}  |  One-way taker cost: {one_way_cost:.2f} bps")

        # Run realistic oracle (trades every bar, comparable to BTC oracle)
        print(f"\n  --- Realistic Oracle (trades every bar, pays costs) ---")
        print(f"  {'H':>3s} | {'Split':>5s} | {'PF':>7s} | {'Sharpe':>8s} | {'Trades':>7s} | "
              f"{'WinRate':>7s} | {'Return%':>8s} | {'MaxDD%':>7s} | {'AvgWin':>7s} | {'AvgLoss':>8s} | {'Cost':>5s}")
        print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-"
              f"{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*5}")

        for h in args.horizons:
            results = run_realistic_oracle(df_5min, info, horizon=h)
            all_results.extend(results)

            for r in results:
                pf_str = f"{r.pf:.3f}" if r.pf < 100 else "INF"
                print(f"  H{h:>2d} | {r.split:>5s} | {pf_str:>7s} | {r.sharpe:>8.1f} | "
                      f"{r.trades:>7,d} | {r.win_rate:>6.1f}% | {r.total_return_pct:>7.1f}% | "
                      f"{r.max_drawdown_pct:>6.2f}% | {r.avg_profit_bps:>6.1f} | "
                      f"{r.avg_loss_bps:>7.2f} | {r.cost_per_trade_bps:>5.2f}")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY — Oracle PF Comparison (Val / Test)")
    print(f"{'='*80}")

    print(f"\n  {'Asset':>20s} | {'H1 Val':>8s} | {'H1 Test':>8s} | {'H3 Val':>8s} | "
          f"{'H3 Test':>8s} | {'H6 Val':>8s} | {'H6 Test':>8s} | {'Cost(bps)':>9s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")

    # BTC reference row
    print(f"  {'BTC (Binance)':>20s} | {BTC_REFERENCE['oracle_pf_5min_h1_val']:>8.3f} | "
          f"{BTC_REFERENCE['oracle_pf_5min_h1_test']:>8.3f} | {'—':>8s} | {'—':>8s} | "
          f"{'—':>8s} | {'—':>8s} | {BTC_REFERENCE['taker_cost_binance_bps']:>9.2f}")
    print(f"  {'BTC (Hyperliquid)':>20s} | {'1.368':>8s} | {'1.306':>8s} | {'—':>8s} | {'—':>8s} | "
          f"{'—':>8s} | {'—':>8s} | {BTC_REFERENCE['taker_cost_hyperliquid_bps']:>9.2f}")

    # CME results
    for asset_key in args.assets:
        info = ASSETS[asset_key]
        asset_results = [r for r in all_results if r.asset == info['name']]

        vals = {}
        for r in asset_results:
            key = f"H{r.horizon}_{r.split}"
            vals[key] = r.pf

        one_way = compute_taker_cost_bps(
            pd.read_parquet(DATA_DIR / info['file'])['close'].mean(), info
        )

        h1v = vals.get('H1_val', 0)
        h1t = vals.get('H1_test', 0)
        h3v = vals.get('H3_val', 0)
        h3t = vals.get('H3_test', 0)
        h6v = vals.get('H6_val', 0)
        h6t = vals.get('H6_test', 0)

        print(f"  {info['name']:>20s} | {h1v:>8.3f} | {h1t:>8.3f} | {h3v:>8.3f} | "
              f"{h3t:>8.3f} | {h6v:>8.3f} | {h6t:>8.3f} | {one_way:>9.2f}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    for asset_key in args.assets:
        info = ASSETS[asset_key]
        asset_results = [r for r in all_results if r.asset == info['name']]
        val_h1 = [r for r in asset_results if r.horizon == 1 and r.split == 'val']
        test_h1 = [r for r in asset_results if r.horizon == 1 and r.split == 'test']

        if val_h1 and test_h1:
            avg_pf = (val_h1[0].pf + test_h1[0].pf) / 2
            btc_avg = (BTC_REFERENCE['oracle_pf_5min_h1_val'] +
                       BTC_REFERENCE['oracle_pf_5min_h1_test']) / 2

            delta = avg_pf - btc_avg
            if avg_pf > 1.2:
                verdict = "STRONG PASS"
            elif avg_pf > 1.1:
                verdict = "PASS"
            elif avg_pf > 1.0:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

            print(f"  {info['name']:>20s}: H1 avg PF = {avg_pf:.3f} "
                  f"({'+'if delta>0 else ''}{delta:.3f} vs BTC) → {verdict}")


if __name__ == "__main__":
    main()
