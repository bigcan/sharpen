import os

import numpy as np
import pandas as pd


def oracle_ceiling_test(prices: np.ndarray, taker_fee_bps: float = 5.0, horizon: int = 1):
    T = len(prices)
    cost_bps = taker_fee_bps * 2  # Round-trip

    position = 0.0  # {-1, 0, +1}
    balance = 100000.0
    portfolio_values = [balance]
    trade_count = 0

    entry_price = 0.0

    for t in range(T - horizon):
        fill_price = prices[t]
        future_price = prices[t + horizon]

        expected_return_bps = ((future_price - fill_price) / fill_price) * 10000

        # Oracle decision
        if expected_return_bps > cost_bps:
            target_position = 1.0
        elif expected_return_bps < -cost_bps:
            target_position = -1.0
        else:
            target_position = 0.0

        # Execute if position changes
        if target_position != position:
            # Close old position
            if position != 0:
                pnl = position * (prices[t] - entry_price)
                fee = abs(position) * prices[t] * (taker_fee_bps / 10000)
                balance += pnl - fee
                trade_count += 1

            # Open new position
            if target_position != 0:
                entry_price = prices[t]
                fee = abs(target_position) * prices[t] * (taker_fee_bps / 10000)
                balance -= fee
                trade_count += 1

            position = target_position

        # Mark-to-market
        if position != 0:
            mtm = balance + position * (prices[t] - entry_price)
        else:
            mtm = balance
        portfolio_values.append(mtm)

    # Compute PF
    returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-9)
    pos_ret = returns[returns > 0].sum()
    neg_ret = abs(returns[returns < 0].sum())
    pf = pos_ret / neg_ret if neg_ret > 1e-9 else float('inf')

    return pf, trade_count, returns

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Oracle Ceiling Audit")
    parser.add_argument("--data", default="data/bitfinex/btc_usdt_perp_2025_1min.parquet",
                        help="Path to OHLCV parquet")
    parser.add_argument("--price_col", default="close", choices=["close", "mid_price"],
                        help="Price column for oracle (default: close)")
    args = parser.parse_args()

    file_path = args.data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_parquet(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Price column: {args.price_col}")
    print(df.columns)

    # Resolve price column
    if args.price_col == 'close' and 'close' in df.columns:
        prices = df['close'].values.astype(np.float64)
    elif args.price_col == 'mid_price':
        if 'mid_price' in df.columns:
            prices = df['mid_price'].values.astype(np.float64)
        elif 'high' in df.columns and 'low' in df.columns:
            prices = ((df['high'] + df['low']) / 2).values.astype(np.float64)
        elif 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            prices = ((df['bid_price_1'] + df['ask_price_1']) / 2).values.astype(np.float64)
        else:
            print("Cannot compute mid_price from available columns.")
            return
    else:
        print(f"Cannot find price column '{args.price_col}'.")
        return

    pf, count, rets = oracle_ceiling_test(prices, taker_fee_bps=5.0, horizon=1)
    print("=== FULL DATASET ===")
    print(f"Horizon=1, Taker=5bps -> PF = {pf:.4f}, Trades = {count}, Avg return count: {len(rets)}")

    # Mathematical validation check
    rets_1min = np.diff(prices) / prices[:-1] * 10000
    mean_abs_ret = np.mean(np.abs(rets_1min))
    median_abs_ret = np.median(np.abs(rets_1min))
    std_ret = np.std(rets_1min)

    print("\n=== MATHEMATICAL PROPERTIES (1-min returns) ===")
    print(f"Mean |return|: {mean_abs_ret:.4f} bps")
    print(f"Median |return|: {median_abs_ret:.4f} bps")
    print(f"Std return: {std_ret:.4f} bps")

    cond_exp = np.mean(np.abs(rets_1min)[np.abs(rets_1min) > 10.0])
    print(f"E[|r| | |r| > 10 bps]: {cond_exp:.4f} bps")

    pf_theoretical = cond_exp / 10.0
    print(f"Theoretical PF bound (no friction): {pf_theoretical:.4f}")

    # Zero fee
    pf_0, count_0, _ = oracle_ceiling_test(prices, taker_fee_bps=0.0, horizon=1)
    print(f"\nHorizon=1, Taker=0bps -> PF = {pf_0:.4f}, Trades = {count_0}")

    # Hyperliquid fee
    pf_hl, count_hl, _ = oracle_ceiling_test(prices, taker_fee_bps=2.5, horizon=1)
    print(f"Horizon=1, Taker=2.5bps -> PF = {pf_hl:.4f}, Trades = {count_hl}")

if __name__ == '__main__':
    main()
