"""
DP Oracle: True Optimal Trading with Transaction Costs (1-Min BTC)
=================================================================

Backward-induction Dynamic Programming to find the MAXIMUM possible
profit factor with perfect foresight, accounting for transaction costs.

Unlike our H=1 bar-by-bar oracle (which re-evaluates every bar and churns),
this oracle finds the globally optimal sequence of Buy/Hold/Sell decisions
that maximizes total PnL after paying round-trip taker fees.

Key difference: this oracle naturally HOLDS through swings, paying the
10bps RT cost only once per swing instead of per bar.

State: (time t, position p ∈ {-1, 0, +1})
Action: {go_long, go_short, go_flat} → transition cost if position changes
Reward: position × (mid[t+1] - mid[t]) per step (mark-to-market)
Cost: taker_fee_bps × 2 per leg when position changes (entry + exit)

Usage:
    python scripts/dp_oracle_1min.py                          # Full year
    python scripts/dp_oracle_1min.py --start 2025-05-01 --end 2025-06-30  # May-Jun
    python scripts/dp_oracle_1min.py --fee 2.5                # Hyperliquid fees
    python scripts/dp_oracle_1min.py --resolution 5min        # 5-min bars
"""

import argparse
import time
import numpy as np
import pandas as pd


def load_data(path: str, start: str = None, end: str = None,
              resolution: str = "1min",
              price_col: str = "close") -> np.ndarray:
    """Load prices from parquet, optionally resample.

    Args:
        price_col: Which price column to use ('close' or 'mid_price').
                   'close' is the bar close — executable at bar boundaries.
                   'mid_price' = (high+low)/2 — incorporates intra-bar extremes.
    """
    df = pd.read_parquet(path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if resolution != "1min":
        # Support arbitrary resolutions: "2min", "3min", "5min", "10min", etc.
        agg_dict = {
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }
        if 'mid_price' in df.columns:
            agg_dict['mid_price'] = 'last'
        df = df.resample(resolution).agg(agg_dict).dropna()

    # Resolve price column
    if price_col == 'mid_price' and 'mid_price' not in df.columns:
        df['mid_price'] = (df['high'] + df['low']) / 2.0
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in data. "
                         f"Available: {list(df.columns)}")

    prices = df[price_col].values.astype(np.float64)
    print(f"Loaded {len(prices):,} bars ({resolution}), price_col='{price_col}', "
          f"{df.index[0]} → {df.index[-1]}")
    return prices, df


def dp_oracle(prices: np.ndarray, fee_bps: float = 5.0,
              max_pos: int = 1) -> dict:
    """
    DP backward induction for optimal trading with transaction costs.

    Returns the maximum achievable PnL path and trade statistics.

    States: position ∈ {-max_pos, ..., 0, ..., +max_pos}
    For simplicity and speed, we use {-1, 0, +1}.
    """
    T = len(prices)
    fee_frac = fee_bps / 10000.0  # Convert bps to fraction

    # Position states: 0=-1(short), 1=0(flat), 2=+1(long)
    n_states = 3
    POS_SHORT, POS_FLAT, POS_LONG = 0, 1, 2
    pos_values = np.array([-1.0, 0.0, 1.0])

    # V[t, pos] = max future PnL from time t onward, given position pos
    V = np.zeros((T, n_states), dtype=np.float64)
    # Policy[t, pos] = optimal next position
    policy = np.ones((T, n_states), dtype=np.int32)  # default: stay flat

    # Terminal condition: force flat at end (close all positions)
    for p in range(n_states):
        if p != POS_FLAT:
            # Cost to close position at terminal time
            V[T - 1, p] = -abs(pos_values[p]) * prices[T - 1] * fee_frac
        else:
            V[T - 1, p] = 0.0
        policy[T - 1, p] = POS_FLAT

    # Backward induction
    t0 = time.time()
    report_interval = T // 10

    for t in range(T - 2, -1, -1):
        if report_interval > 0 and t % report_interval == 0:
            elapsed = time.time() - t0
            pct = (T - 2 - t) / (T - 1) * 100
            print(f"  DP step {T-2-t:>8,}/{T-1:,} ({pct:5.1f}%) "
                  f"elapsed {elapsed:.1f}s", end='\r')

        dp = prices[t + 1] - prices[t]  # Price change this bar

        for cur_pos in range(n_states):
            best_val = -np.inf
            best_next = cur_pos

            for next_pos in range(n_states):
                # Holding reward: current position × price change
                reward = pos_values[cur_pos] * dp

                # Transaction cost if position changes
                if next_pos != cur_pos:
                    # Size of position change
                    delta = abs(pos_values[next_pos] - pos_values[cur_pos])
                    cost = delta * prices[t] * fee_frac
                    reward -= cost

                val = reward + V[t + 1, next_pos]
                if val > best_val:
                    best_val = val
                    best_next = next_pos

            V[t, cur_pos] = best_val
            policy[t, cur_pos] = best_next

    elapsed = time.time() - t0
    print(f"\n  DP complete in {elapsed:.1f}s")

    # Forward pass: extract optimal trajectory
    # positions[t] = the position held DURING bar t (arrival state)
    # policy[t, state] = the state to transition to at END of bar t
    positions = np.zeros(T, dtype=np.int32)
    positions[0] = POS_FLAT  # Start flat (no position at bar 0)

    for t in range(T - 1):
        positions[t + 1] = policy[t, positions[t]]

    # Compute PnL series
    pnl_per_step = np.zeros(T)
    total_fees = 0.0
    trade_count = 0

    for t in range(T - 1):
        dp = prices[t + 1] - prices[t]
        # Holding PnL: position held during bar t × price change
        pnl_per_step[t] = pos_values[positions[t]] * dp

        # Fee on position change (transition at end of bar t)
        if positions[t + 1] != positions[t]:
            delta = abs(pos_values[positions[t + 1]] - pos_values[positions[t]])
            cost = delta * prices[t] * fee_frac
            pnl_per_step[t] -= cost
            total_fees += cost
            trade_count += 1

    # Portfolio value series
    cumulative_pnl = np.cumsum(pnl_per_step)
    initial_capital = 100000.0
    portfolio_values = initial_capital + cumulative_pnl

    # Profit factor from PnL steps
    pos_pnl = pnl_per_step[pnl_per_step > 0].sum()
    neg_pnl = abs(pnl_per_step[pnl_per_step < 0].sum())
    pf = pos_pnl / neg_pnl if neg_pnl > 1e-9 else float('inf')

    # Return-based PF (for comparison with our standard metric)
    returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-9)
    pos_ret = returns[returns > 0].sum()
    neg_ret = abs(returns[returns < 0].sum())
    pf_returns = pos_ret / neg_ret if neg_ret > 1e-9 else float('inf')

    # Trade statistics
    pos_series = pos_values[positions]
    in_market = np.sum(pos_series != 0) / T * 100

    # Hold duration stats
    hold_durations = []
    current_start = None
    current_pos = POS_FLAT
    for t in range(T):
        if positions[t] != current_pos:
            if current_pos != POS_FLAT and current_start is not None:
                hold_durations.append(t - current_start)
            if positions[t] != POS_FLAT:
                current_start = t
            else:
                current_start = None
            current_pos = positions[t]
    if current_pos != POS_FLAT and current_start is not None:
        hold_durations.append(T - current_start)

    hold_durations = np.array(hold_durations) if hold_durations else np.array([0])

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + 1e-9)
    max_dd = drawdown.max() * 100

    total_return_pct = (portfolio_values[-1] / initial_capital - 1) * 100

    # Long vs short breakdown
    n_long = np.sum(positions == POS_LONG)
    n_short = np.sum(positions == POS_SHORT)
    n_flat = np.sum(positions == POS_FLAT)

    return {
        'profit_factor_pnl': pf,
        'profit_factor_returns': pf_returns,
        'total_return_pct': total_return_pct,
        'total_pnl': cumulative_pnl[-1],
        'total_fees': total_fees,
        'trade_count': trade_count,
        'in_market_pct': in_market,
        'max_drawdown_pct': max_dd,
        'mean_hold_bars': hold_durations.mean() if len(hold_durations) > 0 else 0,
        'median_hold_bars': np.median(hold_durations) if len(hold_durations) > 0 else 0,
        'max_hold_bars': hold_durations.max() if len(hold_durations) > 0 else 0,
        'min_hold_bars': hold_durations.min() if len(hold_durations) > 0 else 0,
        'n_swings': len(hold_durations),
        'n_long_bars': n_long,
        'n_short_bars': n_short,
        'n_flat_bars': n_flat,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'pnl_per_step': pnl_per_step,
        'optimal_value_flat': V[0, POS_FLAT],
    }


def bar_by_bar_oracle(prices: np.ndarray, fee_bps: float = 5.0,
                      horizon: int = 1) -> dict:
    """Our existing H=1 bar-by-bar oracle for direct comparison."""
    T = len(prices)
    fee_frac = fee_bps / 10000.0
    cost_bps = fee_bps * 2  # Round-trip

    position = 0.0
    balance = 100000.0
    entry_price = 0.0
    portfolio_values = [balance]
    trade_count = 0

    for t in range(T - horizon):
        fill_mid = prices[t]
        future_mid = prices[t + horizon]
        expected_return_bps = ((future_mid - fill_mid) / fill_mid) * 10000

        if expected_return_bps > cost_bps:
            target = 1.0
        elif expected_return_bps < -cost_bps:
            target = -1.0
        else:
            target = 0.0

        # Execute position change
        if target != position:
            if position != 0:
                pnl = position * (prices[t] - entry_price)
                fee = abs(position) * prices[t] * fee_frac
                balance += pnl - fee
                trade_count += 1
            if target != 0:
                entry_price = prices[t]
                fee = abs(target) * prices[t] * fee_frac
                balance -= fee
                trade_count += 1
            position = target

        # Mark to market
        if position != 0:
            mtm = balance + position * (prices[t] - entry_price)
        else:
            mtm = balance
        portfolio_values.append(mtm)

    # Close final position
    if position != 0:
        pnl = position * (prices[-1] - entry_price)
        fee = abs(position) * prices[-1] * fee_frac
        balance += pnl - fee
        trade_count += 1
        portfolio_values.append(balance)

    pv = np.array(portfolio_values)
    returns = np.diff(pv) / (pv[:-1] + 1e-9)
    pos_ret = returns[returns > 0].sum()
    neg_ret = abs(returns[returns < 0].sum())
    pf = pos_ret / neg_ret if neg_ret > 1e-9 else float('inf')
    total_return = (pv[-1] / 100000.0 - 1) * 100

    return {
        'profit_factor': pf,
        'total_return_pct': total_return,
        'trade_count': trade_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DP Oracle: True optimal trading ceiling with transaction costs")
    parser.add_argument("--data", default="data/bitfinex/btc_usdt_perp_2025_1min.parquet")
    parser.add_argument("--start", default=None, help="Start date filter")
    parser.add_argument("--end", default=None, help="End date filter")
    parser.add_argument("--fee", type=float, default=5.0, help="One-way taker fee in bps")
    parser.add_argument("--resolution", default="1min",
                        help="Bar resolution: 1min, 2min, 3min, 5min, 10min, 15min, 30min, etc.")
    parser.add_argument("--max-bars", type=int, default=0,
                        help="Limit bars for testing (0=all)")
    parser.add_argument("--price_col", default="close", choices=["close", "mid_price"],
                        help="Price column for oracle (default: close)")
    args = parser.parse_args()

    prices, df = load_data(args.data, args.start, args.end, args.resolution,
                           price_col=args.price_col)

    if args.max_bars > 0:
        prices = prices[:args.max_bars]
        print(f"Truncated to {len(prices):,} bars for testing")

    print(f"\nFee: {args.fee} bps one-way ({args.fee*2} bps round-trip)")
    print(f"Price range: ${prices.min():,.0f} – ${prices.max():,.0f}")
    print(f"Mean |1-bar return|: {np.mean(np.abs(np.diff(prices)/prices[:-1]))*10000:.2f} bps")
    print()

    # === Bar-by-bar oracle (our current method) ===
    print("=" * 70)
    print("BAR-BY-BAR ORACLE (H=1) — Our current method")
    print("=" * 70)
    bbb = bar_by_bar_oracle(prices, fee_bps=args.fee, horizon=1)
    print(f"  Profit Factor:  {bbb['profit_factor']:.3f}")
    print(f"  Total Return:   {bbb['total_return_pct']:+.2f}%")
    print(f"  Trade Count:    {bbb['trade_count']:,}")
    print()

    # === DP Oracle (true ceiling) ===
    print("=" * 70)
    print("DP ORACLE — True optimal with hold-through-swing")
    print("=" * 70)
    dp = dp_oracle(prices, fee_bps=args.fee)

    print(f"\n  Profit Factor (PnL-based):     {dp['profit_factor_pnl']:.3f}")
    print(f"  Profit Factor (Returns-based): {dp['profit_factor_returns']:.3f}")
    print(f"  Total Return:            {dp['total_return_pct']:+.2f}%")
    print(f"  Total PnL:               ${dp['total_pnl']:+,.2f}")
    print(f"  Total Fees Paid:         ${dp['total_fees']:,.2f}")
    print(f"  DP Optimal Value (flat): ${dp['optimal_value_flat']:,.2f}")
    print()
    print(f"  Trade Count:      {dp['trade_count']:,} "
          f"(vs {bbb['trade_count']:,} bar-by-bar)")
    print(f"  Swing Count:      {dp['n_swings']:,}")
    print(f"  In-Market:        {dp['in_market_pct']:.1f}%")
    print(f"  Max Drawdown:     {dp['max_drawdown_pct']:.2f}%")
    print()
    print(f"  Hold Duration (bars):")
    print(f"    Mean:    {dp['mean_hold_bars']:.1f}")
    print(f"    Median:  {dp['median_hold_bars']:.1f}")
    print(f"    Min:     {dp['min_hold_bars']}")
    print(f"    Max:     {dp['max_hold_bars']}")
    print()
    print(f"  Position Breakdown:")
    print(f"    Long:  {dp['n_long_bars']:,} bars ({dp['n_long_bars']/len(prices)*100:.1f}%)")
    print(f"    Short: {dp['n_short_bars']:,} bars ({dp['n_short_bars']/len(prices)*100:.1f}%)")
    print(f"    Flat:  {dp['n_flat_bars']:,} bars ({dp['n_flat_bars']/len(prices)*100:.1f}%)")

    # === Comparison ===
    print()
    print("=" * 70)
    print("COMPARISON — Does multi-bar holding change the ceiling?")
    print("=" * 70)
    ratio = dp['profit_factor_returns'] / bbb['profit_factor'] if bbb['profit_factor'] > 0 else float('inf')
    print(f"  Bar-by-bar PF: {bbb['profit_factor']:.3f}")
    print(f"  DP Oracle PF:  {dp['profit_factor_returns']:.3f}")
    print(f"  Ratio:         {ratio:.2f}x")
    print()

    if dp['profit_factor_returns'] >= 2.0:
        print("  >>> VERDICT: 1-MIN IS VIABLE <<<")
        print("  >>> DP Oracle PF >= 2.0 — massive headroom for sub-oracle agents")
        print("  >>> Our H=1 oracle was SEVERELY underestimating the ceiling")
        print("  >>> RESEARCH DIRECTION CHANGE WARRANTED")
    elif dp['profit_factor_returns'] >= 1.5:
        print("  >>> VERDICT: 1-MIN MARGINAL BUT POSSIBLE <<<")
        print("  >>> PF 1.5-2.0 — some headroom, but agent needs strong swing detection")
        print("  >>> Consider reopening 1-min as secondary research track")
    elif dp['profit_factor_returns'] >= 1.25:
        print("  >>> VERDICT: 1-MIN STILL DIFFICULT <<<")
        print("  >>> PF 1.25-1.5 — thin margin even for optimal trader")
        print("  >>> Stay on 5-min. Multi-bar helps but not enough.")
    else:
        print("  >>> VERDICT: 1-MIN CONFIRMED DEAD <<<")
        print("  >>> Even optimal hold-through-swing can't fix the fee moat")
        print("  >>> Stay on 5-min. No research direction change needed.")

    # === PF-XCHECK: cross-validate close vs mid_price ===
    print()
    print("=" * 70)
    print("PF-XCHECK — Cross-validation: close vs mid_price")
    print("=" * 70)

    # Build the *other* price series for comparison
    other_col = 'mid_price' if args.price_col == 'close' else 'close'
    try:
        _, df_xcheck = load_data(args.data, args.start, args.end, args.resolution,
                                 price_col=other_col)
        xcheck_prices = df_xcheck[other_col].values.astype(np.float64)
        # Limit to max 50K bars for speed
        xcheck_limit = min(50000, len(xcheck_prices))
        primary_limit = min(50000, len(prices))

        xcheck_result = dp_oracle(xcheck_prices[:xcheck_limit], fee_bps=args.fee)
        primary_result = dp_oracle(prices[:primary_limit], fee_bps=args.fee)

        pf_primary = primary_result['profit_factor_pnl']
        pf_other = xcheck_result['profit_factor_pnl']

        # Compute divergence
        avg_pf = (pf_primary + pf_other) / 2.0
        divergence_pct = abs(pf_primary - pf_other) / avg_pf * 100 if avg_pf > 1e-9 else 0

        print(f"\n  PF-XCHECK (PnL-based, first {primary_limit:,} bars):")
        print(f"    {args.price_col:10s} PF = {pf_primary:.3f}")
        print(f"    {other_col:10s} PF = {pf_other:.3f}")
        print(f"    Divergence = {divergence_pct:.1f}%")

        if divergence_pct > 30:
            print(f"\n  *** WARNING: PF divergence {divergence_pct:.1f}% > 30% ***")
            print("  *** Data may have corrupted high/low values inflating mid_price. ***")
            print("  *** Run scripts/clean_ohlcv.py before trusting results. ***")
        else:
            print("    OK — divergence within 30% tolerance")
    except Exception as e:
        print(f"\n  PF-XCHECK skipped: {e}")

    # === Fee sweep ===
    print()
    print("=" * 70)
    print("FEE SENSITIVITY — DP Oracle at different fee tiers")
    print("=" * 70)
    for fee in [0.0, 1.0, 2.5, 3.5, 5.0, 7.5, 10.0]:
        r = dp_oracle(prices[:min(50000, len(prices))], fee_bps=fee)
        print(f"  Fee={fee:5.1f}bps  PF={r['profit_factor_returns']:7.3f}  "
              f"Return={r['total_return_pct']:+8.2f}%  "
              f"Trades={r['trade_count']:>6,}  "
              f"MeanHold={r['mean_hold_bars']:5.1f} bars")


if __name__ == "__main__":
    main()
