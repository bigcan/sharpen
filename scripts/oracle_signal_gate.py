"""
Signal-Gated Oracle: Does filtering bars by signal strength preserve oracle PF?
================================================================================

GO/NO-GO validation for the Signal-Gated RL hypothesis.

Computes gate signals on each bar (causally), ranks bars by signal strength,
then runs a constrained DP oracle that can only change position on "gated" bars.
Sweeps gate pass rates to find the optimal filtering level.

If the constrained oracle PF stays close to (or exceeds) the unconstrained oracle,
the hypothesis is confirmed: filtering noise bars preserves alpha while reducing
action frequency.

Usage:
    python scripts/oracle_signal_gate.py                                  # Gold 3m default
    python scripts/oracle_signal_gate.py --resolution 5min                # Gold 5m
    python scripts/oracle_signal_gate.py --data data/bitfinex/btc_usdt_perp_2025_1min.parquet --fee 5.0  # BTC
    python scripts/oracle_signal_gate.py --gate_mode atr                  # ATR-only gate
"""

import argparse
import time
import numpy as np
import pandas as pd


def load_and_resample(path: str, resolution: str = "3min",
                      start: str = None, end: str = None) -> pd.DataFrame:
    """Load 1-min parquet, resample to target resolution, return full OHLCV DataFrame."""
    df = pd.read_parquet(path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if resolution != "1min":
        agg_dict = {
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }
        df = df.resample(resolution).agg(agg_dict).dropna()

    # Drop zero-volume bars (non-trading periods)
    if 'volume' in df.columns:
        df = df[df['volume'] > 0]

    df = df.reset_index()
    print(f"Loaded {len(df):,} bars ({resolution}), "
          f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    return df


def compute_gate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal gate signals on each bar. All signals use only past/current data.

    Returns DataFrame with added signal columns.
    """
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    n = len(df)

    # 1. Parkinson volatility: sqrt(log(H/L)^2 / (4*ln2))
    hl_ratio = np.where(low > 0, high / low, 1.0)
    parkinson = np.sqrt(np.log(hl_ratio) ** 2 / (4 * np.log(2)))

    # 2. ATR ratio: ATR(14) / EMA(ATR, 50)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_ema_50 = pd.Series(atr_14).ewm(span=50, adjust=False).mean().values
    atr_ratio = np.where(atr_ema_50 > 1e-10, atr_14 / atr_ema_50, 1.0)

    # 3. Volume ratio: volume / EMA(volume, 50)
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False).mean().values
    volume_ratio = np.where(vol_ema_50 > 1e-10, volume / vol_ema_50, 1.0)

    # 4. Absolute log return
    log_ret = np.zeros(n)
    log_ret[1:] = np.abs(np.log(np.where(close[1:] > 0, close[1:], 1e-9) /
                                 np.where(close[:-1] > 0, close[:-1], 1e-9)))

    df = df.copy()
    df['parkinson_vol'] = parkinson
    df['atr_ratio'] = atr_ratio
    df['volume_ratio'] = volume_ratio
    df['abs_log_return'] = log_ret

    # Composite signal (simple sum of z-scored components)
    # Z-score each signal using expanding stats (causal)
    signals = np.column_stack([parkinson, atr_ratio, volume_ratio, log_ret])
    composite = np.zeros(n)
    for i in range(4):
        s = pd.Series(signals[:, i])
        expanding_mean = s.expanding(min_periods=2).mean().values
        expanding_std = s.expanding(min_periods=2).std().values
        # Fill NaN from warmup with 0 (neutral z-score)
        expanding_mean = np.nan_to_num(expanding_mean, nan=0.0)
        expanding_std = np.where(np.isnan(expanding_std) | (expanding_std < 1e-10), 1.0, expanding_std)
        z = (signals[:, i] - expanding_mean) / expanding_std
        z = np.nan_to_num(z, nan=0.0)
        composite += z

    df['composite_signal'] = composite

    return df


def dp_oracle_constrained(prices: np.ndarray, fee_bps: float = 5.0,
                           gate_mask: np.ndarray = None) -> dict:
    """DP backward induction with optional gate constraint.

    When gate_mask[t] = False, the oracle cannot change position at time t.
    This simulates a signal-gated agent that can only act on high-signal bars.

    Args:
        prices: Price array
        fee_bps: One-way taker fee in bps
        gate_mask: Boolean array, True = can trade, False = must hold.
                   If None, no constraint (equivalent to unconstrained oracle).
    """
    T = len(prices)
    fee_frac = fee_bps / 10000.0

    n_states = 3
    _, POS_FLAT, _ = 0, 1, 2
    pos_values = np.array([-1.0, 0.0, 1.0])

    V = np.zeros((T, n_states), dtype=np.float64)
    policy = np.ones((T, n_states), dtype=np.int32)

    # Terminal: force flat
    for p in range(n_states):
        if p != POS_FLAT:
            V[T - 1, p] = -abs(pos_values[p]) * prices[T - 1] * fee_frac
        policy[T - 1, p] = POS_FLAT

    # Backward induction
    for t in range(T - 2, -1, -1):
        dp = prices[t + 1] - prices[t]
        can_trade = gate_mask[t] if gate_mask is not None else True

        for cur_pos in range(n_states):
            if can_trade:
                # Normal: consider all transitions
                best_val = -np.inf
                best_next = cur_pos

                for next_pos in range(n_states):
                    reward = pos_values[cur_pos] * dp
                    if next_pos != cur_pos:
                        delta = abs(pos_values[next_pos] - pos_values[cur_pos])
                        cost = delta * prices[t] * fee_frac
                        reward -= cost
                    val = reward + V[t + 1, next_pos]
                    if val > best_val:
                        best_val = val
                        best_next = next_pos

                V[t, cur_pos] = best_val
                policy[t, cur_pos] = best_next
            else:
                # Gate closed: forced hold
                reward = pos_values[cur_pos] * dp
                V[t, cur_pos] = reward + V[t + 1, cur_pos]
                policy[t, cur_pos] = cur_pos

    # Forward pass
    positions = np.zeros(T, dtype=np.int32)
    positions[0] = POS_FLAT

    for t in range(T - 1):
        positions[t + 1] = policy[t, positions[t]]

    # Compute PnL
    pnl_per_step = np.zeros(T)
    total_fees = 0.0
    trade_count = 0

    for t in range(T - 1):
        dp = prices[t + 1] - prices[t]
        pnl_per_step[t] = pos_values[positions[t]] * dp

        if positions[t + 1] != positions[t]:
            delta = abs(pos_values[positions[t + 1]] - pos_values[positions[t]])
            cost = delta * prices[t] * fee_frac
            pnl_per_step[t] -= cost
            total_fees += cost
            trade_count += 1

    # Stats
    cumulative_pnl = np.cumsum(pnl_per_step)
    initial_capital = 100000.0
    portfolio_values = initial_capital + cumulative_pnl

    pos_pnl = pnl_per_step[pnl_per_step > 0].sum()
    neg_pnl = abs(pnl_per_step[pnl_per_step < 0].sum())
    pf = pos_pnl / neg_pnl if neg_pnl > 1e-9 else float('inf')

    returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-9)
    pos_ret = returns[returns > 0].sum()
    neg_ret = abs(returns[returns < 0].sum())
    pf_returns = pos_ret / neg_ret if neg_ret > 1e-9 else float('inf')

    total_return_pct = (portfolio_values[-1] / initial_capital - 1) * 100

    pos_series = pos_values[positions]
    in_market = np.sum(pos_series != 0) / T * 100

    # Hold durations
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

    # Gate utilization
    gated_bars = int(np.sum(gate_mask)) if gate_mask is not None else T
    gate_pct = gated_bars / T * 100

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
        'n_swings': len(hold_durations),
        'gate_open_bars': gated_bars,
        'gate_open_pct': gate_pct,
    }


def build_gate_mask(df: pd.DataFrame, pass_rate: float,
                    gate_mode: str = "composite") -> np.ndarray:
    """Build boolean gate mask based on signal percentile threshold.

    Args:
        df: DataFrame with gate signal columns (from compute_gate_signals)
        pass_rate: Fraction of bars to let through (0.0 to 1.0)
        gate_mode: Which signal to use for ranking
            - "composite": sum of z-scored signals
            - "atr": ATR ratio only
            - "parkinson": Parkinson volatility only
            - "volume": Volume ratio only
            - "return": Absolute log return only

    Returns:
        Boolean array, True = gate open (can trade)
    """
    n = len(df)

    if pass_rate >= 1.0:
        return np.ones(n, dtype=bool)
    if pass_rate <= 0.0:
        return np.zeros(n, dtype=bool)

    signal_map = {
        'composite': 'composite_signal',
        'atr': 'atr_ratio',
        'parkinson': 'parkinson_vol',
        'volume': 'volume_ratio',
        'return': 'abs_log_return',
    }

    col = signal_map.get(gate_mode, 'composite_signal')
    signal = df[col].values

    # Use expanding percentile (causal) to set threshold
    # At each bar, threshold = (1 - pass_rate) percentile of all signals seen so far
    mask = np.zeros(n, dtype=bool)
    min_warmup = 50  # Don't filter during warmup

    for t in range(n):
        if t < min_warmup:
            mask[t] = True  # Always open during warmup
        else:
            # Expanding threshold: percentile of signal[0:t+1]
            threshold = np.percentile(signal[:t + 1], (1 - pass_rate) * 100)
            mask[t] = signal[t] >= threshold

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Signal-Gated Oracle: GO/NO-GO validation for signal filtering")
    parser.add_argument("--data", default="data/cme/gc_2025_lob1_1min_stitched.parquet",
                        help="Path to 1-min parquet")
    parser.add_argument("--resolution", default="3min",
                        help="Target bar resolution (3min, 5min, 15min, etc.)")
    parser.add_argument("--fee", type=float, default=2.0,
                        help="One-way taker fee in bps (default: 2.0 for Gold)")
    parser.add_argument("--start", default=None, help="Start date filter")
    parser.add_argument("--end", default=None, help="End date filter")
    parser.add_argument("--gate_mode", default="composite",
                        choices=["composite", "atr", "parkinson", "volume", "return"],
                        help="Gate signal mode")
    parser.add_argument("--max_bars", type=int, default=0,
                        help="Limit bars for testing (0=all)")
    args = parser.parse_args()

    # Load and resample
    df = load_and_resample(args.data, args.resolution, args.start, args.end)

    if args.max_bars > 0:
        df = df.iloc[:args.max_bars].reset_index(drop=True)
        print(f"Truncated to {len(df):,} bars")

    prices = df['close'].values.astype(np.float64)
    print(f"\nAsset: {args.data}")
    print(f"Resolution: {args.resolution}")
    print(f"Fee: {args.fee} bps one-way ({args.fee * 2} bps round-trip)")
    print(f"Gate mode: {args.gate_mode}")
    print(f"Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")
    print(f"Mean |1-bar return|: {np.mean(np.abs(np.diff(prices)/prices[:-1]))*10000:.2f} bps")
    print()

    # Compute gate signals
    print("Computing gate signals...")
    df = compute_gate_signals(df)

    # Signal statistics
    print("\nGate Signal Statistics:")
    for col in ['parkinson_vol', 'atr_ratio', 'volume_ratio', 'abs_log_return', 'composite_signal']:
        vals = df[col].values
        print(f"  {col:20s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"p25={np.percentile(vals, 25):.4f}  p50={np.percentile(vals, 50):.4f}  "
              f"p75={np.percentile(vals, 75):.4f}  p95={np.percentile(vals, 95):.4f}")
    print()

    # === Unconstrained Oracle (baseline) ===
    print("=" * 80)
    print("UNCONSTRAINED DP ORACLE (baseline)")
    print("=" * 80)
    t0 = time.time()
    baseline = dp_oracle_constrained(prices, fee_bps=args.fee, gate_mask=None)
    baseline_time = time.time() - t0

    print(f"  PF (PnL):     {baseline['profit_factor_pnl']:.3f}")
    print(f"  PF (Returns): {baseline['profit_factor_returns']:.3f}")
    print(f"  Return:       {baseline['total_return_pct']:+.2f}%")
    print(f"  Trades:       {baseline['trade_count']:,}")
    print(f"  In-Market:    {baseline['in_market_pct']:.1f}%")
    print(f"  MaxDD:        {baseline['max_drawdown_pct']:.2f}%")
    print(f"  Mean Hold:    {baseline['mean_hold_bars']:.1f} bars")
    print(f"  Time:         {baseline_time:.1f}s")
    print()

    # === Sweep gate pass rates ===
    pass_rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    print("=" * 80)
    print(f"SIGNAL-GATED ORACLE SWEEP (gate_mode={args.gate_mode})")
    print("=" * 80)
    print(f"{'Pass%':>6s}  {'OpenBars':>8s}  {'PF_PnL':>7s}  {'PF_Ret':>7s}  "
          f"{'Return%':>8s}  {'Trades':>7s}  {'InMkt%':>7s}  {'MaxDD%':>7s}  "
          f"{'MnHold':>7s}  {'PF_Ratio':>8s}")
    print("-" * 95)

    results = []
    for pr in pass_rates:
        gate_mask = build_gate_mask(df, pr, gate_mode=args.gate_mode)
        r = dp_oracle_constrained(prices, fee_bps=args.fee, gate_mask=gate_mask)

        pf_ratio = r['profit_factor_pnl'] / baseline['profit_factor_pnl'] if baseline['profit_factor_pnl'] > 0 else 0
        r['pass_rate'] = pr
        r['pf_ratio_vs_baseline'] = pf_ratio
        results.append(r)

        print(f"{pr*100:5.0f}%  {r['gate_open_bars']:>8,}  {r['profit_factor_pnl']:7.3f}  "
              f"{r['profit_factor_returns']:7.3f}  {r['total_return_pct']:+7.2f}%  "
              f"{r['trade_count']:>7,}  {r['in_market_pct']:6.1f}%  "
              f"{r['max_drawdown_pct']:6.2f}%  {r['mean_hold_bars']:6.1f}  "
              f"{pf_ratio:7.2f}x")

    print()

    # === Analysis ===
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find best PF across pass rates
    best = max(results, key=lambda x: x['profit_factor_pnl'])
    best_pr = best['pass_rate']

    print(f"\n  Baseline (unconstrained) PF: {baseline['profit_factor_pnl']:.3f}")
    print(f"  Best constrained PF:         {best['profit_factor_pnl']:.3f} "
          f"at {best_pr*100:.0f}% pass rate")
    print(f"  PF ratio (best/baseline):    {best['pf_ratio_vs_baseline']:.3f}x")

    # Check if any pass rate EXCEEDS unconstrained
    exceeded = [r for r in results if r['profit_factor_pnl'] > baseline['profit_factor_pnl'] * 1.01]
    if exceeded:
        print(f"\n  ** STRONG SIGNAL: {len(exceeded)} pass rate(s) EXCEED unconstrained oracle **")
        print("  ** Filtering noise bars IMPROVES oracle PF **")
        for r in exceeded:
            print(f"     {r['pass_rate']*100:.0f}%: PF={r['profit_factor_pnl']:.3f} "
                  f"({r['pf_ratio_vs_baseline']:.3f}x baseline)")

    # Check 30% criterion
    r_30 = next((r for r in results if r['pass_rate'] == 0.30), None)
    if r_30:
        ratio_30 = r_30['pf_ratio_vs_baseline']
        print("\n  30% pass rate check:")
        print(f"    PF = {r_30['profit_factor_pnl']:.3f} ({ratio_30:.2f}x baseline)")
        print(f"    Trades = {r_30['trade_count']:,} (vs {baseline['trade_count']:,} baseline)")
        trade_reduction = 1 - r_30['trade_count'] / max(baseline['trade_count'], 1)
        print(f"    Trade reduction: {trade_reduction*100:.1f}%")

    # === VERDICT ===
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Find the most capital-efficient point: highest PF with meaningful trade count
    efficient = [r for r in results if r['trade_count'] >= 10 and r['pass_rate'] < 1.0]
    if efficient:
        best_eff = max(efficient, key=lambda x: x['profit_factor_pnl'])
        eff_ratio = best_eff['pf_ratio_vs_baseline']

        if eff_ratio >= 1.0:
            print("\n  >>> GO: Signal gating PRESERVES or IMPROVES oracle PF <<<")
            print(f"  >>> Best: {best_eff['pass_rate']*100:.0f}% pass, "
                  f"PF={best_eff['profit_factor_pnl']:.3f}, "
                  f"{best_eff['trade_count']} trades <<<")
            print("  >>> Proceed to Phase 1: Build SignalGatedWrapper <<<")
        elif eff_ratio >= 0.70:
            print(f"\n  >>> MARGINAL GO: PF retention {eff_ratio:.0%} (>= 70% threshold) <<<")
            print("  >>> Filtering costs some oracle PF but may help RL agent by reducing noise <<<")
            print("  >>> Proceed with caution to Phase 1 <<<")
        else:
            print(f"\n  >>> NO-GO: PF retention {eff_ratio:.0%} (< 70% threshold) <<<")
            print("  >>> Signal filtering destroys too much oracle alpha <<<")
            print("  >>> The alpha is distributed across ALL bars, not concentrated <<<")
            print("  >>> Hypothesis FALSIFIED. Do not proceed to Phase 1. <<<")
    else:
        print("\n  >>> INCONCLUSIVE: Too few trades at all pass rates <<<")

    # Per-signal mode comparison
    if args.gate_mode == "composite":
        print()
        print("=" * 80)
        print("PER-SIGNAL MODE COMPARISON (at 30% pass rate)")
        print("=" * 80)
        print(f"{'Mode':>12s}  {'PF_PnL':>7s}  {'PF_Ret':>7s}  {'Trades':>7s}  {'Ratio':>7s}")
        print("-" * 50)

        for mode in ["composite", "atr", "parkinson", "volume", "return"]:
            mask = build_gate_mask(df, 0.30, gate_mode=mode)
            r = dp_oracle_constrained(prices, fee_bps=args.fee, gate_mask=mask)
            ratio = r['profit_factor_pnl'] / baseline['profit_factor_pnl']
            marker = " <-- BEST" if ratio >= 1.0 else ""
            print(f"{mode:>12s}  {r['profit_factor_pnl']:7.3f}  "
                  f"{r['profit_factor_returns']:7.3f}  "
                  f"{r['trade_count']:>7,}  {ratio:6.2f}x{marker}")


if __name__ == "__main__":
    main()
