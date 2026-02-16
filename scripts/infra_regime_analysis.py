"""
Infrastructure Diagnostic 1: Train/Test Distribution Shift Analysis
====================================================================
Compares market regimes across train (Jan-Oct), val (Nov), test (Dec) 2025.
Checks for structural breaks that would cause both PPO and BDQ to fail on test.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

DATA_PATH = "data/processed/btc_2025_full_year.parquet"

PERIODS = {
    "TRAIN (Jan-Oct)": ("2025-01-01", "2025-10-31 23:59:59"),
    "VAL   (Nov)":     ("2025-11-01", "2025-11-30 23:59:59"),
    "TEST  (Dec)":     ("2025-12-01", "2025-12-31 23:59:59"),
}


def hr(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def hurst_exponent(ts, max_lag=200):
    """Rescaled range (R/S) Hurst exponent estimate."""
    ts = np.asarray(ts, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    if len(ts) < max_lag * 2:
        return float('nan')
    lags = range(2, max_lag)
    tau = []
    rs = []
    for lag in lags:
        chunks = len(ts) // lag
        if chunks < 2:
            break
        rs_vals = []
        for i in range(chunks):
            chunk = ts[i * lag:(i + 1) * lag]
            mean_adj = chunk - chunk.mean()
            cum_dev = np.cumsum(mean_adj)
            r = cum_dev.max() - cum_dev.min()
            s = chunk.std(ddof=1)
            if s > 1e-12:
                rs_vals.append(r / s)
        if rs_vals:
            tau.append(lag)
            rs.append(np.mean(rs_vals))
    if len(tau) < 5:
        return float('nan')
    log_tau = np.log(tau)
    log_rs = np.log(rs)
    slope, _, _, _, _ = sp_stats.linregress(log_tau, log_rs)
    return slope


def analyze_period(name, df_period):
    """Compute regime statistics for one period."""
    p = df_period['mid_price'].values
    n = len(p)
    if n < 100:
        print(f"  {name}: INSUFFICIENT DATA ({n} rows)")
        return None

    ret_pct = (p[-1] - p[0]) / p[0] * 100
    rng_pct = (p.max() - p.min()) / p[0] * 100

    # Trend R²
    x = np.arange(n)
    sl, ic = np.polyfit(x, p, 1)
    pred = sl * x + ic
    ss_r = np.sum((p - pred) ** 2)
    ss_t = np.sum((p - p.mean()) ** 2)
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0

    # Returns
    rets = np.diff(p) / p[:-1]
    rets = rets[np.isfinite(rets)]
    vol_ann = np.std(rets) * np.sqrt(525_600) * 100  # annualized vol from minute returns

    # Autocorrelation
    acf = {}
    for lag in [1, 5, 10, 60]:
        if len(rets) > lag + 10:
            acf[lag] = np.corrcoef(rets[lag:], rets[:-lag])[0, 1]
        else:
            acf[lag] = float('nan')

    # Hurst exponent
    h = hurst_exponent(rets)

    # Hourly reversal rate
    ts = pd.to_datetime(df_period['timestamp'])
    hourly = df_period.set_index(ts)['mid_price'].resample('1h').last().dropna()
    hr_rets = hourly.pct_change().dropna().values
    if len(hr_rets) > 1:
        reversals = int((np.sign(hr_rets[:-1]) != np.sign(hr_rets[1:])).sum())
        total_hr = len(hr_rets) - 1
    else:
        reversals, total_hr = 0, 1

    # Regime classification
    if abs(ret_pct) < 3 and r2 < 0.3:
        regime = "RANGING / CHOPPY"
    elif ret_pct > 5 and r2 > 0.5:
        regime = "UPTRENDING"
    elif ret_pct < -5 and r2 > 0.5:
        regime = "DOWNTRENDING"
    elif abs(ret_pct) > 10 and r2 > 0.6:
        regime = "STRONG TREND"
    else:
        regime = "MIXED / TRANSITIONAL"

    result = {
        'name': name, 'n_rows': n,
        'start_price': p[0], 'end_price': p[-1],
        'high': p.max(), 'low': p.min(),
        'return_pct': ret_pct, 'range_pct': rng_pct,
        'trend_r2': r2, 'slope_per_hr': sl * 60,
        'vol_ann': vol_ann,
        'acf': acf, 'hurst': h,
        'reversal_rate': reversals / total_hr if total_hr > 0 else 0,
        'regime': regime,
        'returns': rets,
    }

    print(f"\n  === {name} ({n:,} rows) ===")
    print(f"    Price: ${p[0]:,.0f} → ${p[-1]:,.0f}  (high=${p.max():,.0f}  low=${p.min():,.0f})")
    print(f"    Return: {ret_pct:+.1f}%  |  Range: {rng_pct:.1f}%")
    print(f"    Trend R²: {r2:.3f}  |  Slope: ${sl * 60:.1f}/hr")
    print(f"    Volatility (ann): {vol_ann:.0f}%")
    print(f"    Hurst: {h:.3f}  ({'mean-reverting' if h < 0.45 else 'random walk' if h < 0.55 else 'trending'})")
    print(f"    ACF: lag1={acf[1]:.4f}  lag5={acf[5]:.4f}  lag10={acf[10]:.4f}  lag60={acf[60]:.4f}")
    print(f"    Hourly reversals: {reversals}/{total_hr} ({reversals/total_hr*100:.0f}%)")
    print(f"    REGIME: {regime}")

    return result


def run_ks_tests(results):
    """KS test comparing return distributions between periods."""
    hr("DISTRIBUTION COMPARISON (KS Tests)")
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            r1 = results[n1]['returns']
            r2 = results[n2]['returns']
            ks_stat, p_val = sp_stats.ks_2samp(r1, r2)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {n1} vs {n2}:")
            print(f"    KS stat={ks_stat:.4f}  p={p_val:.2e}  {sig}")
            # Also compare means and variances
            t_stat, t_p = sp_stats.ttest_ind(r1, r2, equal_var=False)
            lev_stat, lev_p = sp_stats.levene(r1, r2)
            print(f"    Mean test: t={t_stat:.3f}  p={t_p:.2e}")
            print(f"    Variance test (Levene): F={lev_stat:.3f}  p={lev_p:.2e}")


def summarize(results):
    """Print summary comparison table."""
    hr("SUMMARY TABLE")
    header = f"  {'Period':<20} {'Return':>8} {'Vol(ann)':>10} {'R²':>6} {'Hurst':>7} {'RevRate':>8} {'Regime':<20}"
    print(header)
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*6} {'-'*7} {'-'*8} {'-'*20}")
    for name, r in results.items():
        print(f"  {name:<20} {r['return_pct']:>+7.1f}% {r['vol_ann']:>9.0f}% {r['trend_r2']:>6.3f} {r['hurst']:>7.3f} {r['reversal_rate']:>7.0%} {r['regime']:<20}")


if __name__ == "__main__":
    hr("INFRASTRUCTURE DIAGNOSTIC 1: REGIME ANALYSIS")
    print(f"Data: {DATA_PATH}")

    try:
        df = pd.read_parquet(DATA_PATH, engine='fastparquet')
    except Exception:
        df = pd.read_parquet(DATA_PATH, engine='pyarrow')

    # Ensure mid_price exists
    if 'mid_price' not in df.columns:
        if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2.0
        else:
            print("ERROR: Cannot compute mid_price. Missing bid_price_1/ask_price_1.")
            sys.exit(1)

    if 'timestamp' not in df.columns:
        print("ERROR: No timestamp column.")
        sys.exit(1)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Total rows: {len(df):,}  |  {df['timestamp'].min()} → {df['timestamp'].max()}")

    results = {}
    for name, (start, end) in PERIODS.items():
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        subset = df[mask].copy()
        r = analyze_period(name, subset)
        if r:
            results[name] = r

    if len(results) >= 2:
        run_ks_tests(results)
        summarize(results)

    hr("VERDICT")
    if results:
        train_r = list(results.values())[0]
        test_r = list(results.values())[-1]
        regime_match = train_r['regime'] == test_r['regime']
        vol_ratio = test_r['vol_ann'] / train_r['vol_ann'] if train_r['vol_ann'] > 0 else float('inf')
        hurst_diff = abs(test_r['hurst'] - train_r['hurst'])

        print(f"  Regime match: {'YES ✓' if regime_match else 'NO ✗ — DISTRIBUTION SHIFT LIKELY'}")
        print(f"  Volatility ratio (test/train): {vol_ratio:.2f}x")
        print(f"  Hurst difference: {hurst_diff:.3f}  ({'acceptable' if hurst_diff < 0.1 else 'SIGNIFICANT'})")
        if not regime_match or vol_ratio > 1.5 or vol_ratio < 0.67 or hurst_diff > 0.1:
            print("\n  ⚠️  DISTRIBUTION SHIFT DETECTED — agents trained on one regime may fail on another.")
        else:
            print("\n  ✓  No major distribution shift detected.")

    hr("DIAGNOSTIC 1 COMPLETE")
