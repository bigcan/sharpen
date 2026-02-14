"""Analyze BTC market regime during training, validation, and test periods."""
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/btc_2025_jan_jun.parquet', columns=['timestamp', 'mid_price'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

train = df[(df['timestamp'] >= '2025-01-01') & (df['timestamp'] < '2025-06-01')]
val = df[(df['timestamp'] >= '2025-06-01') & (df['timestamp'] < '2025-06-15')]
test = df[(df['timestamp'] >= '2025-06-15') & (df['timestamp'] <= '2025-06-30')]

for name, s in [('TRAIN Jan-May', train), ('VAL Jun 1-14', val), ('TEST Jun 15-30', test)]:
    p = s['mid_price'].values
    n = len(p)
    ret = (p[-1] - p[0]) / p[0] * 100
    rng = (p.max() - p.min()) / p[0] * 100

    # Trend strength
    x = np.arange(n)
    sl, ic = np.polyfit(x, p, 1)
    pred = sl * x + ic
    ss_r = np.sum((p - pred) ** 2)
    ss_t = np.sum((p - p.mean()) ** 2)
    r2 = 1 - ss_r / ss_t if ss_t else 0

    # Volatility
    rets = np.diff(p) / p[:-1]
    vol = np.std(rets) * np.sqrt(525600) * 100

    # Hourly reversals
    h = s.set_index('timestamp')['mid_price'].resample('1h').last().dropna()
    hr = h.pct_change().dropna().values
    if len(hr) > 1:
        sc = int((np.sign(hr[:-1]) != np.sign(hr[1:])).sum())
        tot = len(hr) - 1
    else:
        sc, tot = 0, 1

    print(f"=== {name} ({n:,} rows) ===")
    print(f"  Start: ${p[0]:,.0f}  End: ${p[-1]:,.0f}")
    print(f"  High:  ${p.max():,.0f}  Low: ${p.min():,.0f}")
    print(f"  Return: {ret:+.1f}%  |  Range: {rng:.1f}%")
    print(f"  Trend R2: {r2:.3f}  |  Slope: ${sl * 60:.1f}/hr")
    print(f"  Volatility (ann): {vol:.0f}%")
    print(f"  Hourly reversals: {sc}/{tot} ({sc/tot*100:.0f}%)")

    # Regime classification
    if abs(ret) < 3 and r2 < 0.3:
        regime = "RANGING / CHOPPY"
    elif ret > 5 and r2 > 0.5:
        regime = "UPTRENDING"
    elif ret < -5 and r2 > 0.5:
        regime = "DOWNTRENDING"
    elif abs(ret) > 10 and r2 > 0.6:
        regime = "STRONG TREND"
    else:
        regime = "MIXED / TRANSITIONAL"
    print(f"  REGIME: {regime}")

    # Daily breakdown for val/test
    if name != 'TRAIN Jan-May':
        daily = s.set_index('timestamp')['mid_price'].resample('1D').agg(['first', 'last', 'max', 'min']).dropna()
        print(f"  --- Daily Breakdown ---")
        for dt, row in daily.iterrows():
            d_ret = (row['last'] - row['first']) / row['first'] * 100
            d_rng = (row['max'] - row['min']) / row['first'] * 100
            arrow = "^" if d_ret > 0.5 else ("v" if d_ret < -0.5 else "-")
            print(f"  {dt.strftime('%b %d')} {arrow} ${row['first']:,.0f} -> ${row['last']:,.0f} ({d_ret:+.1f}%) rng={d_rng:.1f}%")
    print()
