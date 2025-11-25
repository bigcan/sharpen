import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load Results
df = pd.read_csv("results/phase8_ensemble_test.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Calculate Daily Returns
df['returns'] = df['account_value'].pct_change().fillna(0)
df['cumulative_return'] = (1 + df['returns']).cumprod()

# Plot Equity Curve with Regimes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# 1. Equity Curve
ax1.plot(df.index, df['cumulative_return'], label='Ensemble Strategy', color='blue', linewidth=2)
ax1.set_title('Phase 8: Regime-Aware Ensemble Performance (2024 Out-of-Sample)', fontsize=14)
ax1.set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Regime Strip
# Map regimes to colors
regime_colors = {'Bull': 'green', 'Bear': 'red', 'Sideways': 'gray'}
# Create a numeric representation for plotting
# We plot colored bars
for date, row in df.iterrows():
    color = regime_colors.get(row['regime'], 'gray')
    ax2.axvspan(date, date + pd.Timedelta(days=1), color=color, alpha=0.6)

ax2.set_title('Active Market Regime (HMM Detected)', fontsize=12)
ax2.set_yticks([])
ax2.set_xlabel('Date', fontsize=12)

# Formatting
plt.tight_layout()
plt.savefig('figs/phase8_performance.png')
print("Performance plot saved to figs/phase8_performance.png")
