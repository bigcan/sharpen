import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro.data.regimes import HMMRegimeDetector, MarketRegime

def label_regimes():
    ticker = 'AAPL'
    file_path = f"data/sp500_multi_2016_2025/{ticker}.parquet"
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_parquet(file_path)
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    
    # Returns
    returns = df['close'].pct_change().dropna()
    
    # HMM
    print("Fitting HMM (3 Components)...")
    detector = HMMRegimeDetector(n_components=3, random_state=42)
    regimes = detector.fit_predict(returns.values)
    
    # Create Regime Series
    regime_series = pd.Series(index=returns.index, data=regimes)
    
    # Map to labels
    # HMMRegimeDetector maps states to: 0=BEAR, 1=BULL, 2=CRISIS, 3=SIDEWAYS
    # Note: The mapping is dynamic inside HMMRegimeDetector but it returns the Enum value.
    # MarketRegime Enum: BEAR=0, BULL=1, CRISIS=2, SIDEWAYS=3
    
    label_map = {
        MarketRegime.BEAR: 'Bear',
        MarketRegime.BULL: 'Bull',
        MarketRegime.CRISIS: 'Crisis',
        MarketRegime.SIDEWAYS: 'Sideways'
    }
    
    regime_labels = regime_series.map(label_map)
    
    print("\nRegime Distribution:")
    print(regime_labels.value_counts(normalize=True))
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, np.log(df['close']), label='Log Price', color='black', alpha=0.6)
    
    colors = {
        MarketRegime.BEAR: 'orange', 
        MarketRegime.BULL: 'green', 
        MarketRegime.CRISIS: 'red', 
        MarketRegime.SIDEWAYS: 'blue'
    }
    
    # Fill areas
    # We need to align regime_series with df (returns dropped first row)
    aligned_regimes = regime_series.reindex(df.index).ffill().bfill()
    
    for regime_val, color in colors.items():
        mask = (aligned_regimes == regime_val)
        if mask.any():
             plt.fill_between(df.index, df['close'].min(), df['close'].max(), 
                              where=mask, color=color, alpha=0.3, 
                              transform=plt.gca().get_xaxis_transform(), 
                              label=label_map.get(regime_val, 'Unknown'))

    plt.title(f'HMM Regime Detection on {ticker}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figs/phase5', exist_ok=True)
    plt.savefig('figs/phase5/hmm_regimes.png')
    print("\nPlot saved to figs/phase5/hmm_regimes.png")

if __name__ == "__main__":
    label_regimes()
