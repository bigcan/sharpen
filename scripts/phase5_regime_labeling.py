import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from finrl_pro.data.loader_pro import ProFeatureAssembler

SNAPSHOT_ID = "36629e7c-ff6a-4585-9fcf-284058413028"

def label_regimes():
    print(f"Loading Snapshot {SNAPSHOT_ID}...")
    assembler = ProFeatureAssembler()
    # We only need price for regime labeling (SPY close, VIX close)
    # But we need to use the assembler to get the aligned dataframe
    features_cfg = {
        "stockstats_overrides": ["close"],
        "dataset_hash_source": "regime_labeling_v1"
    } 
    
    assembly = assembler.assemble_from_snapshot(
        snapshot_id=SNAPSHOT_ID,
        features_cfg=features_cfg
    )
    
    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(assembly.price_ary, index=assembly.dates, columns=assembly.tickers)
    
    # Extract Key Series
    spy = df['SPY']
    vix = df['^VIX']
    
    # Compute Indicators
    spy_sma200 = spy.rolling(window=200).mean()
    
    # Label Regimes
    # 0: Correction/Bear (Default)
    # 1: Bull
    # 2: Crisis
    
    regimes = pd.Series(index=df.index, data=0) # Default to Correction/Bear
    
    # Bull: VIX < 20 AND SPY > SMA200
    bull_mask = (vix < 20) & (spy > spy_sma200)
    regimes[bull_mask] = 1
    
    # Crisis: VIX > 30
    crisis_mask = (vix > 30)
    regimes[crisis_mask] = 2
    
    # Map to String Labels for Display
    label_map = {0: 'Correction/Bear', 1: 'Bull', 2: 'Crisis'}
    regime_labels = regimes.map(label_map)
    
    print("\nRegime Distribution (Full History):")
    print(regime_labels.value_counts(normalize=True))
    print(regime_labels.value_counts())
    
    # Define Splits
    train_end = '2019-01-01'
    val_end = '2021-01-01'
    
    train_mask = (df.index < train_end)
    val_mask = (df.index >= train_end) & (df.index < val_end)
    test_mask = (df.index >= val_end)
    
    print("\n--- Split Analysis ---")
    
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        subset = regime_labels[mask]
        print(f"\n{name} Set ({len(subset)} days):")
        dist = subset.value_counts(normalize=True)
        counts = subset.value_counts()
        for label in ['Bull', 'Correction/Bear', 'Crisis']:
            pct = dist.get(label, 0.0) * 100
            cnt = counts.get(label, 0)
            print(f"  {label:15s}: {cnt:4d} ({pct:5.1f}%)")
            
        # Warn if any regime is missing or too low
        if any(dist.get(l, 0) < 0.01 for l in ['Bull', 'Correction/Bear', 'Crisis']):
             print(f"  WARNING: One or more regimes are very scarce (<1%) in {name}!")

    # Generate Plot
    plt.figure(figsize=(15, 8))
    
    # Plot SPY (Log Scale)
    plt.subplot(2, 1, 1)
    plt.plot(df.index, np.log10(spy), label='SPY (Log)', color='black', linewidth=1)
    
    # Color background by regime
    # We need to find segments
    # Simple scatter for visualization
    colors = {0: 'orange', 1: 'green', 2: 'red'}
    # Create a colored strip at the bottom
    
    for label, code in {'Bull': 1, 'Correction': 0, 'Crisis': 2}.items():
         mask = (regimes == code)
         # This is slow for plotting, let's just fill between
         plt.fill_between(df.index, 0, 1, where=mask, color=colors[code], alpha=0.3, transform=plt.gca().get_xaxis_transform(), label=label)

    plt.title('SPY Price & Market Regimes')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, vix, label='VIX', color='purple', linewidth=1)
    plt.axhline(20, color='green', linestyle='--')
    plt.axhline(30, color='red', linestyle='--')
    plt.title('VIX & Thresholds')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figs/phase5', exist_ok=True)
    plt.savefig('figs/phase5/regimes.png')
    print(f"\nPlot saved to figs/phase5/regimes.png")
    
    # Export Regime Labels to CSV for Training usage
    regime_df = pd.DataFrame({'regime': regimes, 'regime_label': regime_labels}, index=df.index)
    regime_df.to_csv('data/phase5_regimes.csv')
    print("Regime labels exported to data/phase5_regimes.csv")

if __name__ == "__main__":
    label_regimes()