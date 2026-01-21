
import sys
import os
import json
import numpy as np
import pandas as pd
from finrl_pro_ds.data.snapshot import main as snapshot_main
from finrl_pro_ds.data.loader_pro import ProFeatureAssembler

def prepare_phase5_dataset():
    print("Starting Phase 5 Full Dataset Preparation (Liquid 20 + Macro)...")
    
    # 1. Define Multi-Asset Universe (Liquid 20 + Macro)
    liquid_20 = [
        # Indices
        'SPY', 'QQQ', 'IWM',
        # Sectors
        'XLK', 'XLF', 'XLE', 'XLV',
        # Top Stocks
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'JPM', 'XOM', 'JNJ', 'PG', 'V',
        # Defensive
        'GLD', 'TLT'
    ]
    
    macro = [
        '^VIX', # Volatility
        '^TNX', # 10-Year Treasury Yield
        'DX-Y.NYB' # US Dollar Index
    ]
    
    all_tickers = liquid_20 + macro
    
    start_date = '2000-01-01' 
    end_date = '2025-01-01'
    
    print(f"Fetching data for {len(all_tickers)} assets.")
    print(f"Liquid 20: {liquid_20}")
    print(f"Macro: {macro}")
    
    # 2. Create Snapshot (Simulate CLI call)
    # We need to capture stdout to get the snapshot_id
    from io import StringIO
    import contextlib
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        try:
            snapshot_main([
                '--provider', 'yahoo',
                '--tickers', ','.join(all_tickers),
                '--start', start_date,
                '--end', end_date,
                '--interval', '1d'
            ])
        except SystemExit as e:
            if e.code != 0:
                print(f"Snapshot creation failed with code {e.code}")
                # Print captured output for debugging
                print("Captured Output:\n", f.getvalue())
                return
    
    output = f.getvalue()
    # print("Snapshot Output:", output) # Verbose
    
    try:
        # Extract JSON from the last line or so
        lines = output.strip().split('\n')
        json_str = ""
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('}'):
                # Work backwards to find {
                for j in range(i, -1, -1):
                    if lines[j].strip().startswith('{'):
                        json_str = "\n".join(lines[j:i+1])
                        break
                break
        
        if not json_str:
             # Fallback: try parsing the whole thing if it's just json
             json_str = output
             
        result = json.loads(json_str)
        snapshot_id = result['snapshot_id']
        print(f"Snapshot created successfully: {snapshot_id}")
    except json.JSONDecodeError:
        print("Failed to parse snapshot output JSON")
        print("Raw Output:", output)
        return

    # 3. Load using ProFeatureAssembler to verify
    print("Loading data with ProFeatureAssembler to verify integrity...")
    assembler = ProFeatureAssembler()
    
    # Basic config
    features_cfg = {
        "families": {
            "momentum": ["rsi_14", "macd"],
            "volatility": ["atr_14"]
        },
        # We just want to verify we can calculate features for all
        "stockstats_overrides": ["close", "high", "low", "open", "volume"] 
    }
    
    assembly = assembler.assemble_from_snapshot(
        snapshot_id=snapshot_id,
        features_cfg=features_cfg
    )
    
    # 4. Verify Shapes
    T = len(assembly.dates)
    N = len(all_tickers)
    
    print(f"Time Steps: {T}")
    print(f"Assets: {N}")
    print(f"Price Array Shape: {assembly.price_ary.shape}")
    
    # Check for missing assets in the output
    missing = set(all_tickers) - set(assembly.tickers)
    if missing:
        print(f"WARNING: The following tickers are missing from the assembly: {missing}")
    else:
        print("All tickers present.")

    # 5. Check for NaNs in Macro Data specifically
    # Macro data often has different holidays or gaps
    print("\nChecking Macro Data availability:")
    df_prices = pd.DataFrame(assembly.price_ary, index=assembly.dates, columns=assembly.tickers)
    
    for m in macro:
        if m in df_prices.columns:
            null_count = df_prices[m].isnull().sum()
            total_count = len(df_prices)
            print(f"  {m}: {total_count - null_count}/{total_count} present ({null_count} NaNs)")
        else:
            print(f"  {m}: MISSING")

    print("\nPhase 5 Data Preparation COMPLETE")

if __name__ == "__main__":
    prepare_phase5_dataset()
