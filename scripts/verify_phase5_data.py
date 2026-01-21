
import sys
import os
import json
import numpy as np
from finrl_pro_ds.data.snapshot import main as snapshot_main
from finrl_pro_ds.data.loader_pro import ProFeatureAssembler

def verify_phase5_data():
    print("Starting Phase 5 Data Verification...")
    
    # 1. Define Multi-Asset Universe (Liquid 20)
    tickers = [
        # Indices
        'SPY', 'QQQ', 'IWM',
        # Sectors
        'XLK', 'XLF', 'XLE', 'XLV',
        # Top Stocks
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'JPM', 'XOM', 'JNJ', 'PG', 'V',
        # Defensive
        'GLD', 'TLT'
    ]
    start_date = '2000-01-01' 
    end_date = '2025-01-01'
    
    print(f"Fetching data for: {tickers}")
    
    # 2. Create Snapshot (Simulate CLI call)
    # We need to capture stdout to get the snapshot_id
    from io import StringIO
    import contextlib
    
    f = StringIO()
    with contextlib.redirect_stdout(f):
        try:
            snapshot_main([
                '--provider', 'yahoo',
                '--tickers', ','.join(tickers),
                '--start', start_date,
                '--end', end_date,
                '--interval', '1d'
            ])
        except SystemExit as e:
            if e.code != 0:
                print(f"Snapshot creation failed with code {e.code}")
                return
    
    output = f.getvalue()
    print("Snapshot Output:", output)
    
    try:
        result = json.loads(output)
        snapshot_id = result['snapshot_id']
        print(f"Snapshot created: {snapshot_id}")
    except json.JSONDecodeError:
        print("Failed to parse snapshot output JSON")
        return

    # 3. Load using ProFeatureAssembler
    print("Loading data with ProFeatureAssembler...")
    assembler = ProFeatureAssembler()
    
    # Basic config
    features_cfg = {
        "families": {
            "momentum": ["rsi_14", "macd"],
            "volatility": ["atr_14"]
        },
        "stockstats_overrides": ["close", "high", "low", "open", "volume", "rsi_14", "macd"] # Just pick 7
    }
    
    assembly = assembler.assemble_from_snapshot(
        snapshot_id=snapshot_id,
        features_cfg=features_cfg
    )
    
    # 4. Verify Shapes
    print("Verifying shapes...")
    T = len(assembly.dates)
    N = len(tickers)
    F = 7 # Default tech dim
    
    print(f"Price Array Shape: {assembly.price_ary.shape}")
    print(f"Tech Array Shape: {assembly.tech_ary.shape}")
    
    assert assembly.price_ary.shape == (T, N), f"Price shape mismatch. Expected ({T}, {N}), got {assembly.price_ary.shape}"
    assert assembly.tech_ary.shape == (T, N * F), f"Tech shape mismatch. Expected ({T}, {N*F}), got {assembly.tech_ary.shape}"
    
    # 5. Check for NaNs
    print("Checking for NaNs...")
    if np.isnan(assembly.price_ary).any():
        print("WARNING: NaNs found in price_ary")
    else:
        print("Price Array: OK (No NaNs)")
        
    if np.isnan(assembly.tech_ary).any():
        print("WARNING: NaNs found in tech_ary")
    else:
        print("Tech Array: OK (No NaNs)")
        
    print("Phase 5 Data Verification PASSED")

if __name__ == "__main__":
    verify_phase5_data()
