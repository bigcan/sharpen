"""
Test script to run the Phase 10 Synapse notebook locally.
This tests the data download and basic setup without Colab dependencies.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Phase 10: Synapse Arbitrator Validation - Local Test")
print("=" * 60)
print(f"\nPython version: {sys.version}")
print(f"yfinance version: {yf.__version__}")
print(f"pandas version: {pd.__version__}")

# Configuration
TICKERS = [
    "WMT", "KO", "PEP", "MCD", "NKE", "COST", "CL",
    "CSCO", "INTC", "ORCL", "IBM", "VZ", "T", "ADBE", "TXN",
    "BA", "CAT", "DIS", "PFE", "HON"
]

print(f"\nTickers to download: {len(TICKERS)}")
print(f"Date range: 2020-01-01 to 2024-12-31")

# Test data download
print("\n" + "=" * 60)
print("Testing Data Download")
print("=" * 60)

df_list = []
for tic in TICKERS:
    try:
        print(f"Downloading {tic}...", end=" ")
        data = yf.download(tic, start="2020-01-01", end="2024-12-31", progress=False, auto_adjust=False)
        
        # Debug: show what we got
        if data.empty:
            print(f"⚠ No data returned")
            continue
            
        print(f"Shape: {data.shape}, ", end="")
        
        # Handle multi-level columns from yfinance (common in newer versions)
        if isinstance(data.columns, pd.MultiIndex):
            print(f"MultiIndex detected, ", end="")
            data.columns = [col[0] for col in data.columns]  # Take first level
        
        data = data.reset_index()
        data['tic'] = tic
        
        # Standardize column names (handle both old and new yfinance formats)
        col_map = {
            'Date': 'date', 'date': 'date',
            'Open': 'open', 'open': 'open',
            'High': 'high', 'high': 'high',
            'Low': 'low', 'low': 'low',
            'Close': 'close', 'close': 'close',
            'Adj Close': 'adj_close', 'adj_close': 'adj_close',
            'Volume': 'volume', 'volume': 'volume'
        }
        data = data.rename(columns=col_map)
        
        # Select required columns (handle missing adj_close gracefully)
        if 'adj_close' not in data.columns:
            data['adj_close'] = data['close']
        
        required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'tic']
        data = data[required_cols]
        df_list.append(data)
        print(f"✓ {len(data)} rows")
    except Exception as e:
        import traceback
        print(f"✗ Error: {e}")
        traceback.print_exc()

print(f"\nSuccessfully downloaded: {len(df_list)}/{len(TICKERS)} tickers")

if df_list:
    df = pd.concat(df_list, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    print(f"\n✅ Total: {len(df)} rows for {df['tic'].nunique()} tickers")
    print("\nSample data:")
    print(df.head(10).to_string())
    print("\nData types:")
    print(df.dtypes)
    
    # Save to CSV for verification
    df.to_csv("test_download_data.csv", index=False)
    print(f"\n✅ Data saved to test_download_data.csv")
else:
    print("\n❌ No data downloaded! Trying batch download...")
    
    # Alternative: batch download
    try:
        data = yf.download(TICKERS, start="2020-01-01", end="2024-12-31", group_by='ticker', progress=True)
        print(f"Batch download shape: {data.shape}")
        print(f"Batch download columns (first 10): {list(data.columns[:10])}")
        
        if not data.empty:
            # Process batch download
            df_list = []
            for tic in TICKERS:
                try:
                    if tic in data.columns.get_level_values(0):
                        tic_data = data[tic].copy()
                    else:
                        continue
                    tic_data = tic_data.reset_index()
                    tic_data['tic'] = tic
                    tic_data.columns = [c.lower() if isinstance(c, str) else c for c in tic_data.columns]
                    if 'adj close' in tic_data.columns:
                        tic_data = tic_data.rename(columns={'adj close': 'adj_close'})
                    df_list.append(tic_data)
                    print(f"  ✓ Processed {tic}: {len(tic_data)} rows")
                except Exception as e:
                    print(f"  ✗ Error processing {tic}: {e}")
            
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                df['date'] = pd.to_datetime(df['date'])
                print(f"\n✅ Batch method succeeded: {len(df)} rows for {df['tic'].nunique()} tickers")
                df.to_csv("test_download_data.csv", index=False)
                print(f"Data saved to test_download_data.csv")
    except Exception as e:
        print(f"Batch download failed: {e}")
        import traceback
        traceback.print_exc()
