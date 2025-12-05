import yfinance as yf
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import config

def download_data():
    """Downloads data for tickers specified in config.py."""
    print(f"Downloading data for {len(config.TICKERS)} tickers from {config.START_DATE} to {config.END_DATE}...")
    
    df_list = []
    for tic in config.TICKERS:
        try:
            # Download data
            data = yf.download(tic, start=config.START_DATE, end=config.END_DATE, progress=False, auto_adjust=False)
            
            if data.empty:
                print(f"  Warning: No data for {tic}")
                continue
                
            # Handle MultiIndex columns if present (yfinance update)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
                
            data = data.reset_index()
            data['tic'] = tic
            
            # Rename columns to standard format
            data = data.rename(columns={
                'Date': 'date', 
                'Open': 'open', 
                'High': 'high', 
                'Low': 'low', 
                'Close': 'close', 
                'Adj Close': 'adj_close', 
                'Volume': 'volume'
            })
            
            # Ensure adj_close exists
            if 'adj_close' not in data.columns:
                data['adj_close'] = data['close']
                
            # Select required columns
            data = data[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'tic']]
            df_list.append(data)
            # print(f"  ✓ {tic}: {len(data)} rows")
            
        except Exception as e:
            print(f"  ✗ Error downloading {tic}: {e}")

    if not df_list:
        raise ValueError("No data downloaded!")

    df = pd.concat(df_list, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

def preprocess_data(df):
    """Adds technical indicators to the dataframe."""
    print("Adding technical indicators...")
    result = []
    for tic in df['tic'].unique():
        tic_df = df[df['tic'] == tic].copy().sort_values('date').reset_index(drop=True)
        
        # Use stockstats for indicators
        stock = StockDataFrame.retype(tic_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy())
        
        for ind in config.INDICATORS:
            tic_df[ind] = stock[ind].values
            
        result.append(tic_df)

    df_processed = pd.concat(result, ignore_index=True)
    
    # Fill NaN values (forward fill then backward fill)
    for ind in config.INDICATORS:
        df_processed[ind] = df_processed.groupby('tic')[ind].transform(lambda x: x.ffill().bfill())
        
    df_processed = df_processed.dropna()
    print(f"✅ Final dataset: {len(df_processed)} rows")
    return df_processed

def get_rolling_windows(df):
    """Generates train/test splits based on rolling window configuration."""
    dates = sorted(df['date'].unique())
    total_days = len(dates)
    
    windows = []
    current_idx = 0
    
    while current_idx + config.TRAIN_WINDOW_SIZE + config.TEST_WINDOW_SIZE <= total_days:
        train_start_idx = current_idx
        train_end_idx = current_idx + config.TRAIN_WINDOW_SIZE
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + config.TEST_WINDOW_SIZE
        
        train_start_date = dates[train_start_idx]
        train_end_date = dates[train_end_idx - 1] # Inclusive
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx - 1]   # Inclusive
        
        windows.append({
            'train_start': train_start_date,
            'train_end': train_end_date,
            'test_start': test_start_date,
            'test_end': test_end_date
        })
        
        current_idx += config.ROLLING_STEP_SIZE
        
    return windows

if __name__ == "__main__":
    # Test the data loader
    df = download_data()
    df = preprocess_data(df)
    windows = get_rolling_windows(df)
    print(f"Generated {len(windows)} rolling windows.")
    for i, w in enumerate(windows):
        print(f"Window {i+1}: Train [{w['train_start'].date()} -> {w['train_end'].date()}] | Test [{w['test_start'].date()} -> {w['test_end'].date()}]")
