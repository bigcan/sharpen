import pandas as pd
import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro.data.yahoo_loader import YahooLoader

def main():
    output_file = "data/sp500_phase9_2005_2025.parquet"
    
    # Tickers from Manifest
    tickers = [
        # Consumer
        "WMT", "KO", "PEP", "MCD", "NKE", "COST", "CL",
        # Tech/Telecom
        "CSCO", "INTC", "ORCL", "IBM", "VZ", "T", "ADBE", "TXN",
        # Industrial/Pharma
        "BA", "CAT", "DIS", "PFE", "HON"
    ]
    
    start_date = "2005-01-01"
    end_date = "2025-01-01"
    
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    loader = YahooLoader()
    df = loader.fetch(tickers=tickers, start=start_date, end=end_date)
    
    # Post-processing to ensure format matches expectations (timestamp, ticker, open, high, low, close, volume)
    # YahooLoader.fetch might return MultiIndex columns if multiple tickers are passed, 
    # or a stacked dataframe. Let's check the output structure by running a small test or handling it robustly.
    # The YahooLoader.fetch in `finrl_pro/data/yahoo_loader.py` (from previous read) seems to handle standardization 
    # but let's double check the behavior for multiple tickers. 
    # Actually, looking at the code I read earlier:
    # "If multiple tickers, columns are MultiIndex: (ticker, field)"
    # So I need to stack it to get the long format expected by FinRL.
    
    if isinstance(df.columns, pd.MultiIndex):
        # Reshape MultiIndex columns (Ticker, Price) -> Long format
        df = df.stack(level=0).reset_index()
        # Rename columns to match FinRL standard
        # Expected: timestamp, ticker, open, high, low, close, volume
        # Currently likely: Date, level_1 (Ticker), Close, High, Low, Open, Volume
        
        # Check column names
        # The stack puts the top level (Price or Ticker?) depending on yfinance version/loader.
        # yfinance group_by='ticker' usually gives Top Level = Ticker.
        # So stack(level=0) puts Ticker into a column.
        pass
    
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    
    # Map common names
    rename_map = {
        'date': 'timestamp',
        'level_1': 'ticker', # If stacked from (Ticker, Price)
        'adj close': 'close', # Prefer adjusted close if available, but loader might just give 'close'
    }
    df = df.rename(columns=rename_map)
    
    # Ensure we have 'ticker' column. If not, and it's a single ticker, we might need to add it?
    # With list of tickers, we definitely need to handle the structure.
    # Let's just rely on a robust reshaping logic that prints head if unsure.
    # Actually, creating a robust script is better.
    
    # Let's re-write the reshaping part to be sure.
    # Yfinance with group_by='ticker' returns:
    #           AAPL                  MSFT
    #           Open High ...         Open High ...
    # Date
    
    # df.stack(level=0) results in:
    #                     Open High ...
    # Date       Ticker
    
    # reset_index() results in:
    # Date, Ticker, Open, High ...
    
    # Perfect.
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
    # Ensure sorted
    if 'timestamp' in df.columns and 'ticker' in df.columns:
        df = df.sort_values(['timestamp', 'ticker'])
        
    # Save
    os.makedirs("data", exist_ok=True)
    df.to_parquet(output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
