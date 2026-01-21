"""
Verify Alpaca Setup and Connection for Phase 8 Paper Trading.

This script checks:
1. Environment Variables (ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)
2. Connectivity to Alpaca Paper Trading API
3. Data Fetching (Bar Data format)
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure we can import finrl_pro_ds
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.execution.alpaca_broker import AlpacaBroker

def main():
    print("=== Alpaca Execution Handler Setup Verification ===")
    
    # 1. Load Environment
    # Explicitly look for .env in the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env_path = os.path.join(project_root, '.env')
    print(f"... Loading .env from: {env_path} ...")
    load_dotenv(dotenv_path=env_path)
    
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    
    if not key or not secret:
        print("[!] ERROR: Alpaca credentials not found.")
        print("    Please set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in your environment or .env file.")
        sys.exit(1)
    else:
        print(f"[OK] Credentials found (Key ID: {key[:4]}...)")

    # 2. Connect
    try:
        print("... Connecting to Alpaca (Paper) ...")
        broker = AlpacaBroker(paper=True)
        acct = broker.get_account()
        print(f"[OK] Connection Successful!")
        print(f"     Account Status: {acct['status']}")
        print(f"     Equity: ${acct['equity']}")
    except Exception as e:
        print(f"[!] Connection Failed: {e}")
        sys.exit(1)

    # 3. Test Data Fetching
    try:
        print("... Testing Data Fetch (AAPL, 1Day, Limit 5) ...")
        df = broker.get_bar_data(["AAPL"], timeframe="1Day", limit=5)
        
        print(f"[OK] Data Fetched. Shape: {df.shape}")
        print(f"     Columns: {df.columns.tolist()}")
        
        # Verify Format
        required_cols = ["tic", "date", "close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[!] Format Error: Missing columns {missing}")
        else:
            print("[OK] DataFrame format is correct (FinRL Standard).")
            print(df.head(2))
            
    except Exception as e:
        print(f"[!] Data Fetch Failed: {e}")
        sys.exit(1)

    print("\n=== Setup Complete: Alpaca Handler is Ready ===")

if __name__ == "__main__":
    main()
