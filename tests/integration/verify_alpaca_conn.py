"""Verify Alpaca Broker Connectivity."""

import os
import sys
from finrl_pro.execution.alpaca_broker import AlpacaBroker

def main():
    print("--- Checking Alpaca Connectivity ---")
    
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    
    if not key or not secret:
        print("[!] Alpaca Credentials NOT found in environment.")
        print("Please set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.")
        # We don't fail the test suite if creds aren't there, just warn
        # This allows CI to pass without secrets
        return

    try:
        broker = AlpacaBroker(paper=True)
        acct = broker.get_account()
        print(f"[+] Connection Successful!")
        print(f"    Account ID: {acct['id']}")
        print(f"    Equity: ${acct['equity']}")
        print(f"    Status: {acct['status']}")
        
        positions = broker.get_positions()
        print(f"[+] Positions: {len(positions)}")
        for p in positions:
            print(f"    - {p['symbol']}: {p['qty']} @ {p['current_price']}")

    except Exception as e:
        print(f"[-] Connection Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
