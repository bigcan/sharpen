"""
Script to run the Phase 8 Paper Trading Loop.
"""
import os
import sys
import time
from finrl_pro_ds.execution.paper_trade import PaperTradingSystem

def main():
    print("Initializing Paper Trading System...")
    try:
        system = PaperTradingSystem()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    print("Entering Execution Loop (Daily)...")
    # In a real deployment, this might run as a cron job or a daemon.
    # For demonstration, we run one cycle and then wait.
    
    while True:
        try:
            system.run_cycle()
        except Exception as e:
            print(f"Error in cycle: {e}")
        
        # Wait for next day (simulation) or exit
        print("Cycle complete. Sleeping for 24 hours (demo mode - Ctrl+C to exit)...")
        time.sleep(86400)

if __name__ == "__main__":
    main()
