import argparse
import sys
import os
import pandas as pd
import numpy as np

def run_test():
    print("Testing TA + Torch Interaction...", flush=True)
    # Generate dummy data
    df = pd.DataFrame({'close': np.random.randn(100) + 100})
    
    # Run TA function
    print("Running TA (RSI)...", flush=True)
    from ta.momentum import RSIIndicator
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()
    print("TA Success.", flush=True)
    
    # Run Torch
    print("Running Torch...", flush=True)
    import torch
    if torch.cuda.is_available():
        t = torch.tensor([1.0]).cuda()
        print("Torch CUDA Success.", flush=True)
    else:
        print("Torch CPU Success.", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=["torch_first", "ta_first"], required=True)
    args = parser.parse_args()

    print(f"Testing Order: {args.order}", flush=True)

    if args.order == "torch_first":
        import torch
        print("Imported Torch.", flush=True)
        import ta
        print("Imported TA.", flush=True)
        run_test()
    
    elif args.order == "ta_first":
        import ta
        print("Imported TA.", flush=True)
        import torch
        print("Imported Torch.", flush=True)
        run_test()

if __name__ == "__main__":
    main()
