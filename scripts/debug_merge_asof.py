import argparse
import sys
import os
import numpy as np

print("STEP 0: Importing Torch First...", flush=True)
import torch
if torch.cuda.is_available():
    print("CUDA Available.", flush=True)
    t = torch.tensor([1.0]).cuda()

print("STEP 1: Importing Pandas...", flush=True)
import pandas as pd

def main():
    print("STEP 2: Generating Data for Merge Asof...", flush=True)
    df1 = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    df1['timestamp'] = pd.to_datetime(df1['a'], unit='s')
    
    df2 = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]})
    df2['timestamp'] = pd.to_datetime(df2['a'], unit='s')
    
    print("STEP 3: Running merge_asof...", flush=True)
    try:
        merged = pd.merge_asof(
            df1.sort_values('timestamp'), 
            df2.sort_values('timestamp'), 
            on='timestamp', 
            direction='backward'
        )
        print("STEP 4: Merge Success.", flush=True)
        print(merged)
        print("FULL SUCCESS.", flush=True)
    except Exception as e:
        print(f"CRASHED: {e}", flush=True)

if __name__ == "__main__":
    main()
