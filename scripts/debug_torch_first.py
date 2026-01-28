import argparse
import sys
import os
# FORCE IMPORT ORDER: Torch First
print("Importing TEST: Torch First...", flush=True)

import torch
if torch.cuda.is_available():
    print("CUDA Available.", flush=True)
    t = torch.tensor([1.0]).cuda()

import pandas as pd
import numpy as np
# Explicitly PyArrow if needed
import pyarrow

def load_data():
    # Simulate Parquet Load
    print("Loading Parquet with PyArrow...", flush=True)
    # Create dummy parquet
    df = pd.DataFrame(np.random.randn(100, 20), columns=[f"col_{i}" for i in range(20)])
    df.to_parquet("temp.parquet", engine='pyarrow')
    
    df_loaded = pd.read_parquet("temp.parquet", engine='pyarrow')
    print(f"Loaded Parquet. Shape: {df_loaded.shape}", flush=True)

def main():
    print("Running Main Logic...", flush=True)
    load_data()
    print("Torch First Test SUCCESS.", flush=True)

if __name__ == "__main__":
    main()
