import argparse
import sys
import os

def load_data():
    # Simulate Parquet Load
    print("Loading Parquet with PyArrow...", flush=True)
    import pandas as pd
    import numpy as np
    # Create dummy parquet
    df = pd.DataFrame(np.random.randn(100, 20), columns=[f"col_{i}" for i in range(20)])
    df.to_parquet("temp.parquet", engine='pyarrow')
    
    df_loaded = pd.read_parquet("temp.parquet", engine='pyarrow')
    print(f"Loaded Parquet. Shape: {df_loaded.shape}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=["torch_first", "pandas_first"], required=True)
    args = parser.parse_args()

    print(f"Testing Import Order: {args.order}", flush=True)

    if args.order == "torch_first":
        print("Importing Torch...", flush=True)
        import torch
        if torch.cuda.is_available():
            print("CUDA Available.", flush=True)
            t = torch.tensor([1.0]).cuda()
        
        print("Importing Pandas...", flush=True)
        import pandas as pd
        load_data()
        
    elif args.order == "pandas_first":
        print("Importing Pandas...", flush=True)
        import pandas as pd
        load_data()
        
        print("Importing Torch...", flush=True)
        import torch
        if torch.cuda.is_available():
            print("CUDA Available.", flush=True)
            t = torch.tensor([1.0]).cuda()

    print("Import Order Test SUCCESS.", flush=True)

if __name__ == "__main__":
    main()
