
import pandas as pd
import torch
import torch.nn as nn
import os
import sys

# Define the network architectures to compare
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def audit_capacity():
    data_path = "data/btc_lob_jan2023.parquet"
    
    print(f"--- DeepScalper Data/Model Capacity Audit ---")
    
    # 1. Data Audit
    if os.path.exists(data_path):
        try:
            df = pd.read_parquet(data_path)
            rows = len(df)
            print(f"[DATA] File: {data_path}")
            print(f"[DATA] Rows: {rows:,}")
            
            # Estimate effective samples (assuming window size overlaps, but unique information is row-based)
            effective_samples = rows 
        except Exception as e:
            print(f"[DATA] Error reading file: {e}")
            return
    else:
        print(f"[DATA] File not found: {data_path}. Using estimated 1.1M rows.")
        rows = 1100000
        effective_samples = rows

    print("-" * 30)

    # 2. Model Audit (The "Big" vs "Small" comparison)
    # Simplified representation of the BDQ Network (Micro + Macro streams)
    # This approximates the core dense layers which contain the bulk of parameters
    
    # Original: [1024, 1024, 512]
    big_model = SimpleMLP(input_size=31, hidden_sizes=[1024, 1024, 512], output_size=15) # 3 streams * 5 bins
    big_params = big_model.count_params()
    
    # New: [256, 128]
    small_model = SimpleMLP(input_size=31, hidden_sizes=[256, 128], output_size=15)
    small_params = small_model.count_params()

    # 3. Ratio Calculation
    print(f"[MODEL] Original Architecture [1024, 1024, 512]")
    print(f"[MODEL] Parameters: {big_params:,}")
    ratio_big = effective_samples / big_params
    print(f"[METRIC] Samples per Parameter: {ratio_big:.2f}")
    if ratio_big < 5.0:
        print("  -> CRITICAL RISK: < 5.0 indicates massive overfitting potential.")
    
    print("-" * 30)
    
    print(f"[MODEL] Optimized Architecture [256, 128]")
    print(f"[MODEL] Parameters: {small_params:,}")
    ratio_small = effective_samples / small_params
    print(f"[METRIC] Samples per Parameter: {ratio_small:.2f}")
    if ratio_small > 10.0:
        print("  -> HEALTHY: > 10.0 allows generalization.")

if __name__ == "__main__":
    audit_capacity()
