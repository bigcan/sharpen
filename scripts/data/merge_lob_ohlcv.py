"""
Merge LOB and OHLCV Data
Combines Micro (LOB) and Macro (OHLCV) features into a single Parquet file for DeepScalper.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from sharpen.data.feature_engineering import DeepScalperFeatureEngineer

# Ensure package is in path
sys.path.append(str(Path(__file__).parent.parent.parent))

def merge_data(lob_path: str, ohlcv_path: str, output_path: str):
    """
    Merges High-Freq LOB data with 1-min OHLCV data.
    """
    print(f"Loading LOB data from {lob_path}...")
    try:
        lob_df = pd.read_parquet(lob_path)
    except Exception as e:
        print(f"Error loading LOB data: {e}")
        return

    print(f"Loading OHLCV data from {ohlcv_path}...")
    try:
        ohlcv_df = pd.read_parquet(ohlcv_path)
    except Exception as e:
        print(f"Error loading OHLCV data: {e}")
        return

    print("Initializing Feature Engineer...")
    fe = DeepScalperFeatureEngineer()

    print("Processing Micro Features...")
    try:
        # Expected columns in LOB: timestamp, bid_price_1.., ask_price_1.., bid_vol_1.., ask_vol_1..
        # Check if we need to rename or if fetch_kaggle_lob normalized them
        # fetch_kaggle_lob normalizes to lowercase snake_case
        micro_features = fe.process_micro(lob_df)
    except Exception as e:
        print(f"Micro feature processing failed: {e}")
        return

    print("Processing Macro Features...")
    try:
        macro_features = fe.process_macro(ohlcv_df)
    except Exception as e:
        print(f"Macro feature processing failed: {e}")
        return

    print("Aligning Datasets (merge_asof)...")
    try:
        # Align macro to micro (forward fill macro steps)
        # DeepScalperFeatureEngineer has align_multimodal
        merged_df = fe.align_multimodal(micro_features, macro_features)
    except Exception as e:
        print(f"Alignment failed: {e}")
        return

    print(f"Saving merged dataset to {output_path}...")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure timestamp is a column, not just index
    if isinstance(merged_df.index, pd.DatetimeIndex):
        merged_df = merged_df.reset_index()
        # Rename index col if it became 'index' or something else, but usually it keeps name 'timestamp' if set
        if 'timestamp' not in merged_df.columns and 'index' in merged_df.columns:
             merged_df = merged_df.rename(columns={'index': 'timestamp'})

    merged_df.to_parquet(output_path, index=False)
    print("Done.")
    print(f"Final Shape: {merged_df.shape}")
    print(f"Columns: {merged_df.columns.tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lob", required=True, help="Path to processed LOB parquet")
    parser.add_argument("--input-ohlcv", required=True, help="Path to OHLCV parquet")
    parser.add_argument("--output", default="c:/data/btc_lob_jan2023.parquet", help="Final output path")

    args = parser.parse_args()

    merge_data(args.input_lob, args.input_ohlcv, args.output)
