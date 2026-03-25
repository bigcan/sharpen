"""
Fetch Kaggle LOB Data
Download Bitcoin Perpetual LOB data (250ms coverage) from Kaggle.
"""
import os
import argparse
import glob
import subprocess
import pandas as pd
from pathlib import Path

def download_kaggle_lob(output_dir: str):
    """Downloads dataset using Kaggle CLI."""
    dataset = "siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset} to {output_path}...")
    try:
        ret = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_path), "--unzip"],
            check=False,
        )
        if ret.returncode != 0:
            raise RuntimeError("Kaggle download failed. Is kaggle CLI installed and configured?")
    except Exception as e:
        print(f"Error: {e}")
        return False

    return True

def process_kaggle_files(input_dir: str, output_file: str):
    """Reads raw CSVs, standardizes them, and saves to Parquet."""
    print(f"Processing CSV files in {input_dir}...")
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print("No CSV files found!")
        return

    dfs = []
    # Kaggle dataset structure: usually split by days or chunks
    # We need to inspect one to know the columns, but based on description:
    # 250ms snapshots, 10 levels.

    for f in csv_files:
        print(f"Reading {f}...")
        try:
            # Load raw CSV. The header is likely numeric indices.
            df = pd.read_csv(f)

            # Data Structure Analysis:
            # Col 0: Index (ignore)
            # Col 1: Timestamp (ms)
            # Col 2: Timestamp (str)
            # Cols 3-22: Bids (Price, Volume) * 10 levels (Descending Price)
            # Cols 23-42: Asks (Price, Volume) * 10 levels (Ascending Price)

            # Map columns by index
            # Extract data values (skip header/index if needed, but read_csv handles header)

            # Create a clean dataframe
            clean_df = pd.DataFrame()

            # Timestamp (Col 1) - Check name, likely '1' or similar due to numeric header
            # Or retrieve by position iloc[:, 1]
            clean_df['timestamp'] = df.iloc[:, 1]

            # Bids: Cols 3 to 22
            # 10 levels
            for i in range(1, 11):
                base_idx = 3 + (i-1)*2
                clean_df[f'bid_price_{i}'] = df.iloc[:, base_idx]
                clean_df[f'bid_vol_{i}'] = df.iloc[:, base_idx+1]

            # Asks: Cols 23 to 42
            for i in range(1, 11):
                base_idx = 23 + (i-1)*2
                clean_df[f'ask_price_{i}'] = df.iloc[:, base_idx]
                clean_df[f'ask_vol_{i}'] = df.iloc[:, base_idx+1]

            dfs.append(clean_df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if not dfs:
        return

    print("Concatenating...")
    full_df = pd.concat(dfs, ignore_index=True)

    print("Sorting and Deduplicating...")
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='ms') # Usually ms for crypto data
    full_df = full_df.sort_values('timestamp').drop_duplicates('timestamp')

    print(f"Saving {len(full_df)} rows to {output_file}...")
    full_df.to_parquet(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="c:/data/raw/kaggle_lob", help="Download directory")
    parser.add_argument("--output-file", default="c:/data/raw/kaggle_lob.parquet", help="Final parquet output")
    parser.add_argument("--validate-only", action="store_true", help="Only validate download")
    args = parser.parse_args()

    if not args.validate_only:
        if download_kaggle_lob(args.output_dir):
            process_kaggle_files(args.output_dir, args.output_file)
    else:
        # Validation checks
        p = Path(args.output_dir)
        if p.exists() and any(p.glob("*.csv")):
            print("Validation Passed: Data found.")
        else:
            print("Validation Failed: No data found.")
