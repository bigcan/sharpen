
import wandb
import pandas as pd
import numpy as np
import sys

# Constants
ENTITY = "bigcan-chiwin-technology"
PROJECT = "FinRL-Pro-DS"
RUN_ID = "nwn7ck2e"

def calculate_real_sps():
    print(f"Fetching history for run {ENTITY}/{PROJECT}/{RUN_ID}...")
    api = wandb.Api()
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    except Exception as e:
        print(f"Error fetching run: {e}")
        return

    # Fetch full history (scan_history is more efficient for large histories)
    history_data = []
    # We specifically want 'train/sps', '_timestamp', 'train/global_step'
    # Also 'hpo/status' to distinguish phases if logged in history
    keys = ["_timestamp", "train/global_step", "train/sps", "hpo/t0/completed", "hpo/t1/completed", "hpo/t2/completed"]
    
    scan = run.scan_history(keys=keys)
    for row in scan:
        history_data.append(row)
    
    if not history_data:
        print("No history found.")
        return

    df = pd.DataFrame(history_data)
    df['_timestamp'] = pd.to_numeric(df['_timestamp'])
    
    print(f"Total Log Records: {len(df)}")
    
    # 1. Inspect Reported SPS (if available)
    if 'train/sps' in df.columns:
        valid_sps = df['train/sps'].dropna()
        if not valid_sps.empty:
            avg_sps = valid_sps.mean()
            median_sps = valid_sps.median()
            p95_sps = np.percentile(valid_sps, 95)
            print(f"\n[Reported Metric 'train/sps']")
            print(f"  Mean:   {avg_sps:.2f}")
            print(f"  Median: {median_sps:.2f}")
            print(f"  Max:    {valid_sps.max():.2f}")
        else:
            print("\n[Reported Metric 'train/sps'] Not found/Empty")
    
    # 2. Calculate Derived SPS (Step Delta / Time Delta) for MAIN TRAINING
    # precise scaling calculation requires filtering out HPO phase if possible
    
    # Identify Training Phase: Look for steady strictly increasing 'train/global_step'
    if 'train/global_step' in df.columns:
        train_df = df.dropna(subset=['train/global_step']).sort_values('train/global_step')
        
        if len(train_df) > 1:
            # Calculate rolling SPS to see stability
            train_df['time_delta'] = train_df['_timestamp'].diff()
            train_df['step_delta'] = train_df['train/global_step'].diff()
            train_df['calc_sps'] = train_df['step_delta'] / train_df['time_delta']
            
            # Filter unrealistic values (e.g. gaps between HPO and Train)
            # Standard training logging is filtered
            valid_calc = train_df[train_df['time_delta'] > 0]
            valid_calc = valid_calc[valid_calc['calc_sps'] < 2000] # remove outliers
            
            real_mean = valid_calc['calc_sps'].mean()
            real_median = valid_calc['calc_sps'].median()
            
            print(f"\n[Derived SPS (Steps / Time)]")
            print(f"  Mean:   {real_mean:.2f}")
            print(f"  Median: {real_median:.2f}")
            
            # Check for HPO gap
            # If there's a large time gap, that's likely the HPO -> Train transition
            # We want the speed of the dense training block
            
            print(f"  Values count: {len(valid_calc)}")

if __name__ == "__main__":
    calculate_real_sps()
