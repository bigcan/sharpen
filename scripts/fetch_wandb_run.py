import wandb
import json
import os

import argparse

def fetch_run_data(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS"):
    run_path = f"{entity}/{project}/{run_id}"
    output_file = f"c:/FinRL/FinRL-Pro_DS/results/run_data_{run_id}.json"
    
    print(f"Fetching run data for {run_path}...")
    api = wandb.Api()
    try:
        run = api.run(run_path)
        
        # Collect data
        data = {
            "id": run.id,
            "name": run.name,
            "status": run.state,
            "config": run.config,
            "summary": run.summary._json_dict,
            "tags": run.tags,
            "created_at": run.created_at,
            "url": run.url,
            "history": [] # Attempt to fetch history sample
        }
        
        # Fetch some history for performance metrics (SPS)
        # We process 'train/global_step' vs '_runtime' to calc SPS if not logged
        history = run.history(keys=["step", "train/global_step", "_runtime"], samples=100)
        data["history"] = history.to_dict(orient="records")

        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Successfully saved run data to {output_file}")
        
    except Exception as e:
        print(f"Error fetching run data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="WandB Run ID")
    args = parser.parse_args()
    
    fetch_run_data(args.run_id)
