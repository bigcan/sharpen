import wandb
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

def list_recent_runs(entity="bigcan-chiwin-technology", project="FinRL-Pro-DS"):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    
    recent_runs = []
    for run in runs[:10]:
        recent_runs.append({
            "id": run.id,
            "name": run.name,
            "status": run.state,
            "created_at": run.created_at,
            "tags": run.tags,
            "summary": run.summary._json_dict
        })
    return recent_runs

if __name__ == "__main__":
    print("Listing recent runs...")
    runs = list_recent_runs()
    with open("recent_runs.json", "w") as f:
        json.dump(runs, f, indent=4)
    print("Saved recent runs to recent_runs.json")
