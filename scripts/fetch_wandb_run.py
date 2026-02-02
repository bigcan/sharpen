import wandb
import os
import json
from dotenv import load_dotenv

load_dotenv()

def fetch_run_data(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS"):
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        data = {
            "name": run.name,
            "status": run.state,
            "config": run.config,
            "summary": run.summary._json_dict,
            "created_at": run.created_at,
            "duration": run.summary.get("_runtime", 0)
        }
        return data
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    run_id = "uiuivysu"
    print(f"Fetching data for run {run_id}...")
    data = fetch_run_data(run_id)
    
    with open("run_data_uiuivysu.json", "w") as f:
        json.dump(data, f, indent=4)
    
    if "error" in data:
        print(f"Error: {data['error']}")
    else:
        print(f"Successfully fetched data for {data['name']}")
