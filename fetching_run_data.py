import wandb
import sys
import json

# Set stdout to use utf-8 encoding to avoid encoding errors
sys.stdout.reconfigure(encoding='utf-8')

run_path = "bigcan-chiwin-technology/FinRL-Pro-DS/udyl6rr6"
try:
    api = wandb.Api()
    run = api.run(run_path)
    
    print(f"Run Name: {run.name}")
    print(f"ID: {run.id}")
    print(f"Status: {run.state}")
    print("\n--- Config ---")
    print(json.dumps(run.config, indent=2))
    
    print("\n--- Summary Metrics ---")
    # Filter out large artifacts or non-serializable objects if any, though summary is usually dict
    print(json.dumps(run.summary._json_dict, indent=2))
    
    print("\n--- History (Last 5 steps) ---")
    history = run.history().tail(5)
    print(history)

except Exception as e:
    print(f"Error fetching run: {e}")
