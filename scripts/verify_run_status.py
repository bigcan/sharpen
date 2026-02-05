import wandb
import sys

try:
    api = wandb.Api()
    run = api.run('bigcan-chiwin-technology/FinRL-Pro-DS/9tyrvz4b')
    summary = run.summary
    
    hpo_status = summary.get("hpo/status", "N/A")
    train_status = summary.get("train/status", "N/A")
    train_started = summary.get("train/started", "N/A")
    
    print(f"HPO Status: {hpo_status}")
    print(f"Train Status: {train_status}")
    print(f"Train Started: {train_started}")
    
    train_keys = [k for k in summary.keys() if k.startswith('train/')]
    print(f"Training Metrics Found: {len(train_keys)}")
    
    if len(train_keys) > 0:
        print("CONFIRMED: Run transitioned to training phase.")
    else:
        print("NEGATIVE: No training metrics found.")
        
except Exception as e:
    print(f"Error: {e}")
