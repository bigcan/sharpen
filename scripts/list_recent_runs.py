
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def list_recent_runs():
    api = wandb.Api()
    project = "bigcan-chiwin-technology/FinRL-Pro-DS"
    print(f"Checking project: {project}")
    runs = api.runs(project, order="-created_at")
    
    print(f"{'Name':<40} {'ID':<15} {'Status':<15} {'Created':<25}")
    print("-" * 95)
    for i, run in enumerate(runs):
        if i >= 10: break
        print(f"{run.name[:39]:<40} {run.id:<15} {run.state:<15} {run.created_at:<25}")

if __name__ == "__main__":
    list_recent_runs()
