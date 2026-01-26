import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def list_artifacts(run_path):
    api = wandb.Api()
    try:
        run = api.run(run_path)
        print(f"Artifacts for run {run.id}:")
        for artifact in run.logged_artifacts():
            print(f" - {artifact.name} ({artifact.type})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    RUN_PATH = "bigcan-chiwin-technology/FinRL-Pro-DS/514e17aa"
    list_artifacts(RUN_PATH)
