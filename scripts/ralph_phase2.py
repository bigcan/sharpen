import sys
import os

# Add cwd to path to find scripts module
sys.path.append(os.getcwd())

from scripts.fetch_wandb_run import poll_run_until_complete

print("Polling WandB for Ralph_Autonomous run...")
# Poll every 15 min (900s), max wait 8 hours (28800s)
try:
    result = poll_run_until_complete(poll_interval=900, max_wait=28800, tag="Ralph_Autonomous")
    print(f"Run completed with state: {result.get('state')}")
    print(f"Run ID: {result.get('run_id')}")
except Exception as e:
    print(f"Error polling run: {e}")
    sys.exit(1)
