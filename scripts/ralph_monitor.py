import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from scripts.fetch_wandb_run import poll_run_until_complete

print("Starting Ralph Monitor...")
# Poll every 15 minutes, wait up to 24 hours
result = poll_run_until_complete(poll_interval=900, max_wait=86400, tag="Ralph_Autonomous")
print(f"Monitor Finished. Result: {result}")
