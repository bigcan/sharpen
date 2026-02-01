import wandb
import time
from dotenv import load_dotenv

load_dotenv()

def check_step_growth():
    api = wandb.Api()
    run = api.run("bigcan-chiwin-technology/FinRL-Pro-DS/hlmj8s0t")
    
    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    
    # Get current step
    step1 = run.summary.get("train/global_step", 0)
    print(f"Step (T0): {step1:,}")
    
    print("Waiting 30s...")
    time.sleep(30)
    
    # Reload run to get fresh data
    run = api.run("bigcan-chiwin-technology/FinRL-Pro-DS/hlmj8s0t")
    step2 = run.summary.get("train/global_step", 0)
    print(f"Step (T+30s): {step2:,}")
    
    delta = step2 - step1
    print(f"Delta: {delta:,} steps")
    
    if delta > 0:
        rate = delta / 30
        print(f"Rate: {rate:.2f} steps/sec")
        remaining = 30000000 - step2
        eta_seconds = remaining / rate
        print(f"ETA: {eta_seconds/3600:.1f} hours ({eta_seconds/86400:.1f} days)")
        print("VERDICT: TRAINING IS ACTIVELY RUNNING")
    else:
        print("WARNING: No step advancement detected")

if __name__ == "__main__":
    check_step_growth()
