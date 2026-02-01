
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

def monitor_run(run_path):
    api = wandb.Api()
    try:
        run = api.run(run_path)
        print(f"\n🚀 Run: {run.name} ({run.id})")
        print(f"Status: {run.state.upper()}")
        print(f"Created: {run.created_at}")
        import datetime
        print(f"Now (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        
        # Get history for charts/trends
        print("\n📈 History Information:")
        history = run.history(samples=100)
        if not history.empty:
            print(f"  - Available Columns: {list(history.columns)[:10]}...")
            # Try to find a step column
            step_cols = [c for c in history.columns if 'step' in c.lower()]
            if step_cols:
                 print(f"  - Last Step ({step_cols[0]}): {history[step_cols[0]].iloc[-1]}")
            
        print("\n📰 Last Log Lines:")
        try:
            # Use internal _api to get logs if possible, or just print specific error if visible in summary
            for line in run.file("output.log").download(replace=True).readlines()[-10:]:
                print(f"  {line.strip()}")
        except:
             print("  - Could not retrieve output.log (may not exist or permission denied)")
            
        print("\n📈 Summary Highlights:")
        for k in ["train/global_step", "train/episode_reward", "best_trial/sharpe"]:
            if k in run.summary:
                print(f"  - {k}: {run.summary[k]}")

        print("\n📋 Config Snippet:")
        # Show key configs only
        relevant = ["strategy", "total_timesteps", "dqn_update_interval", "learning_rate"]
        total_timesteps = None
        for k, v in run.config.items():
            if k in relevant:
                print(f"  - {k}: {v}")
            if "total_timesteps" in k or k == "training" and isinstance(v, dict):
                if isinstance(v, dict) and "total_timesteps" in v:
                    total_timesteps = v.get("total_timesteps")
                elif k == "total_timesteps":
                    total_timesteps = v

        # ETA Calculation
        if "train/global_step" in run.summary and total_timesteps:
            current_step = run.summary["train/global_step"]
            if current_step and current_step > 0:
                import datetime
                created = datetime.datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                now = datetime.datetime.now(datetime.timezone.utc)
                elapsed = (now - created).total_seconds()
                
                steps_remaining = total_timesteps - current_step
                rate = current_step / elapsed  # steps per second
                seconds_remaining = steps_remaining / rate if rate > 0 else 0
                
                hours_remaining = int(seconds_remaining // 3600)
                mins_remaining = int((seconds_remaining % 3600) // 60)
                
                print(f"\n⏱️ ETA Calculation:")
                print(f"  - Total Target: {int(total_timesteps):,}")
                print(f"  - Current Step: {int(current_step):,}")
                print(f"  - Progress: {(current_step/total_timesteps)*100:.2f}%")
                print(f"  - Elapsed: {elapsed/3600:.1f} hours")
                print(f"  - Rate: {rate:.2f} steps/sec")
                print(f"  - Remaining: ~{hours_remaining}h {mins_remaining}m")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # The run path provided by user: bigcan-chiwin-technology/FinRL-Pro-DS/hlmj8s0t
    monitor_run("bigcan-chiwin-technology/FinRL-Pro-DS/hlmj8s0t")
