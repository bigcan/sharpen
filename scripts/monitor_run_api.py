
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
        for k, v in run.config.items():
            if k in relevant:
                print(f"  - {k}: {v}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # The run path provided by user: bigcan-chiwin-technology/FinRL-Pro-DS/vg8osmsc
    monitor_run("bigcan-chiwin-technology/FinRL-Pro-DS/vg8osmsc")
