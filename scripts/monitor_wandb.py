import wandb
import sys
import datetime
import time

def monitor_run(run_path):
    print(f"Connecting to WandB to monitor: {run_path}...")
    api = wandb.Api()
    
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Error fetching run: {e}")
        return

    print(f"run_id: {run.id}")
    print(f"name: {run.name}")
    print(f"state: {run.state}")
    
    # Heartbeat check
    if run.heartbeatAt:
        heartbeat_time = datetime.datetime.fromisoformat(run.heartbeatAt.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - heartbeat_time
        print(f"Last Heartbeat: {delta.total_seconds():.1f}s ago ({run.heartbeatAt})")
        
        if run.state == "running" and delta.total_seconds() > 300: # 5 minutes
            print("WARNING: Run is 'running' but no heartbeat for > 5 mins. Potentially ZOMBIE.")
    else:
        print("Last Heartbeat: None")

    # Metrics check
    history = run.history(samples=50) # Get last 50
    if not history.empty:
        print("\nRecent Metrics (Last 5):")
        print(history.tail(5)[['global_step', 'train/reward', 'fps'] if 'fps' in history.columns else history.columns[:5]])
        
        if 'fps' in history.columns:
            last_fps = history['fps'].iloc[-1]
            print(f"\nCurrent FPS: {last_fps}")
            if last_fps < 5:
                print("CRITICAL: FPS is very low (< 5)!")
            elif last_fps < 15:
                print("WARNING: FPS is low (< 15).")
    else:
        print("\nNo metrics history found yet.")

    print(f"\nRun URL: {run.url}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_wandb.py <entity/project/run_id>")
        sys.exit(1)
    
    run_path = sys.argv[1]
    monitor_run(run_path)
