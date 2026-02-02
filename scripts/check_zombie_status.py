import os
import argparse
import paramiko
import wandb
from dotenv import load_dotenv

load_dotenv()

def check_status(run_id, entity="bigcan-chiwin-technology", project="FinRL-Pro-DS"):
    print(f"--- Diagnosing Run: {entity}/{project}/{run_id} ---")
    
    # 1. WandB API Check
    print("\n[1/2] Checking WandB API Status...")
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        print(f"  > State: {run.state}")
        print(f"  > Heartbeat: {run.heartbeatAt}")
        print(f"  > Runtime: {run.summary.get('_runtime', 'Unknown')}s")
        if run.state == "running":
            print("  > Status: API reports RUNNING (Zombie candidate)")
        else:
            print("  > Status: API reports FINISHED/CRASHED")
            
    except Exception as e:
        print(f"  > Error querying API: {e}")

    # 2. Remote Process Check
    print("\n[2/2] Checking Remote Process (SSH)...")
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT") or 22)
    password = os.getenv("GPUHUB_PASSWORD")
    
    if not host or not password:
        print("  > Error: Missing GPUHUB credentials in .env")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        cmd = f"ps aux | grep '{run_id}' | grep -v grep"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        processes = stdout.read().decode().strip()
        
        if processes:
            print(f"  > Found Active Process(es):\n{processes}")
            print("  > Conclusion: Run is ALIVE.")
        else:
            print("  > No matching processes found on remote server.")
            print("  > Conclusion: Run is DEAD (Hardware Confirm).")
            
    except Exception as e:
        print(f"  > SSH Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="WandB Run ID")
    args = parser.parse_args()
    check_status(args.run_id)
