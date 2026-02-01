import paramiko
import os
import time
import argparse
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

def get_env_var(key):
    val = os.getenv(key)
    if not val:
        print(f"ERROR: Missing environment variable: {key}")
        sys.exit(1)
    return val

def monitor_loop(interval=300, workspace="/workspace/DeepScalper"):
    host = get_env_var("GPUHUB_HOST")
    port = int(get_env_var("GPUHUB_PORT"))
    password = get_env_var("GPUHUB_PASSWORD")
    
    print(f"=== HPO Monitor Agent Started ===")
    print(f"Target: {host}:{port}")
    print(f"Workspace: {workspace}")
    print(f"Interval: {interval} seconds")
    print(f"=================================\n")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        print("Connected to remote server successfully.\n")
        
        last_log_pos = 0
        
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Checking status...", end=" ", flush=True)
            
            # 1. Check if process is running
            stdin, stdout, stderr = ssh.exec_command(f"cat {workspace}/run.pid")
            pid = stdout.read().decode().strip()
            
            if not pid or not pid.isdigit():
                print(f"WARNING: No PID found in {workspace}/run.pid. Is the process running?")
            else:
                # Check if PID is actually active
                stdin, stdout, stderr = ssh.exec_command(f"ps -p {pid} -o comm=")
                process_name = stdout.read().decode().strip()
                if not process_name:
                    print(f"CRITICAL: Process {pid} is NOT running!")
                else:
                    # 2. Get Trial Info from logs
                    # We look for "Trial X finished" or "Best is trial" lines
                    cmd = f"grep 'Trial' {workspace}/run.log | tail -n 1"
                    stdin, stdout, stderr = ssh.exec_command(cmd)
                    latest_trial_log = stdout.read().decode().strip()
                    
                    # 3. Get Error status
                    cmd_err = f"grep -i 'Error' {workspace}/run.log | tail -n 1"
                    stdin, stdout, stderr = ssh.exec_command(cmd_err)
                    latest_error = stdout.read().decode().strip()
                    
                    print(f"(PID: {pid})")
                    if latest_trial_log:
                        print(f"    > Latest Status: {latest_trial_log}")
                    else:
                         print(f"    > Status: No 'Trial' logs found yet. (Starting up?)")
                         
                    if latest_error:
                        print(f"    > ! ALERT ! Possible Error Found: {latest_error}")
            
            # Sleep
            time.sleep(interval)
            
            # Re-establish connection check if needed (Paramiko usually keeps alive but good to be safe)
            if ssh.get_transport() is None or not ssh.get_transport().is_active():
                 print("\nConnection lost. Reconnecting...")
                 ssh.connect(host, port=port, username='root', password=password)

    except KeyboardInterrupt:
        print("\nStopping Monitor Agent.")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Remote HPO Progress")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 300)")
    parser.add_argument("--workspace", type=str, default="/workspace/DeepScalper", help="Remote workspace path")
    
    args = parser.parse_args()
    
    monitor_loop(interval=args.interval, workspace=args.workspace)
