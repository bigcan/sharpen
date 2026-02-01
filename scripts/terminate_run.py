import os
import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

import os
import sys
import paramiko
import argparse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def terminate_run(target_run_id):
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")

    if not target_run_id:
        print("Error: No Run ID specified.")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}:{port}...")
        ssh.connect(host, port=port, username='root', password=password)
        
        print(f"Searching for processes matching run_id: {target_run_id}...")
        # Find PIDs with the run ID in the command line
        cmd_find = f"ps aux | grep '{target_run_id}' | grep -v grep"
        stdin, stdout, stderr = ssh.exec_command(cmd_find)
        process_lines = stdout.read().decode().strip().split('\n')
        
        pids_to_kill = []
        if process_lines and process_lines[0]:
            print(f"Found match(es):")
            for line in process_lines:
                print(f"  {line}")
                parts = line.split()
                if len(parts) > 1:
                    pids_to_kill.append(parts[1])
        else:
            print(f"No active processes found for {target_run_id}.")
            
            # Additional check: orphaned wandb-service processes?
            # It's hard to link wandb-service to a specific run ID solely by ps aux sometimes if run_id isn't in arg.
            # But usually 'wandb-service(RunId)' appears in some versions, or we check the log file handle.
            # For now, we trust grep.
            pass

        if pids_to_kill:
            print(f"Killing PIDs: {pids_to_kill}")
            kill_cmd = f"kill -9 {' '.join(pids_to_kill)}"
            ssh.exec_command(kill_cmd)
            print("Kill command sent.")
            
            # Verify
            stdin, stdout, stderr = ssh.exec_command(cmd_find)
            remaining = stdout.read().decode().strip()
            if remaining:
               print(f"WARNING: Processes still exist:\n{remaining}")
            else:
               print("SUCCESS: Target processes terminated.")
        else:
            print("Nothing to kill.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?", default="7n11ubgv", help="Run ID to terminate")
    args = parser.parse_args()
    terminate_run(args.run_id)
