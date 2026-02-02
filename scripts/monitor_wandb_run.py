import os
import paramiko
import json
import time
import re
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def monitor_run(run_id_fragment):
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    if not all([host, port, password]):
        print("Error: Missing env vars GPUHUB_HOST, GPUHUB_PORT, or GPUHUB_PASSWORD")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}:{port}...")
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. List ALL Python Processes (to find hidden/zombie processes)
        print(f"\n--- Process Search (All Python) ---")
        cmd_ps = "ps aux | grep python | grep -v grep"
        stdin, stdout, stderr = ssh.exec_command(cmd_ps)
        ps_out = stdout.read().decode().strip()
        print(ps_out if ps_out else "No python processes found.")

        # 2. Check Remote Config (to handle 5k vs 100k mismatch)
        print(f"\n--- Remote Config Check (deepscalper_pilot_test.yaml) ---")
        cmd_cat_conf = "cat /workspace/DeepScalper/configs/deepscalper_pilot_test.yaml"
        stdin, stdout, stderr = ssh.exec_command(cmd_cat_conf)
        print(stdout.read().decode().strip())

        # 3. Check Head of Run Log (for logged config parameters)
        print(f"\n--- Log Head (First 50 lines) ---")
        cmd_head = "head -n 50 /workspace/DeepScalper/run.log"
        stdin, stdout, stderr = ssh.exec_command(cmd_head)
        print(stdout.read().decode())

        # 4. Check Tail of Run Log
        print(f"\n--- Log Tail (Last 20 lines) ---")
        cmd_tail = "tail -n 20 /workspace/DeepScalper/run.log"
        stdin, stdout, stderr = ssh.exec_command(cmd_tail)
        print(stdout.read().decode())

        # Check for Adaptive Fold patch
        print(f"\n--- Patch Verification ---")
        cmd_grep_patch = "grep 'Adaptive Fold' /workspace/DeepScalper/run.log"
        stdin, stdout, stderr = ssh.exec_command(cmd_grep_patch)
        patch_out = stdout.read().decode().strip()
        if patch_out:
            print(f"SUCCESS: Patch Active! (Found 'Adaptive Fold')")
        else:
            print("WARNING: 'Adaptive Fold' not found. Patch might be missing.")

        print(f"\n--- HPO Results ---")
        # Grep Best Sharpe
        cmd_best = "grep 'Best Sharpe:' /workspace/DeepScalper/run.log | tail -n 1"
        stdin, stdout, stderr = ssh.exec_command(cmd_best)
        sharpe_out = stdout.read().decode().strip()
        if sharpe_out:
            print(f"{sharpe_out}")
        else:
            print("Best Sharpe not yet found in logs.")

        # Cat best_params.yaml
        print(f"\n--- Best Parameters (best_params.yaml) ---")
        cmd_cat_params = "cat /workspace/DeepScalper/best_params.yaml"
        stdin, stdout, stderr = ssh.exec_command(cmd_cat_params)
        params_out = stdout.read().decode().strip()
        if params_out and "No such file" not in params_out:
            print(params_out)
        else:
            print("best_params.yaml not found (HPO might still be running).")


    except Exception as e:
        print(f"SSH/Parsing Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id_fragment", help="Partial Run ID to search for (e.g. 7n11ubgv)")
    args = parser.parse_args()
    monitor_run(args.run_id_fragment)
