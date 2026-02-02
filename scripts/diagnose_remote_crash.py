import os
import argparse
import paramiko
from dotenv import load_dotenv

load_dotenv()

def diagnose_crash(run_id):
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT") or 22)
    password = os.getenv("GPUHUB_PASSWORD")
    workspace = "/workspace/DeepScalper"
    
    print(f"--- Diagnosing Crash for Run: {run_id} ---")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {host}:{port}...")
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. List recent log files
        print(f"Searching for logs in {workspace}...")
        stdin, stdout, stderr = ssh.exec_command(f"ls -lt {workspace}/*.log {workspace}/*.txt | head -n 5")
        files = stdout.read().decode().strip()
        print(f"Recent Logs:\n{files}")
        
        # 2. Try to find a file containing the run_id, OR check the most recent nohup/debug log
        # Often output is redirected to a generic file if not named by run_id
        # We will try to grep the run_id in the recent files to find the right one
        
        target_file = None
        if files:
            for line in files.split('\n'):
                parts = line.split()
                if len(parts) > 8:
                    fname = parts[-1]
                    # Check if this file mentions the run_id
                    check_cmd = f"grep -l '{run_id}' {fname}"
                    stdin, out, err = ssh.exec_command(check_cmd)
                    if out.read().decode().strip():
                        target_file = fname
                        break
        
        # Fallback: check "run.log" then "nohup.out"
        if not target_file:
             stdin, out, err = ssh.exec_command(f"ls {workspace}/run.log")
             if out.read().decode().strip():
                 target_file = f"{workspace}/run.log"
             else:
                 stdin, out, err = ssh.exec_command(f"ls {workspace}/nohup.out")
                 if out.read().decode().strip():
                     target_file = f"{workspace}/nohup.out"

        if target_file:
            print(f"\nAnalyzing Log File: {target_file}")
            # Tail the last 50 lines
            stdin, stdout, stderr = ssh.exec_command(f"tail -n 50 {target_file}")
            log_tail = stdout.read().decode().strip()
            print(f"--- TAIL OF LOG ---\n{log_tail}\n-------------------")
            
            # Simple Heuristic Check
            if "CUDA out of memory" in log_tail:
                print("\n[DIAGNOSIS]: OOM (Out of Memory) Error detected.")
            elif "KeyError" in log_tail:
                print("\n[DIAGNOSIS]: KeyError detected (Configuration mismatch?).")
            elif "Traceback" in log_tail:
                print("\n[DIAGNOSIS]: Python Exception/Traceback found.")
            else:
                 print("\n[DIAGNOSIS]: No obvious error signature in tail. Review logs manually.")
                 
        else:
            print(f"\nCould not identify a specific log file for run {run_id}.")
            
    except Exception as e:
        print(f"SSH Error: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="WandB Run ID")
    args = parser.parse_args()
    diagnose_crash(args.run_id)
