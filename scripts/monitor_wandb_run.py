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
        
        # 1. Find Process and Heavy PID
        print(f"\n--- Finding Process for {run_id_fragment} ---")
        cmd_ps_all = "ps aux | grep 'hlmj8s0t' | grep -v grep | sort -nr -k 3"
        stdin, stdout, stderr = ssh.exec_command(cmd_ps_all)
        ps_lines = stdout.read().decode().strip().split('\n')
        
        target_pid = None
        if ps_lines and ps_lines[0]:
             parts = ps_lines[0].split()
             if len(parts) > 1:
                 target_pid = parts[1]
                 print(f"Found Target PID: {target_pid} (Heavy Load)")
                 print(f"Top Process: {ps_lines[0]}")
        else:
            print("No matching process found.")
            return

        # 2. Check File Descriptors (Where is output going?)
        log_file = None
        if target_pid:
            print(f"\n--- Output Redirection for PID {target_pid} ---")
            cmd_fd = f"ls -l /proc/{target_pid}/fd/1 /proc/{target_pid}/fd/2"
            stdin, stdout, stderr = ssh.exec_command(cmd_fd)
            fd_out = stdout.read().decode()
            print(fd_out)
            
            # Detect log file from FD output
            # Format: l-wx------ ... -> /workspace/DeepScalper/run.log
            match = re.search(r'->\s+(/.+\.log)', fd_out)
            if match:
                log_file = match.group(1)
                print(f"Detected Active Log File: {log_file}")
            else:
                # Fallback
                log_file = "/workspace/DeepScalper/run.log"
                print(f"Could not detect from FD, defaulting to {log_file}")

        # 3. Check Log Freshness
        cmd_date = "date +%s"
        stdin, stdout, stderr = ssh.exec_command(cmd_date)
        remote_now = int(stdout.read().decode().strip())
        
        cmd_stat = f"stat -c %Y {log_file}"
        stdin, stdout, stderr = ssh.exec_command(cmd_stat)
        try:
            log_mtime = int(stdout.read().decode().strip())
            diff = remote_now - log_mtime
            print(f"\n--- Freshness Check ---")
            print(f"Remote Time: {remote_now}")
            print(f"Log Mod Time: {log_mtime}")
            print(f"Time Since Last Write: {diff} seconds")
            
            if diff > 300:
                print("WARNING: No log output for > 5 minutes. Process might be STALLED or HUNG.")
        except:
             print(f"Could not stat log file {log_file}")

        # 4. Analyze Max Drawdown Loop
        print(f"\n--- Drawdown Diagnosis ---")
        cmd_count = f"grep -c 'Hit Max Drawdown Stop' {log_file}"
        stdin, stdout, stderr = ssh.exec_command(cmd_count)
        try:
            count = stdout.read().decode().strip()
            print(f"Total Drawdown Stops: {count}")
        except:
            print("Could not grep log file.")
        
        # 5. Find Latest Simulation Date
        # Look for date patterns YYYY-MM-DD
        print(f"\n--- Latest Simulation Date ---")
        cmd_date_sim = f"grep -oE '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' {log_file} | tail -n 1"
        stdin, stdout, stderr = ssh.exec_command(cmd_date_sim)
        latest_date = stdout.read().decode().strip()
        print(f"Latest Date in Logs: {latest_date}")
        
        # 6. GPU Usage Check
        print(f"\n--- NVIDIA-SMI (GPU Usage) ---")
        cmd_gpu = "nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv,noheader"
        stdin, stdout, stderr = ssh.exec_command(cmd_gpu)
        print(stdout.read().decode())

        # 7. Real Tail
        # print(f"\n--- Tail of {log_file} (Last 20 lines) ---")
        # cmd_tail = f"tail -n 20 {log_file}"
        # stdin, stdout, stderr = ssh.exec_command(cmd_tail)
        # print(stdout.read().decode())
        
        # 8. Check File Size growth
        cmd_du = f"ls -l --block-size=M {log_file}"
        stdin, stdout, stderr = ssh.exec_command(cmd_du)
        print(f"\nLog Size: {stdout.read().decode().strip()}")

    except Exception as e:
        print(f"SSH/Parsing Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    monitor_run("hlmj8s0t")
