import os
import paramiko
import time
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def live_status_check(run_id_fragment):
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    if not all([host, port, password]):
        print("Error: Missing env vars")
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. Check if process is running
        print("=== PROCESS CHECK ===")
        cmd = f"ps aux | grep '{run_id_fragment}' | grep python | grep -v grep | head -n 1"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        ps_out = stdout.read().decode().strip()
        if ps_out:
            parts = ps_out.split()
            pid = parts[1] if len(parts) > 1 else "?"
            cpu = parts[2] if len(parts) > 2 else "?"
            print(f"PID: {pid}, CPU: {cpu}%")
        else:
            print("NO PROCESS FOUND!")
            return
        
        # 2. Check GPU activity
        print("\n=== GPU CHECK ===")
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode().strip())
        
        # 3. Take TWO snapshots of global_step from WandB summary (if synced)
        # Or, check the training log for step advancement
        
        # 4. Check log file modification time NOW vs 60 seconds from now
        print("\n=== LOG FRESHNESS CHECK (Waiting 30s for delta) ===")
        
        # Find the active log file
        cmd = f"ls -lt /workspace/DeepScalper/*.log 2>/dev/null | head -n 1"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        log_info = stdout.read().decode().strip()
        print(f"Most Recent Log: {log_info}")
        
        # Get file size now
        cmd = "stat -c %s /workspace/DeepScalper/run.log 2>/dev/null"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        size1 = stdout.read().decode().strip()
        print(f"Log Size (T0): {size1} bytes")
        
        time.sleep(30)
        
        # Get file size again
        cmd = "stat -c %s /workspace/DeepScalper/run.log 2>/dev/null"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        size2 = stdout.read().decode().strip()
        print(f"Log Size (T+30s): {size2} bytes")
        
        try:
            delta = int(size2) - int(size1)
            print(f"Growth: {delta} bytes")
            if delta > 0:
                print("VERDICT: Log is actively growing - TRAINING IS RUNNING")
            else:
                print("WARNING: Log did not grow - may be stalled or finished")
        except:
            print("Could not compute delta")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    live_status_check("hlmj8s0t")
