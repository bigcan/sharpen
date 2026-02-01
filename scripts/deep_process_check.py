import os
import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def deep_process_check():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. Find ALL python processes sorted by CPU
        print("=== TOP PYTHON PROCESSES (by CPU) ===")
        cmd = "ps aux | grep python | grep -v grep | sort -nr -k 3 | head -n 10"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 2. Find process tree for our run ID
        print("=== PROCESS TREE for hlmj8s0t ===")
        cmd = "pstree -p $(pgrep -f hlmj8s0t | head -n 1) 2>/dev/null || echo 'No tree found'"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 3. Check GPU processes
        print("=== GPU PROCESSES (nvidia-smi) ===")
        cmd = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 4. Check WandB sync process
        print("=== WANDB PROCESSES ===")
        cmd = "ps aux | grep wandb | grep -v grep"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 5. Check last 50 lines of run.log for activity patterns
        print("=== LAST 30 LINES of run.log ===")
        cmd = "tail -n 30 /workspace/DeepScalper/run.log"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    deep_process_check()
