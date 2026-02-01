import os
import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def compact_check():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username='root', password=password)
        
        # 1. GPU Compute Apps
        print("=== GPU PROCESSES ===")
        cmd = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 2. Top 3 Python by CPU
        print("=== TOP 3 PYTHON (CPU) ===")
        cmd = "ps aux --sort=-%cpu | grep python | grep -v grep | head -n 3 | awk '{print $2, $3, $11}'"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
        # 3. Check if there's a child under 395138
        print("=== CHILDREN OF SHELL 395138 ===")
        cmd = "pgrep -P 395138"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        children = stdout.read().decode().strip()
        print(f"Child PIDs: {children}")
        
        if children:
            for pid in children.split('\n'):
                cmd2 = f"ps -p {pid} -o pid,pcpu,comm"
                stdin, stdout, stderr = ssh.exec_command(cmd2)
                print(stdout.read().decode())
        
        # 4. Is 395139 still alive?
        print("=== PID 395139 STATUS ===")
        cmd = "ps -p 395139 -o pid,state,pcpu,pmem,time,comm 2>/dev/null || echo 'NOT FOUND'"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

if __name__ == "__main__":
    compact_check()
