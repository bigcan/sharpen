
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def check_space():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port=port, username='root', password=password)
        
        print("\n--- Disk Usage (df -h) ---")
        stdin, stdout, stderr = ssh.exec_command("df -h")
        print(stdout.read().decode())
        
        print("\n--- Workspace Usage (du -sh /workspace/Synapse_V9) ---")
        # Check specific workspace size
        stdin, stdout, stderr = ssh.exec_command("du -sh /workspace/Synapse_V9")
        print(stdout.read().decode())
        
        # Check for large core dumps or logs
        print("\n--- Large Files (>100MB) ---")
        stdin, stdout, stderr = ssh.exec_command("find /workspace -type f -size +100M -exec ls -lh {} \; | head -n 10")
        print(stdout.read().decode())
        
        ssh.close()
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    check_space()
