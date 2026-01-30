
import os
import paramiko
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def check_remote_status():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    remote_workspace = "/workspace/DeepScalper"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("Checking PID...")
    stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.pid")
    pid = stdout.read().decode().strip()
    
    if pid:
        print(f"Found PID: {pid}")
        stdin, stdout, stderr = ssh.exec_command(f"ps -p {pid}")
        print("PS output:")
        print(stdout.read().decode())
    else:
        print("No PID file found.")
        
    print("\nTail of run.log:")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 50 {remote_workspace}/run.log")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    check_remote_status()
