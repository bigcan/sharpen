
import os
import paramiko
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def check_remote_details():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    remote_workspace = "/workspace/DeepScalper"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("Files in workspace:")
    stdin, stdout, stderr = ssh.exec_command(f"ls -lh {remote_workspace}")
    print(stdout.read().decode())
    
    print("\nRunning Python processes:")
    stdin, stdout, stderr = ssh.exec_command("ps -aux | grep python")
    print(stdout.read().decode())
    
    print("\nContent of run.pid (if any):")
    stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.pid")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    check_remote_details()
