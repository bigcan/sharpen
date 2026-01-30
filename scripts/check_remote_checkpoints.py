
import os
import paramiko
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def check_remote_checkpoints():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    remote_workspace = "/workspace/DeepScalper"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("Checkpoints directory structure:")
    stdin, stdout, stderr = ssh.exec_command(f"find {remote_workspace}/checkpoints -maxdepth 2")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    check_remote_checkpoints()
