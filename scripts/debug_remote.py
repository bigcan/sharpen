
import os
import sys
import paramiko
from dotenv import load_dotenv

load_dotenv()

def debug():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    remote_workspace = "/workspace/DeepScalper"
    
    print("--- run.pid ---")
    stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.pid")
    print(stdout.read().decode())
    
    print("--- run.log ---")
    stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.log")
    print(stdout.read().decode())
    
    print("--- Setup Errors ---")
    # Usually setup errors are lost unless redirected?
    # But let's check if there's any stray log
    
    ssh.close()

if __name__ == "__main__":
    debug()
