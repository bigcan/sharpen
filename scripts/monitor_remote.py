import paramiko
import os
import time
import sys
from dotenv import load_dotenv

def get_env_var(name):
    val = os.getenv(name)
    if not val:
        load_dotenv()
        val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing env var: {name}")
    return val

def monitor():
    host = get_env_var("GPUHUB_HOST")
    port = int(get_env_var("GPUHUB_PORT"))
    password = get_env_var("GPUHUB_PASSWORD")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("Connected. Tailing run.log...")
    stdin, stdout, stderr = ssh.exec_command("tail -n 20 /workspace/DeepScalper/run.log")
    print(stdout.read().decode())
    ssh.close()

if __name__ == "__main__":
    monitor()
