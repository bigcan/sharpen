
import os
import paramiko
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_status():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    workspace = "/workspace/Synapse_V9"
    
    # Check Log
    print("--- Remote Log (run.log) ---")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 20 {workspace}/run.log")
    print(stdout.read().decode())
    
    # Check PID
    print("--- Process Status ---")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep [t]rain_deepscalper.py")
    ps_output = stdout.read().decode()
    if ps_output:
        print(ps_output)
    else:
        print("No training process found.")
        
    ssh.close()

if __name__ == "__main__":
    check_status()
