
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def fetch_logs():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    remote_workspace = "/workspace/Synapse_V9"
    cmd = f"tail -n 50 {remote_workspace}/run.log"
    
    print(f"Fetching logs from {cmd}...")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    print("--- REMOTE LOGS START ---")
    print(stdout.read().decode())
    print("--- REMOTE LOGS END ---")
    
    print("Checking Remote Processes...")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep python")
    print(stdout.read().decode())
    
    err = stderr.read().decode()
    if err:
        print("--- STDERR START ---")
        print(err)
        print("--- STDERR END ---")
        
    ssh.close()

if __name__ == "__main__":
    fetch_logs()
