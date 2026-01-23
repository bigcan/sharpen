
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def check_remote():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    # Check Data
    print("--- Checking Data ---")
    stdin, stdout, stderr = ssh.exec_command("ls -l /data/btc_lob_jan2023.parquet")
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    # Check Processes
    print("--- Checking Processes ---")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep python")
    print(stdout.read().decode())
    
    # Check Logs
    print("--- Checking Run Log ---")
    stdin, stdout, stderr = ssh.exec_command("cat /workspace/Synapse_V9/run.log")
    print(stdout.read().decode())
    print("--- End Log ---")
    
    ssh.close()

if __name__ == "__main__":
    check_remote()
