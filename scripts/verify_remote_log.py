
import os
import paramiko
import time
from dotenv import load_dotenv

load_dotenv()

def verify_remote_log():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port} to read logs...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    stdin, stdout, stderr = ssh.exec_command("tail -n 50 /workspace/DeepScalper/run.log")
    log_content = stdout.read().decode()
    err_content = stderr.read().decode()
    
    print("\n--- Remote Log Tail ---")
    print(log_content)
    
    if err_content:
        print("\n--- Stderr ---")
        print(err_content)
        
    ssh.close()

if __name__ == "__main__":
    verify_remote_log()
