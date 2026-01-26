
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def list_processes():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("--- Python Processes ---")
    cmd = "ps -eo pid,lstart,cmd | grep train_deepscalper | grep -v grep"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    output = stdout.read().decode()
    
    if not output:
        print("No python processes found (or command failed).")
    else:
        print(output)
        
    ssh.close()

if __name__ == "__main__":
    list_processes()
