
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def monitor():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("Checking active processes...")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep deepscalper | grep -v grep")
    print(stdout.read().decode())
    
    print("Tailing run.log...")
    stdin, stdout, stderr = ssh.exec_command("tail -n 50 /workspace/DeepScalper/run.log")
    print(stdout.read().decode())
    
    print("Checking checkpoints...")
    stdin, stdout, stderr = ssh.exec_command("ls -l /workspace/DeepScalper/checkpoints/")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    monitor()
