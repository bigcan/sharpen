import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

def diagnose_zombies():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    print("=== NVIDIA-SMI (GPU Processes) ===")
    stdin, stdout, stderr = ssh.exec_command("nvidia-smi")
    print(stdout.read().decode())
    
    print("\n=== Top Python Processes (CPU/RAM) ===")
    # Listing top 10 python processes by CPU usage
    stdin, stdout, stderr = ssh.exec_command("ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | grep python | head -n 10")
    print(stdout.read().decode())
    
    print("\n=== Check PID File vs Reality ===")
    stdin, stdout, stderr = ssh.exec_command("cat /workspace/DeepScalper/run.pid")
    pid_file = stdout.read().decode().strip()
    print(f"PID in run.pid: '{pid_file}'")
    
    if pid_file:
        stdin, stdout, stderr = ssh.exec_command(f"ps -p {pid_file}")
        status = stdout.read().decode().strip()
        print(f"Status of PID {pid_file}:\n{status if status else 'DEAD/GONE'}")

    ssh.close()

if __name__ == "__main__":
    diagnose_zombies()
