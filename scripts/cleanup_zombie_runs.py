
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def cleanup_zombies():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    workspace = "/workspace/Synapse_V9"
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    # 1. Get Current Good PID
    print("Reading current run.pid...")
    stdin, stdout, stderr = ssh.exec_command(f"cat {workspace}/run.pid")
    current_pid = stdout.read().decode().strip()
    
    if not current_pid or not current_pid.isdigit():
        print(f"WARNING: Could not read valid PID from {workspace}/run.pid. (Got: '{current_pid}')")
        print("Aborting cleanup to avoid killing active job by mistake.")
        return

    print(f"Current Active Deployment PID: {current_pid}")
    
    # 2. Find All Training Processes
    print("Scanning for training processes...")
    # Get PIDs of python scripts running deepscalper
    stdin, stdout, stderr = ssh.exec_command("pgrep -f train_deepscalper.py")
    pids = stdout.read().decode().strip().split('\n')
    pids = [p for p in pids if p] # Filter empty strings
    
    print(f"Found PIDs: {pids}")
    
    killed_count = 0
    for pid in pids:
        if pid == current_pid:
            print(f"Skipping active PID {pid}.")
            continue
            
        print(f"Killing Zombie PID {pid}...")
        ssh.exec_command(f"kill -9 {pid}")
        killed_count += 1
        
    if killed_count == 0:
        print("No zombie processes found.")
    else:
        print(f"Successfully killed {killed_count} zombie processes.")
        
    ssh.close()

if __name__ == "__main__":
    cleanup_zombies()
