import paramiko
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def check_liveness():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    logfile = "/workspace/DeepScalper/run.log"
    
    print(f"=== Liveness Check for {logfile} ===")
    
    # 1. Get Modification Time
    cmd = f"stat -c '%y' {logfile}"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    mod_time = stdout.read().decode().strip()
    print(f"File Mod Time: {mod_time}")
    
    # 2. Sample file size change over 10 seconds
    cmd_size = f"stat -c '%s' {logfile}"
    
    stdin, stdout, stderr = ssh.exec_command(cmd_size)
    size_1 = int(stdout.read().decode().strip())
    
    print(f"Monitoring for 10 seconds...")
    time.sleep(10)
    
    stdin, stdout, stderr = ssh.exec_command(cmd_size)
    size_2 = int(stdout.read().decode().strip())
    
    diff = size_2 - size_1
    print(f"Size T0: {size_1} bytes")
    print(f"Size T1: {size_2} bytes")
    print(f"Growth: {diff} bytes")
    
    if diff > 0:
        print("\n[VERDICT]: ALIVE & WRITING. The process is actively logging.")
        print("Tail of active logs:")
        stdin, stdout, stderr = ssh.exec_command(f"tail -n 5 {logfile}")
        print(stdout.read().decode())
    else:
        print("\n[VERDICT]: STALE/SILENT. No logs written in last 10 seconds.")
        print("Checking if stuck in HPO loop or just silent...")
        # Check process state again
        stdin, stdout, stderr = ssh.exec_command(f"ps -p 466965 -o state,etime,cmd") 
        print(stdout.read().decode())

    ssh.close()

if __name__ == "__main__":
    check_liveness()
