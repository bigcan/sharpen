
import os
import paramiko
import sys
from dotenv import load_dotenv

load_dotenv()

def run_remote_cmd():
    if len(sys.argv) < 2:
        print("Usage: python scripts/remote_cmd.py <command>")
        return

    cmd = " ".join(sys.argv[1:])
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT") or 22)
    password = os.getenv("GPUHUB_PASSWORD")
    username = 'root'
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port=port, username=username, password=password)
        
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(f"--- Output of '{cmd}' ---")
        print(stdout.read().decode())
        err = stderr.read().decode()
        if err:
            print(f"--- Error ---")
            print(err)
            
        ssh.close()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    run_remote_cmd()
