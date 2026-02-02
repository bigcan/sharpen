
import os
import sys
import paramiko
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

def get_env_var(key):
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val

import argparse

def debug_remote(custom_cmd=None):
    host = get_env_var("GPUHUB_HOST")
    port = int(get_env_var("GPUHUB_PORT"))
    password = get_env_var("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    remote_workspace = "/workspace/DeepScalper"
    log_file = f"{remote_workspace}/run.log"
    
    if custom_cmd:
        cmd = custom_cmd
    else:
        cmd = f"grep -a -E 'Trial|Error|Exception|Sharpe' {log_file} | tail -n 20"
        
    print(f"Executing remote command: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    print("="*60)
    print(stdout.read().decode())
    print("="*60)
    
    err = stderr.read().decode()
    if err:
        print("STDERR:")
        print(err)
        
    ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, help="Custom command to execute")
    args = parser.parse_args()
    debug_remote(args.cmd)
