import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def run_remote_backtest():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    # Use checkpoint_1000008.pth
    cmd = "cd /workspace/Synapse_V9 && python3 scripts/backtest_deepscalper.py --config finrl_pro_ds/configs/deepscalper_gpuhub.yaml --checkpoint checkpoints/checkpoint_1000008.pth > backtest.log 2>&1 &"
    
    print(f"Executing: {cmd}")
    ssh.exec_command(cmd)
    
    # Check if process started
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep backtest_deepscalper.py | grep -v grep")
    print("Process Status:")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    run_remote_backtest()
