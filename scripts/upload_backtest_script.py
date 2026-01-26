import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def upload_fixed_script():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    
    sftp = ssh.open_sftp()
    local_path = r"c:\FinRL\FinRL-Pro_DS\scripts\backtest_deepscalper.py"
    remote_path = "/workspace/Synapse_V9/scripts/backtest_deepscalper.py"
    
    print(f"Uploading {local_path} to {remote_path}...")
    sftp.put(local_path, remote_path)
    sftp.close()
    ssh.close()
    print("Upload complete.")

if __name__ == "__main__":
    upload_fixed_script()
