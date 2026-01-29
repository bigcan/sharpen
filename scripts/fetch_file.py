
import os
import paramiko
from dotenv import load_dotenv
import argparse

load_dotenv()

def fetch(remote_path, local_path):
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT"))
    password = os.getenv("GPUHUB_PASSWORD")
    
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    sftp = ssh.open_sftp()
    
    print(f"Downloading {remote_path} -> {local_path}...")
    try:
        sftp.get(remote_path, local_path)
        print("Download complete.")
        print(f"Local file size: {os.path.getsize(local_path)} bytes")
    except Exception as e:
        print(f"Error downloading: {e}")
    
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_path", help="Absolute path on remote")
    parser.add_argument("local_path", help="Local destination path")
    args = parser.parse_args()
    
    fetch(args.remote_path, args.local_path)
