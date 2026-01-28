
import os
import paramiko
from dotenv import load_dotenv
import shutil

load_dotenv()

def fetch_db():
    host = os.getenv("GPUHUB_HOST")
    port = int(os.getenv("GPUHUB_PORT") or 22)
    password = os.getenv("GPUHUB_PASSWORD")
    username = 'root'
    
    remote_path = "/workspace/DeepScalper/hpo.db"
    local_path = "hpo.db"
    
    print(f"Connecting to {host}:{port}...")
    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    try:
        print(f"Fetching {remote_path} to {local_path}...")
        sftp.get(remote_path, local_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error fetching file: {e}")
    finally:
        sftp.close()
        transport.close()

if __name__ == "__main__":
    fetch_db()
