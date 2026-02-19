"""Upload any local file to remote DeepScalper workspace via SFTP."""
import os, sys
import paramiko
from dotenv import load_dotenv

load_dotenv()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    os.getenv("GPUHUB_HOST"),
    int(os.getenv("GPUHUB_PORT")),
    username="root",
    password=os.getenv("GPUHUB_PASSWORD"),
)

local_file = sys.argv[1]
remote_file = f"/workspace/DeepScalper/{local_file}"
if len(sys.argv) > 2:
    remote_file = sys.argv[2]

local_abs = os.path.abspath(local_file)
print(f"Uploading {local_abs} -> {remote_file}")
sftp = c.open_sftp()
sftp.put(local_abs, remote_file)
sftp.close()
print("Done!")
c.close()
