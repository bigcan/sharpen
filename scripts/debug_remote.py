import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("GPUHUB_HOST")
port = int(os.getenv("GPUHUB_PORT"))
password = os.getenv("GPUHUB_PASSWORD")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username='root', password=password)

stdin, stdout, stderr = ssh.exec_command("cat /workspace/Synapse_V9/run.log")
print("STDOUT:", stdout.read().decode())
print("STDERR:", stderr.read().decode())

ssh.close()
