"""Upload the locally-fixed ppo_trainer.py to remote via SFTP."""
import os
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

local_path = os.path.join(os.path.dirname(__file__), "..",
    "finrl_pro_ds", "training", "ppo_trainer.py")
local_path = os.path.abspath(local_path)

targets = [
    "/workspace/DeepScalper/finrl_pro_ds/training/ppo_trainer.py",
    "/workspace/DeepScalper_BDQ/finrl_pro_ds/training/ppo_trainer.py",
]

sftp = c.open_sftp()
for remote_path in targets:
    print(f"Uploading {local_path}")
    print(f"      -> {remote_path}")
    sftp.put(local_path, remote_path)
    # Verify
    _, stdout, _ = c.exec_command("sed -n '143p' " + remote_path)
    line = stdout.read().decode().strip()
    print(f"  Line 143: {line}\n")
sftp.close()

print("Done!")
c.close()
