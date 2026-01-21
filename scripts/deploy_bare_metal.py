
import os
import sys
import argparse
import paramiko
import zipfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Load Environment Variables from Root
load_dotenv()

# Configuration
PROJECT_ROOT = Path(os.getcwd())
DEPLOY_EXCLUDES = [
    'mlruns', 'logs', 'wandb', 'results', 'checkpoints', '.git', '.venv', 'venv', '__pycache__', 
    'market_data.parquet', 'btc_lob_jan2023.parquet' # Exclude massive data files
]
ROOT_DATA_EXCLUDE = ['data'] # Only exclude root data folder

def get_env_var(key, default=None):
    val = os.getenv(key, default)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val

def create_filtered_zip(source_dir, output_filename):
    print(f"Creating execution package: {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Proactively remove excluded directories from traversal
            dirs[:] = [d for d in dirs if d not in DEPLOY_EXCLUDES]
            
            # Special handling for root/data
            rel_root = os.path.relpath(root, source_dir)
            if rel_root == "." and 'data' in dirs:
                dirs.remove('data')
                
            for file in files:
                if file.endswith((".pyc", ".pyo", ".zip", ".ds_store")): continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Package created. Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

def deploy(args):
    host = get_env_var("GPUHUB_HOST")
    port = int(get_env_var("GPUHUB_PORT"))
    password = get_env_var("GPUHUB_PASSWORD")
    wandb_key = os.getenv("WANDB_API_KEY", "")
    
    run_name = args.run_name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_run_name = f"{run_name}_{timestamp}"
    
    script_path = args.script
    config_path = args.config
    
    remote_workspace = "/workspace/Synapse_V9"
    zip_name = "deploy_package.zip"
    
    # 1. Create Zip
    create_filtered_zip(PROJECT_ROOT, zip_name)
    
    # 2. Connect
    print(f"Connecting to {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=password)
    sftp = ssh.open_sftp()
    
    # 3. Clean & Upload
    print("Cleaning remote workspace...")
    ssh.exec_command(f"rm -rf {remote_workspace}/*.zip")
    
    if args.upload_data:
        local_data = "c:/data/btc_lob_jan2023.parquet"
        remote_data = "/data/btc_lob_jan2023.parquet"
        print(f"Uploading Data: {local_data} -> {remote_data} (This may take a while)...")
        # Ensure remote dir exists
        ssh.exec_command("mkdir -p /data")
        sftp.put(local_data, remote_data)
        print("Data upload complete.")

    print(f"Uploading {zip_name}...")
    sftp.put(zip_name, f"{remote_workspace}/{zip_name}")
    sftp.close()
    
    # 4. Extract & Setup
    print("Extracting and Setting up...")
    setup_cmds = [
        f"cd {remote_workspace}",
        f"unzip -o {zip_name} > /dev/null",
        "rm deploy_package.zip",
        "pip install -e .", # Ensure local package is installed
        f"wandb login {wandb_key}" if wandb_key else "echo 'No WandB Key provided, skipping login'",
        f"pkill -f {script_path} || true" # Kill previous instances of THIS script
    ]
    
    stdin, stdout, stderr = ssh.exec_command(" && ".join(setup_cmds))
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err and "error" in err.lower():
        print(f"Setup Warning/Error: {err}")
    
    # 5. Launch
    print(f"Launching {script_path} as {full_run_name}...")
    
    # Construct command
    # Assuming script is in scripts/ folder usually
    # We run from workspace root
    cmd = f"nohup python3 {script_path} --config {config_path} --run_name {full_run_name} > run.log 2>&1 & echo $! > run.pid"
    
    exec_cmd = f"cd {remote_workspace} && {cmd}"
    stdin, stdout, stderr = ssh.exec_command(exec_cmd)
    
    # Check if launched
    time.sleep(5)
    stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.pid")
    pid = stdout.read().decode().strip()
    
    if pid and pid.isdigit():
        print(f"SUCCESS: Deployed successfully. PID: {pid}")
        print(f"Logs: {remote_workspace}/run.log")
        print(f"WandB Run: {full_run_name}")
    else:
        print("FAILURE: PID not found. Check remote logs.")
        stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.log")
        print(stdout.read().decode())
    
    ssh.close()
    os.remove(zip_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Script to run (relative to root), e.g., scripts/train_deepscalper.py")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--run_name", default="deepscalper_pilot", help="Base name for WandB run")
    parser.add_argument("--upload_data", action="store_true", help="Upload data file to /data")
    args = parser.parse_args()
    
    deploy(args)
