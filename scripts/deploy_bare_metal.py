
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
    'market_data.parquet', 'btc_lob_jan2023.parquet', 'finrl_pro_ds.egg-info' # Exclude massive data & stale metadata
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
    
    remote_workspace = "/workspace/DeepScalper"
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
    print("Ensuring remote workspace exists...")
    ssh.exec_command(f"mkdir -p {remote_workspace}")
    
    print("Cleaning remote workspace of old zips...")
    ssh.exec_command(f"rm -rf {remote_workspace}/*.zip")
    
    if args.upload_data:
        # Use absolute paths for robust deployment
        local_data = PROJECT_ROOT / "data" / "btc_lob_demo.parquet"
        remote_data = f"{remote_workspace}/btc_lob_demo.parquet"
        
        if not local_data.exists():
            print(f"WARNING: Local data not found at {local_data}. CHECK PATHS.")
        else:
            print(f"Uploading Data: {local_data} -> {remote_data}...")
            sftp.put(str(local_data), remote_data)
            print("Data upload complete.")

    print(f"Uploading {zip_name}...")
    sftp.put(zip_name, f"{remote_workspace}/{zip_name}")
    sftp.close()
    
    # 4. Extract & Setup
    print("Extracting and Setting up...")
    
    # Prepend Miniconda to PATH for all commands
    export_path = "export PATH=/root/miniconda3/bin:$PATH"
    
    setup_cmds = [
        f"{export_path}",
        f"cd {remote_workspace}",
        # CRITICAL: Clean everything to avoid stale deps
        "pip uninstall finrl-pro-ds -y || true",
        "rm -rf finrl_pro_ds.egg-info build dist",
        f"unzip -o {zip_name} > /dev/null",
        "rm deploy_package.zip",
        # Verify setup.py content
        "grep -C 2 'install_requires' setup.py || echo 'setup.py missing'",
        # Base Image is PyTorch 2.8.0 + CUDA 12.8 (Correct for RTX 5090)
        # DO NOT uninstall torch - use the pre-installed version
        "pip install --upgrade -r requirements.txt",
        "pip install -e .",  # Editable install after deps are correct
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
    # SET ULIMIT for high-concurrency shared memory (24 workers * 93 cols)
    # Use -u for unbuffered output to capture crashes
    # Prepend PATH here too
    cmd = f"{export_path} && ulimit -n 65535 && nohup python -u {script_path} --config {config_path} --run_name {full_run_name} > run.log 2>&1 & echo $! > run.pid"
    
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
    parser.add_argument("--run_name", default="DS_GPUHub_BM_V1", help="Base name for WandB run")
    parser.add_argument("--upload_data", action="store_true", help="Upload data file to /data")
    args = parser.parse_args()
    
    deploy(args)
