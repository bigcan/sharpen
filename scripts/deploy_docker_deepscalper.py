import os
import argparse
import paramiko
import zipfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

PROJECT_ROOT = Path(os.getcwd())
DEPLOY_EXCLUDES = [
    'mlruns', 'logs', 'wandb', 'results', 'checkpoints', '.git', '.venv', 'venv', '__pycache__', 
    'market_data.parquet', 'btc_lob_jan2023.parquet'
]

def create_filtered_zip(source_dir, output_filename):
    print(f"Creating source package: {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in DEPLOY_EXCLUDES]
            
            # Exclude data folder content from zip (we rely on host volume mount)
            rel_root = os.path.relpath(root, source_dir)
            if rel_root == "." and 'data' in dirs:
                dirs.remove('data')

            for file in files:
                if file.endswith((".pyc", ".pyo", ".zip", ".ds_store", ".webp", ".jpg", ".png")): continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Package created. Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

def deploy_docker(args):
    gpuhub_host = os.getenv("GPUHUB_HOST")
    gpuhub_port = int(os.getenv("GPUHUB_PORT", 22))
    gpuhub_user = "root"
    gpuhub_password = os.getenv("GPUHUB_PASSWORD")
    wandb_key = os.getenv("WANDB_API_KEY")

    if not gpuhub_host or not gpuhub_password:
        print("Error: GPUHUB_HOST and GPUHUB_PASSWORD must be set in .env")
        return

    remote_workspace = "/workspace/DeepScalper_Docker"
    zip_name = "docker_deploy.zip"
    image_name = f"bigcan/deepscalper:{args.tag}"
    container_name = "deepscalper_v10"

    # 1. Create Zip
    create_filtered_zip(PROJECT_ROOT, zip_name)

    print(f"Connecting to {gpuhub_host}:{gpuhub_port}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(gpuhub_host, port=gpuhub_port, username=gpuhub_user, password=gpuhub_password)
    sftp = ssh.open_sftp()

    # 2. Upload Code
    print(f"Ensuring remote workspace {remote_workspace} exists...")
    ssh.exec_command(f"mkdir -p {remote_workspace}")
    
    print(f"Uploading {zip_name}...")
    sftp.put(zip_name, f"{remote_workspace}/{zip_name}")
    sftp.close()

    # 3. Build & Run
    print("Executing Remote Build & Launch...")
    
    # Combined command for robustness
    # 1. Unzip
    # 2. Build Docker Image (Locally on Remote)
    # 3. Run Container
    
    remote_cmds = [
        f"cd {remote_workspace}",
        f"unzip -o {zip_name} > /dev/null",
        f"rm {zip_name}",
        f"echo 'Building Docker Image {image_name} on Remote...'",
        f"docker build -f Dockerfile.deepscalper -t {image_name} .",
        f"docker stop {container_name} || true",
        f"docker rm {container_name} || true",
        f"""docker run -d \
            --gpus all \
            --name {container_name} \
            --shm-size=32g \
            --restart unless-stopped \
            -v /data:/app/data \
            -v {remote_workspace}/results:/app/results \
            -v {remote_workspace}/wandb:/app/wandb \
            -e WANDB_API_KEY={wandb_key} \
            {image_name} \
            python scripts/train_deepscalper.py --config configs/deepscalper_unified.yaml --run_name {args.run_name}"""
    ]

    full_cmd = " && ".join(remote_cmds)
    
    # Execute (Streaming output)
    stdin, stdout, stderr = ssh.exec_command(full_cmd)
    
    # Stream output to console
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line.strip())
        
    err = stderr.read().decode()
    if err:
        print(f"STDERR: {err}")

    ssh.close()
    if os.path.exists(zip_name):
        os.remove(zip_name)
    print("Deployment cycle completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v10", help="Docker image tag")
    parser.add_argument("--run_name", type=str, default="DeepScalper_Docker_V10_RemoteBuild", help="WandB Run Name")
    args = parser.parse_args()
    deploy_docker(args)
