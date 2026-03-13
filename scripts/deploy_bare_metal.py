
import os
import sys
import json
import argparse
import paramiko
import zipfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Load Environment Variables from Root
load_dotenv()

# Add project root to path for local imports
sys.path.append(os.getcwd())

# Configuration
PROJECT_ROOT = Path(os.getcwd())
DEPLOY_EXCLUDES = [
    'mlruns', 'logs', 'wandb', 'results', 'checkpoints', '.git', '.venv', 'venv', '__pycache__',
    'market_data.parquet', 'btc_lob_jan2023.parquet', 'finrl_pro_ds.egg-info', # Exclude massive data & stale metadata
    'hpo.db', 'hpo.db-journal' # Exclude local HPO state to prevent overwriting remote clean start
]
ROOT_DATA_EXCLUDE = ['data'] # Only exclude root data folder

INSTANCES_FILE = PROJECT_ROOT / "instances.json"

def get_env_var(key, default=None):
    val = os.getenv(key, default)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val

def resolve_instance(instance_name=None):
    """Resolve SSH credentials from instances.json or .env fallback."""
    if INSTANCES_FILE.exists():
        with open(INSTANCES_FILE, encoding="utf-8") as f:
            registry = json.load(f)
        instances = registry.get("instances", {})

        if instance_name:
            if instance_name not in instances:
                available = ", ".join(instances.keys())
                raise ValueError(f"Instance '{instance_name}' not found. Available: {available}")
            inst = instances[instance_name]
        else:
            # Use default from registry
            default_name = registry.get("default", "")
            if default_name and default_name in instances:
                inst = instances[default_name]
                instance_name = default_name
            else:
                # Fall back to .env
                return {
                    "name": "env",
                    "host": get_env_var("GPUHUB_HOST"),
                    "port": int(get_env_var("GPUHUB_PORT")),
                    "password": get_env_var("GPUHUB_PASSWORD"),
                    "gpus": [],
                }

        return {
            "name": instance_name,
            "host": inst["host"],
            "port": inst["port"],
            "password": inst["password"],
            "gpus": inst.get("gpus", []),
        }
    else:
        # No instances.json — use .env
        return {
            "name": "env",
            "host": get_env_var("GPUHUB_HOST"),
            "port": int(get_env_var("GPUHUB_PORT")),
            "password": get_env_var("GPUHUB_PASSWORD"),
            "gpus": [],
        }

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
                if file in DEPLOY_EXCLUDES or file.startswith("hpo.db"): continue # Exclude specific files like hpo.db*
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Package created. Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

def deploy(args):
    inst = resolve_instance(args.instance)
    host = inst["host"]
    port = inst["port"]
    password = inst["password"]
    print(f"Target instance: {inst['name']} ({host}:{port})"
          + (f" | GPUs: {inst['gpus']}" if inst['gpus'] else ""))
    # CANONICAL NAMING: DeepScalper_V1_{Platform}_{YYYYMMDD}_{HHMM}
    # Suffixes/metadata go in tags, not run name
    from finrl_pro_ds.utils.naming import generate_run_name
    version = args.version if args.version else "V1"
    full_run_name = generate_run_name(version=version, platform="GPUHub")

    # If user provided a custom name, add it as a tag instead
    extra_tags = []
    if args.run_name:
        extra_tags.append(args.run_name)

    # Auto-inject platform + GPU model WandB tags
    extra_tags.append("gpuhub")
    # Derive GPU tag from instances.json (e.g., "RTX 4090" -> "rtx4090")
    if inst["gpus"]:
        gpu_idx = int(args.gpu) if args.gpu is not None and args.gpu.isdigit() else 0
        gpu_name = inst["gpus"][min(gpu_idx, len(inst["gpus"]) - 1)]
        extra_tags.append(gpu_name.lower().replace(" ", ""))

    wandb_key = os.getenv("WANDB_API_KEY", "")

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
        # default to demo if not specified
        data_filename = args.data_file if args.data_file else "btc_lob_demo.parquet"
        local_data = PROJECT_ROOT / "data" / data_filename
        remote_data = f"{remote_workspace}/data/{data_filename}" # Keep in data subdir on remote

        # Create remote data dir
        remote_dir = os.path.dirname(remote_data).replace("\\", "/") # Ensure forward slashes for Linux
        ssh.exec_command(f"mkdir -p {remote_dir}")

        if not local_data.exists():
            print(f"WARNING: Local data not found at {local_data}. CHECK PATHS.")
        else:
            # GOLD CACHE LOGIC
            should_upload = True
            local_size = os.path.getsize(local_data)
            try:
                remote_attr = sftp.stat(remote_data)
                remote_size = remote_attr.st_size
                if remote_size == local_size:
                     print(f"Gold Cache Hit: Remote file {data_filename} exists and matches size ({local_size/1024/1024:.2f} MB). Skipping upload.")
                     should_upload = False
                else:
                     print(f"Cache Miss: Size mismatch (Local: {local_size} vs Remote: {remote_size}). Re-uploading...")
            except IOError:
                print(f"Cache Miss: Remote file not found. Uploading...")

            if should_upload:
                print(f"Uploading Data: {local_data} -> {remote_data}...")
                # using generic put
                sftp.put(str(local_data), remote_data)
                print("Data upload complete.")

    print(f"Uploading {zip_name}...")
    sftp.put(zip_name, f"{remote_workspace}/{zip_name}")
    sftp.close()

    # 4. Clean up old processes FIRST (Avoid file locks on hpo.db)
    if not args.no_kill:
        print("Killing old instances (Orchestrator & Workers)...")
        targets = [
            script_path,                # The script we are about to launch
            "train_deepscalper.py",     # The specialized trainer
            "tune_deepscalper.py",      # The HPO tuner
            "backtest_deepscalper.py",  # The backtester
            "wandb-service"             # Optional: Cleanup wandb internal process if stuck
        ]
        unique_targets = list(set(targets))
        kill_cmd_parts = [f"pkill -f {t}" for t in unique_targets]
        full_kill_cmd = " || true; ".join(kill_cmd_parts) + " || true"

        try:
            ssh.exec_command(full_kill_cmd)
            time.sleep(3) # Allow cleanup
        except:
            pass
    else:
        print("SKIPPING kill step (Preserving existing processes)...")

    # 5. Extract & Setup
    print("Extracting and Setting up...")

    # Prepend Miniconda to PATH for all commands
    export_path = "export PATH=/root/miniconda3/bin:$PATH"

    setup_cmds = [
        f"cd {remote_workspace}",
        "echo 'STEP: START'",
        # CRITICAL: Increase file descriptor limit for high-concurrency AsyncVectorEnv
        "ulimit -n 65536",
        # CRITICAL: Resize /dev/shm from default 64MB to 2GB for shared memory IPC
        # Without this, AsyncVectorEnv with SHM + 20 workers triggers [Errno 104]
        "mount -o remount,size=2G /dev/shm || echo 'WARN: /dev/shm remount failed (non-fatal)'",
        # Fresh HPO Logic
        f"{'rm -f hpo.db* && echo STEP: WIPE HPO DB' if args.fresh_hpo else 'echo STEP: RETAIN HPO DB'}",
        "echo 'STEP: UNINSTALL'",
        # CRITICAL: Clean everything to avoid stale deps
        "/root/miniconda3/bin/pip uninstall finrl-pro-ds -y || true",
        "rm -rf finrl_pro_ds.egg-info build dist",
        "echo 'STEP: UNZIP'",
        f"unzip -o {zip_name}",
        "rm deploy_package.zip",
        # Verify setup.py content
        "grep -C 2 'install_requires' setup.py || echo 'setup.py missing'",
        "echo 'STEP: INSTALL REQS'",
        # Base Image is PyTorch 2.8.0 + CUDA 12.8
        "/root/miniconda3/bin/pip install -q --upgrade -r requirements.txt",
        "echo 'STEP: INSTALL PKG'",
        "/root/miniconda3/bin/pip install -q -e ."  # Editable install
    ]

    cmd_chain = " && ".join(setup_cmds) + " && echo SETUP_SUCCESS"
    stdin, stdout, stderr = ssh.exec_command(cmd_chain)
    out = stdout.read().decode()
    err = stderr.read().decode()

    print("Setup Output:")
    print(out.encode("utf-8", errors="replace").decode("utf-8"))
    if err:
        print("Setup Stderr:")
        print(err.encode("utf-8", errors="replace").decode("utf-8"))

    if "SETUP_SUCCESS" not in out:
        print("CRITICAL: Setup failed. Aborting launch.")
        sys.exit(1)
    else:
        print("Setup completed successfully.")

    # 5. Launch
    print(f"Launching {script_path}..." + (f" as {full_run_name}" if full_run_name else " (script will auto-generate name)"))

    # Construct command
    # Assuming script is in scripts/ folder usually
    # We run from workspace root
    # SET ULIMIT for high-concurrency shared memory (24 workers * 93 cols)
    # Use -u for unbuffered output to capture crashes
    # Prepend PATH here too
    # Build run_name arg only if provided
    run_name_arg = f"--run_name {full_run_name}" if full_run_name else ""
    # Merge extra_tags with any tags in extra_args
    all_extra_args = args.extra_args
    if extra_tags:
        # Append extra tags to the command
        tag_str = " ".join(extra_tags)
        if "--tags" in all_extra_args:
            # Append to existing tags
            all_extra_args = all_extra_args.replace("--tags", f"--tags {tag_str}")
        else:
            all_extra_args = f"{all_extra_args} --tags {tag_str}"

    # GPU selection for multi-GPU instances
    gpu_env = f"export CUDA_VISIBLE_DEVICES={args.gpu} &&" if args.gpu is not None else ""

    wandb_env = f"export WANDB_API_KEY={wandb_key} &&" if wandb_key else ""
    discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    discord_env = f"export DISCORD_WEBHOOK_URL='{discord_url}' &&" if discord_url else ""
    from datetime import datetime
    log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Unique HPO DB to prevent locking collisions
    hpo_db = f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    hpo_storage_arg = f"--hpo_storage sqlite:///{remote_workspace}/{hpo_db}"

    version_arg = f"--version {version}" if version else ""

    # FIX: Remove () around ulimit so it applies to the current shell and subsequent nohup process
    cmd = f"{export_path} && {gpu_env} {wandb_env} {discord_env} ulimit -n 65535 || true && nohup python -u {script_path} --config {config_path} {run_name_arg} {version_arg} {hpo_storage_arg} {all_extra_args} > {log_file} 2>&1 & echo $! > run.pid"

    exec_cmd = f"cd {remote_workspace} && {cmd}"
    stdin, stdout, stderr = ssh.exec_command(exec_cmd)

    # Check if launched
    time.sleep(15)
    stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/run.pid")
    pid = stdout.read().decode().strip()

    if pid and pid.isdigit():
        print(f"SUCCESS: Deployed successfully. PID: {pid}")
        print(f"Logs: {remote_workspace}/{log_file}")
        print(f"WandB Run: {full_run_name if full_run_name else '(auto-generated by script)'}")
    else:
        print("FAILURE: PID not found. Check remote logs.")
        stdin, stdout, stderr = ssh.exec_command(f"cat {remote_workspace}/{log_file}")
        print(stdout.read().decode())

    ssh.close()
    os.remove(zip_name)

    # ------------------------------------------------------------------
    # Gap #2 & #3: Resolve Run ID & Log to Registry
    # ------------------------------------------------------------------
    resolved_run_id = None
    if pid and pid.isdigit():
        print(f"\nResolving WandB Run ID (waiting for remote init)...")
        time.sleep(10) # Give WandB a moment to init

        # We need a new SSH connection since we closed the main one
        # (Or we could have kept it open, but let's keep logic isolated)
        try:
            # Re-connect for registry/monitoring tasks
            ssh_reg = paramiko.SSHClient()
            ssh_reg.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_reg.connect(host, port=int(port), username='root', password=password, timeout=30)

            # Check wandb/latest-run symlink target
            # Target format: run-20260218_120000-8charID.wandb
            for attempt in range(6): # Try for 60s
                stdin, stdout, stderr = ssh_reg.exec_command(f"readlink {remote_workspace}/wandb/latest-run")
                target = stdout.read().decode().strip()

                if target and "run-" in target and ".wandb" in target:
                    # Parse ID: run-DATETIME-ID.wandb
                    parts = target.split("-")
                    if len(parts) >= 3:
                        resolved_run_id = parts[-1].replace(".wandb", "")
                        print(f"Resolved Run ID: {resolved_run_id}")
                        ssh_reg.exec_command(f"echo {resolved_run_id} > {remote_workspace}/run_id.txt")
                        break
                time.sleep(10)

            if not resolved_run_id:
                print("WARNING: Could not resolve Run ID from remote (WandB init too slow?)")

            # Log to Registry (results/deploys.db)
            import sqlite3
            from datetime import datetime

            db_path = os.path.join(PROJECT_ROOT, "results", "deploys.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deploys (
                    run_id TEXT,
                    pid TEXT,
                    config_path TEXT,
                    run_name TEXT,
                    gpuhub_host TEXT,
                    deployed_at TEXT,
                    extra_args TEXT,
                    status TEXT
                )
            ''')

            cursor.execute('''
                INSERT INTO deploys (run_id, pid, config_path, run_name, gpuhub_host, deployed_at, extra_args, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                resolved_run_id,
                pid,
                args.config,
                full_run_name or "auto",
                host,
                datetime.now().isoformat(),
                args.extra_args,
                "deployed"
            ))
            conn.commit()
            conn.close()
            print(f"Logged deployment to {db_path}")

            ssh_reg.close()

        except Exception as e:
            print(f"WARNING: Registry logging failed: {e}")

    # ------------------------------------------------------------------
    # Post-deploy: Poll & Collect
    # ------------------------------------------------------------------
    # Post-deploy: poll and collect if --collect flag is set
    if getattr(args, 'collect', False) and pid and pid.isdigit():
        print(f"\n{'='*60}")
        print("  --collect mode: waiting for run to complete...")
        print(f"{'='*60}")
        print("  Poll interval: 5 min | Max wait: 8 hours")
        print("  Press Ctrl+C to cancel (run will continue on remote)\n")

        try:
            from scripts.fetch_wandb_run import poll_run_until_complete
            from scripts.collect_run import collect_run

            # Wait for run to appear in WandB and complete
            # Use resolved ID if we have it, otherwise poll latest
            target_run_id = resolved_run_id

            print(f"  Polling Run ID: {target_run_id if target_run_id else 'LATEST (auto-detect)'}")
            result = poll_run_until_complete(run_id=target_run_id, poll_interval=300, max_wait=28800)

            if result.get("run_id"):
                print(f"\nRun completed: {result['run_id']} (state: {result['state']})")
                collect_run(result["run_id"])
            else:
                print(f"\nWARNING: Polling ended without a completed run: {result}")
                print("  Use 'python scripts/collect_run.py --run_id <ID>' manually later.")
        except KeyboardInterrupt:
            print("\n\nCollection cancelled. Run continues on remote.")
            print("  Use 'python scripts/collect_run.py --run_id <ID>' manually later.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="scripts/run_full_pipeline.py", help="Script to run (default: scripts/run_full_pipeline.py)")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--run_name", default=None, help="WandB run name (target script generates canonical name if not provided)")
    parser.add_argument("--upload_data", action="store_true", help="Upload data file to /data")
    parser.add_argument("--data_file", default=None, help="Specific data filename in data/ to upload (e.g. btc_lob_jan2023.parquet)")
    parser.add_argument("--extra_args", default="", help="Extra arguments to pass to the script (e.g. '--trials 50 --steps 200000')")
    parser.add_argument("--fresh_hpo", action="store_true", help="Wipe existing HPO database for a fresh start")
    parser.add_argument("--no_kill", action="store_true", help="Do NOT kill existing processes (e.g. preserve Synapse run)")
    parser.add_argument("--version", default=None, help="Version tag (e.g. V1.1)")
    parser.add_argument("--collect", action="store_true", help="After deploy, poll WandB until completion then auto-collect all data (blocking)")
    parser.add_argument("--instance", default=None, help="Named instance from instances.json (e.g. gpuhub-1, gpuhub-2). Default: uses 'default' key or .env")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value (e.g. 0, 1, '0,1'). For multi-GPU instances.")
    args = parser.parse_args()

    deploy(args)
