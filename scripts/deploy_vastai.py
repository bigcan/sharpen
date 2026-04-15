"""
Vast.ai Deployment & Management Script for DeepScalper.

Mirrors deploy_bare_metal.py (GPUHub) but adapted for Vast.ai's CLI/API.

Subcommands:
    search   - Search for suitable GPU instances
    create   - Create an instance from an offer
    deploy   - Deploy code and launch training on a running instance
    list     - List active instances
    destroy  - Destroy an instance
    logs     - Tail remote training logs
    status   - Check training status on an instance
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from project root .env
load_dotenv()

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REMOTE_WORKSPACE = "/workspace/DeepScalper"

DEPLOY_EXCLUDES = [
    "mlruns", "logs", "wandb", "results", "checkpoints", ".git", ".venv",
    "venv", "__pycache__", "market_data.parquet", "btc_lob_jan2023.parquet",
    "finrl_pro_ds.egg-info", "hpo.db", "hpo.db-journal",
]

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

DEFAULT_SEARCH_FILTER = (
    'gpu_name in ["RTX_3090","RTX_4090"] '
    "cpu_cores>=8 cpu_ram>=32 disk_space>=50 "
    "cuda_vers>=12.0 reliability>0.95 num_gpus=1"
)

DEPLOY_DB_PATH = PROJECT_ROOT / "results" / "deploys.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vastai_exe() -> str:
    """Return the path to the vastai CLI executable."""
    # On Windows inside the project venv
    local_exe = PROJECT_ROOT / ".venv" / "Scripts" / "vastai.exe"
    if local_exe.exists():
        return str(local_exe)
    # Fallback: assume it is on PATH
    return "vastai"


def _run_vastai(args: list[str], *, capture=True, check=True) -> subprocess.CompletedProcess:
    """Run a vastai CLI command and return the result."""
    cmd = [_vastai_exe()] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
            timeout=120,
        )
        return result
    except FileNotFoundError:
        print("ERROR: vastai CLI not found. Install with: pip install vastai")
        print("Then run: vastai set api-key <YOUR_KEY>")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: vastai command failed (exit {exc.returncode})")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        raise


def _parse_ssh_url(instance_id: str) -> dict:
    """
    Call ``vastai ssh-url <id>`` and parse the result into
    ``{'host': ..., 'port': ...}``.

    Vast.ai ssh-url returns something like ``ssh://root@<host>:<port>``
    """
    result = _run_vastai(["ssh-url", str(instance_id)])
    raw = result.stdout.strip()
    # Formats observed: ssh://root@host:port  OR  root@host -p port
    m = re.match(r"ssh://(?:\w+@)?([^:]+):(\d+)", raw)
    if m:
        return {"host": m.group(1), "port": int(m.group(2))}
    # Alternate format: host -p port
    m = re.match(r"(?:\w+@)?(\S+)\s+-p\s+(\d+)", raw)
    if m:
        return {"host": m.group(1), "port": int(m.group(2))}
    raise RuntimeError(f"Could not parse SSH URL from vastai: '{raw}'")


def _ssh_connect(instance_id: str, *, timeout: int = 30):
    """
    Open an SSH connection to a Vast.ai instance using key-based auth.
    Returns (paramiko.SSHClient, ssh_info_dict).
    """
    import paramiko

    ssh_info = _parse_ssh_url(instance_id)
    key_path = _find_ssh_key()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    ssh.connect(
        ssh_info["host"],
        port=ssh_info["port"],
        username="root",
        key_filename=str(key_path),
        timeout=timeout,
    )
    return ssh, ssh_info


def _find_ssh_key() -> Path:
    """Locate a usable SSH private key, or prompt to generate one."""
    ssh_dir = Path.home() / ".ssh"
    candidates = ["id_ed25519", "id_rsa"]
    for name in candidates:
        p = ssh_dir / name
        if p.exists():
            return p

    print("No SSH key found (~/.ssh/id_ed25519 or ~/.ssh/id_rsa).")
    print("Generating a new ed25519 key pair...")
    ssh_dir.mkdir(exist_ok=True)
    key_path = ssh_dir / "id_ed25519"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", ""],
        check=True,
    )
    print(f"Key generated at {key_path}")

    # Register with Vast.ai
    pub_path = ssh_dir / "id_ed25519.pub"
    if pub_path.exists():
        print("Registering public key with Vast.ai...")
        pub_content = pub_path.read_text().strip()
        _run_vastai(["create", "ssh-key", pub_content], check=False)
        print("SSH key registered.")
    return key_path


def _exec_ssh(ssh, cmd: str, *, timeout: int = 300, print_output=True) -> tuple[str, str]:
    """Execute a command over SSH and optionally print output."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if print_output:
        if out.strip():
            print(out)
        if err.strip():
            print("[stderr]", err)
    return out, err


def create_filtered_zip(source_dir: Path, output_filename: str) -> str:
    """Create a filtered zip of the project for deployment."""
    print(f"Creating execution package: {output_filename}...")
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in DEPLOY_EXCLUDES]
            rel_root = os.path.relpath(root, source_dir)
            if rel_root == "." and "data" in dirs:
                dirs.remove("data")
            for file in files:
                if file.endswith((".pyc", ".pyo", ".zip", ".ds_store")):
                    continue
                if file in DEPLOY_EXCLUDES or file.startswith("hpo.db"):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    size_mb = os.path.getsize(output_filename) / 1024 / 1024
    print(f"Package created. Size: {size_mb:.2f} MB")
    return output_filename


def _ensure_deploy_db() -> sqlite3.Connection:
    """Ensure the deploys.db exists with the expected schema (including platform column)."""
    DEPLOY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DEPLOY_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deploys (
            run_id TEXT,
            pid TEXT,
            config_path TEXT,
            run_name TEXT,
            gpuhub_host TEXT,
            deployed_at TEXT,
            extra_args TEXT,
            status TEXT,
            platform TEXT DEFAULT 'gpuhub'
        )
    """)
    # Add platform column if missing (existing DB from deploy_bare_metal)
    try:
        cursor.execute("ALTER TABLE deploys ADD COLUMN platform TEXT DEFAULT 'gpuhub'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_search(args):
    """Search for available Vast.ai instances."""
    # Build filter string
    if args.gpu:
        gpu_clause = f'gpu_name="{args.gpu}"'
    else:
        gpu_clause = 'gpu_name in ["RTX_3090","RTX_4090"]'

    filter_parts = [
        gpu_clause,
        f"cpu_cores>={args.min_cpu}",
        f"cpu_ram>={args.min_ram}",
        "disk_space>=50",
        "cuda_vers>=12.0",
        "reliability>0.95",
        "num_gpus=1",
    ]
    filter_str = " ".join(filter_parts)

    print(f"Searching Vast.ai offers with filter: {filter_str}")
    print(f"Sort: {args.sort} | Limit: {args.limit}\n")

    result = _run_vastai([
        "search", "offers",
        "--raw",
        "--order", args.sort,
        "--limit", str(args.limit),
        filter_str,
    ])

    try:
        offers = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fallback: print raw output
        print(result.stdout)
        return

    if not offers:
        print("No instances found matching the criteria.")
        return

    # Print table
    header = f"{'ID':>10}  {'GPU':>12}  {'VRAM':>6}  {'CPU':>4}  {'RAM':>6}  {'Disk':>5}  {'CUDA':>5}  {'$/hr':>6}  {'Rel.':>5}  {'DL Mbps':>8}  {'Location':>20}"
    print(header)
    print("-" * len(header))
    for o in offers:
        geo = o.get("geolocation", "?")
        # Truncate long location strings
        if len(geo) > 20:
            geo = geo[:17] + "..."
        print(
            f"{o.get('id', '?'):>10}  "
            f"{o.get('gpu_name', '?'):>12}  "
            f"{o.get('gpu_ram', 0) / 1024:>5.1f}G  "
            f"{int(o.get('cpu_cores_effective', o.get('cpu_cores', 0))):>4}  "
            f"{o.get('cpu_ram', 0) / 1024:>5.1f}G  "
            f"{o.get('disk_space', 0):>5.0f}  "
            f"{o.get('cuda_max_good', 0):>5.1f}  "
            f"{o.get('dph_total', 0):>6.3f}  "
            f"{o.get('reliability2', o.get('reliability', 0)):>5.2f}  "
            f"{o.get('inet_down', 0):>8.1f}  "
            f"{geo:>20}",
        )

    print(f"\nTotal offers: {len(offers)}")
    print("Use: python scripts/deploy_vastai.py create --offer_id <ID>")


def cmd_create(args):
    """Create a Vast.ai instance from an offer ID."""
    offer_id = str(args.offer_id)
    image = args.image
    disk = str(args.disk)

    # Ensure SSH key is available and registered
    _find_ssh_key()

    print(f"Creating instance from offer {offer_id}...")
    print(f"  Image: {image}")
    print(f"  Disk:  {disk} GB")

    result = _run_vastai([
        "create", "instance", offer_id,
        "--image", image,
        "--disk", disk,
        "--direct",
    ])

    print(result.stdout.strip())

    # Parse the instance ID from the output (e.g., "Started. {'new_contract': 12345}")
    m = re.search(r"'new_contract':\s*(\d+)", result.stdout)
    if not m:
        m = re.search(r'"new_contract":\s*(\d+)', result.stdout)
    if not m:
        m = re.search(r"(\d{5,})", result.stdout)

    if m:
        instance_id = m.group(1)
        print(f"\nInstance ID: {instance_id}")
        print("Waiting for instance to start...")

        # Poll until running
        for attempt in range(30):
            time.sleep(10)
            try:
                show_result = _run_vastai(["show", "instance", instance_id, "--raw"], check=False)
                if show_result.returncode != 0:
                    continue
                info = json.loads(show_result.stdout)
                status = info.get("actual_status", info.get("status_msg", "unknown"))
                print(f"  [{attempt + 1}/30] Status: {status}")
                if status == "running":
                    print(f"\nInstance {instance_id} is RUNNING.")
                    try:
                        ssh_info = _parse_ssh_url(instance_id)
                        print(f"  SSH: ssh -p {ssh_info['port']} root@{ssh_info['host']}")
                    except Exception:
                        print("  (SSH URL not yet available - try again in a moment)")
                    print(f"\nNext: python scripts/deploy_vastai.py deploy --instance_id {instance_id} --config <config.yaml>")
                    return
            except (json.JSONDecodeError, Exception):
                continue

        print("\nWARNING: Instance did not reach 'running' state within 5 minutes.")
        print("Check manually: python scripts/deploy_vastai.py list")
    else:
        print("\nCould not parse instance ID from output. Check: python scripts/deploy_vastai.py list")


def _get_instance_info(instance_id: str) -> dict:
    """Fetch instance metadata (GPU, location, etc.) from Vast.ai API."""
    result = _run_vastai(["show", "instance", str(instance_id), "--raw"], check=False)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def _build_platform_tags(instance_info: dict) -> list[str]:
    """Build WandB tags for platform, GPU model, and region from instance metadata."""
    tags = ["vastai"]

    # GPU model tag (e.g., "rtx3090", "rtx4090")
    gpu_name = instance_info.get("gpu_name", "")
    if gpu_name:
        gpu_tag = gpu_name.lower().replace(" ", "").replace("_", "")
        tags.append(gpu_tag)

    # Region tag from geolocation (e.g., "Spain, ES" -> "es")
    geo = instance_info.get("geolocation", "")
    if geo:
        # Extract country code: "Spain, ES" -> "es", "California, US" -> "us"
        parts = [p.strip() for p in geo.split(",")]
        country_code = parts[-1].lower() if parts else ""
        if country_code:
            tags.append(country_code)

    return tags


def cmd_deploy(args):
    """Deploy code and launch training on a running Vast.ai instance."""

    instance_id = str(args.instance_id)

    # Fetch instance metadata for WandB tagging
    print(f"Fetching instance {instance_id} metadata...")
    instance_info = _get_instance_info(instance_id)
    platform_tags = _build_platform_tags(instance_info)
    gpu_name = instance_info.get("gpu_name", "unknown")
    geo = instance_info.get("geolocation", "unknown")
    print(f"  GPU: {gpu_name} | Location: {geo}")
    print(f"  Auto-tags: {platform_tags}")

    # Generate run name from config filename
    from finrl_pro_ds.utils.naming import generate_run_name
    full_run_name = generate_run_name(args.config)

    wandb_key = os.getenv("WANDB_API_KEY", "")
    config_path = args.config
    script_path = "scripts/run_full_pipeline.py"

    zip_name = "deploy_package.zip"

    # ----- 1. Create filtered zip -----
    create_filtered_zip(PROJECT_ROOT, zip_name)

    # ----- 2. Connect via SSH -----
    print(f"Connecting to Vast.ai instance {instance_id}...")
    ssh, ssh_info = _ssh_connect(instance_id)
    sftp = ssh.open_sftp()

    # ----- 3. Prepare remote workspace -----
    print("Ensuring remote workspace exists...")
    _exec_ssh(ssh, f"mkdir -p {REMOTE_WORKSPACE}", print_output=False)
    _exec_ssh(ssh, f"rm -rf {REMOTE_WORKSPACE}/*.zip", print_output=False)

    # ----- 4. Upload data if requested -----
    if args.upload_data:
        data_filename = args.data_file if args.data_file else "btc_lob_demo.parquet"
        local_data = PROJECT_ROOT / "data" / data_filename
        remote_data_dir = f"{REMOTE_WORKSPACE}/data/{os.path.dirname(data_filename)}".rstrip("/")
        remote_data = f"{REMOTE_WORKSPACE}/data/{data_filename}"

        _exec_ssh(ssh, f"mkdir -p {remote_data_dir}", print_output=False)

        if not local_data.exists():
            print(f"WARNING: Local data not found at {local_data}. CHECK PATHS.")
        else:
            should_upload = True
            local_size = os.path.getsize(local_data)
            try:
                remote_attr = sftp.stat(remote_data)
                if remote_attr.st_size == local_size:
                    print(
                        f"Cache Hit: Remote {data_filename} matches size "
                        f"({local_size / 1024 / 1024:.2f} MB). Skipping upload.",
                    )
                    should_upload = False
                else:
                    print(
                        f"Cache Miss: Size mismatch "
                        f"(local={local_size} vs remote={remote_attr.st_size}). Re-uploading...",
                    )
            except IOError:
                print("Cache Miss: Remote file not found. Uploading...")

            if should_upload:
                print(f"Uploading data: {local_data} -> {remote_data}...")
                sftp.put(str(local_data), remote_data)
                print("Data upload complete.")

    # ----- 5. Upload code zip -----
    print(f"Uploading {zip_name}...")
    sftp.put(zip_name, f"{REMOTE_WORKSPACE}/{zip_name}")
    sftp.close()

    # ----- 6. Kill old processes (unless --no_kill) -----
    if not args.no_kill:
        print("Killing old training processes...")
        targets = [
            "run_full_pipeline.py",
            "train_deepscalper.py",
            "tune_deepscalper.py",
            "backtest_deepscalper.py",
            "wandb-service",
        ]
        kill_cmd = " || true; ".join(f"pkill -f {t}" for t in targets) + " || true"
        _exec_ssh(ssh, kill_cmd, print_output=False)
        time.sleep(3)
    else:
        print("SKIPPING kill step (--no_kill).")

    # ----- 7. Extract & install -----
    print("Extracting and installing...")

    fresh_hpo_cmd = "rm -f hpo.db* && echo 'WIPE HPO DB'" if args.fresh_hpo else "echo 'RETAIN HPO DB'"

    setup_cmds = [
        f"cd {REMOTE_WORKSPACE}",
        "echo 'STEP: START'",
        "ulimit -n 65536",
        "mount -o remount,size=2G /dev/shm || echo 'WARN: /dev/shm remount failed (non-fatal)'",
        fresh_hpo_cmd,
        "pip uninstall finrl-pro-ds -y || true",
        "rm -rf finrl_pro_ds.egg-info build dist",
        f"unzip -o {zip_name}",
        f"rm -f {zip_name}",
        "pip install -q --upgrade -r requirements.txt",
        "pip install -q -e .",
    ]

    setup_chain = " && ".join(setup_cmds) + " && echo SETUP_SUCCESS"
    out, err = _exec_ssh(ssh, setup_chain, print_output=True, timeout=600)

    if "SETUP_SUCCESS" not in out:
        print("CRITICAL: Setup failed. Aborting launch.")
        ssh.close()
        os.remove(zip_name)
        sys.exit(1)
    print("Setup completed successfully.")

    # ----- 8. Launch training -----
    log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    hpo_db = f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    hpo_storage_arg = f"--hpo_storage sqlite:///{REMOTE_WORKSPACE}/{hpo_db}"
    run_name_arg = f"--run_name {full_run_name}"

    # Inject platform/location WandB tags into extra_args
    extra = args.extra_args or ""
    if platform_tags:
        tag_str = " ".join(platform_tags)
        if "--tags" in extra:
            extra = extra.replace("--tags", f"--tags {tag_str}")
        else:
            extra = f"{extra} --tags {tag_str}".strip()

    wandb_env = f"export WANDB_API_KEY={wandb_key} &&" if wandb_key else ""

    launch_cmd = (
        f"cd {REMOTE_WORKSPACE} && "
        f"{wandb_env} "
        f"ulimit -n 65535 || true && "
        f"nohup python -u {script_path} --config {config_path} "
        f"{run_name_arg} {hpo_storage_arg} {extra} "
        f"> {log_file} 2>&1 & echo $! > run.pid"
    )

    print(f"Launching training as {full_run_name}...")
    _exec_ssh(ssh, launch_cmd, print_output=False)

    # Verify PID
    time.sleep(15)
    pid_out, _ = _exec_ssh(ssh, f"cat {REMOTE_WORKSPACE}/run.pid", print_output=False)
    pid = pid_out.strip()

    if pid and pid.isdigit():
        print(f"SUCCESS: Deployed. PID: {pid}")
        print(f"Logs: {REMOTE_WORKSPACE}/{log_file}")
        print(f"WandB Run: {full_run_name}")
    else:
        print("FAILURE: PID not found. Remote logs:")
        _exec_ssh(ssh, f"cat {REMOTE_WORKSPACE}/{log_file}")
        ssh.close()
        os.remove(zip_name)
        sys.exit(1)

    ssh.close()
    if os.path.exists(zip_name):
        os.remove(zip_name)

    # ----- 9. Resolve WandB Run ID -----
    resolved_run_id = None
    if pid and pid.isdigit():
        print("\nResolving WandB Run ID (waiting for remote init)...")
        time.sleep(10)
        try:
            ssh2, _ = _ssh_connect(instance_id)
            for attempt in range(6):
                target_out, _ = _exec_ssh(
                    ssh2,
                    f"readlink {REMOTE_WORKSPACE}/wandb/latest-run",
                    print_output=False,
                )
                target = target_out.strip()
                if target and "run-" in target and ".wandb" in target:
                    parts = target.split("-")
                    if len(parts) >= 3:
                        resolved_run_id = parts[-1].replace(".wandb", "")
                        print(f"Resolved Run ID: {resolved_run_id}")
                        _exec_ssh(
                            ssh2,
                            f"echo {resolved_run_id} > {REMOTE_WORKSPACE}/run_id.txt",
                            print_output=False,
                        )
                        break
                time.sleep(10)

            if not resolved_run_id:
                print("WARNING: Could not resolve Run ID (WandB init too slow?).")

            # Log to deploy registry
            conn = _ensure_deploy_db()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO deploys
                    (run_id, pid, config_path, run_name, gpuhub_host, deployed_at, extra_args, status, platform)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_run_id,
                    pid,
                    args.config,
                    full_run_name,
                    f"vastai:{instance_id}",
                    datetime.now().isoformat(),
                    args.extra_args or "",
                    "deployed",
                    "vastai",
                ),
            )
            conn.commit()
            conn.close()
            print(f"Logged deployment to {DEPLOY_DB_PATH}")

            ssh2.close()
        except Exception as exc:
            print(f"WARNING: Registry logging failed: {exc}")

    # ----- 10. Poll & Collect (blocking) -----
    if getattr(args, "collect", False) and pid and pid.isdigit():
        print(f"\n{'=' * 60}")
        print("  --collect mode: waiting for run to complete...")
        print(f"{'=' * 60}")
        print("  Poll interval: 5 min | Max wait: 8 hours")
        print("  Press Ctrl+C to cancel (run will continue on remote)\n")

        try:
            from scripts.collect_run import collect_run
            from scripts.fetch_wandb_run import poll_run_until_complete

            target_run_id = resolved_run_id
            print(f"  Polling Run ID: {target_run_id if target_run_id else 'LATEST (auto-detect)'}")
            result = poll_run_until_complete(
                run_id=target_run_id, poll_interval=300, max_wait=28800,
            )
            if result.get("run_id"):
                print(f"\nRun completed: {result['run_id']} (state: {result['state']})")
                collect_run(result["run_id"])
            else:
                print(f"\nWARNING: Polling ended without a completed run: {result}")
                print("  Use 'python scripts/collect_run.py --run_id <ID>' manually later.")
        except KeyboardInterrupt:
            print("\n\nCollection cancelled. Run continues on remote.")
            print("  Use 'python scripts/collect_run.py --run_id <ID>' manually later.")


def cmd_list(args):
    """List active Vast.ai instances."""
    print("Fetching active instances...\n")

    result = _run_vastai(["show", "instances", "--raw"])

    try:
        instances = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout)
        return

    if not instances:
        print("No active instances.")
        return

    header = f"{'ID':>10}  {'GPU':>12}  {'Status':>12}  {'$/hr':>7}  {'Uptime':>10}  {'SSH'}"
    print(header)
    print("-" * len(header))

    for inst in instances:
        inst_id = inst.get("id", "?")
        gpu = inst.get("gpu_name", "?")
        status = inst.get("actual_status", inst.get("status_msg", "?"))
        dph = inst.get("dph_total", 0)

        # Compute uptime from start_date if available
        start = inst.get("start_date")
        if start:
            try:
                elapsed = time.time() - start
                hours = int(elapsed // 3600)
                mins = int((elapsed % 3600) // 60)
                uptime = f"{hours}h{mins:02d}m"
            except Exception:
                uptime = "?"
        else:
            uptime = "-"

        # SSH info
        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", "")
        ssh_str = f"{ssh_host}:{ssh_port}" if ssh_host else "-"

        print(
            f"{inst_id:>10}  {gpu:>12}  {status:>12}  "
            f"${dph:>6.3f}  {uptime:>10}  {ssh_str}",
        )

    print(f"\nTotal instances: {len(instances)}")


def cmd_destroy(args):
    """Destroy a Vast.ai instance."""
    instance_id = str(args.instance_id)

    # Confirmation prompt
    confirm = input(f"Destroy instance {instance_id}? This cannot be undone. [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    print(f"Destroying instance {instance_id}...")
    result = _run_vastai(["destroy", "instance", instance_id])
    print(result.stdout.strip())

    # Update deploy registry
    try:
        conn = _ensure_deploy_db()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE deploys SET status = 'destroyed' WHERE gpuhub_host = ? AND status = 'deployed'",
            (f"vastai:{instance_id}",),
        )
        if cursor.rowcount > 0:
            print(f"Updated {cursor.rowcount} deploy record(s) to 'destroyed'.")
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"WARNING: Could not update deploy registry: {exc}")

    print("Done.")


def cmd_logs(args):
    """Tail remote training logs on a Vast.ai instance."""
    instance_id = str(args.instance_id)
    lines = args.lines

    print(f"Connecting to instance {instance_id} to tail logs...\n")

    try:
        ssh, _ = _ssh_connect(instance_id)
    except Exception as exc:
        print(f"ERROR: Could not connect: {exc}")
        sys.exit(1)

    # Find the most recent log file
    out, _ = _exec_ssh(
        ssh,
        f"ls -1t {REMOTE_WORKSPACE}/run_*.log 2>/dev/null | head -1",
        print_output=False,
    )
    log_file = out.strip()

    if not log_file:
        print("No log files found on remote.")
        ssh.close()
        return

    print(f"Log file: {log_file}")
    print("-" * 60)

    _exec_ssh(ssh, f"tail -n {lines} {log_file}")
    ssh.close()


def cmd_status(args):
    """Check training status on a Vast.ai instance."""
    instance_id = str(args.instance_id)

    print(f"Connecting to instance {instance_id}...\n")

    try:
        ssh, _ = _ssh_connect(instance_id)
    except Exception as exc:
        print(f"ERROR: Could not connect: {exc}")
        sys.exit(1)

    # Running processes
    print("=== Training Processes ===")
    _exec_ssh(ssh, "ps aux | grep -E 'python.*run_full_pipeline|python.*train_deep|python.*tune_deep' | grep -v grep")

    # PID file
    print("\n=== PID File ===")
    _exec_ssh(ssh, f"cat {REMOTE_WORKSPACE}/run.pid 2>/dev/null || echo 'No PID file'")

    # WandB run ID
    print("\n=== WandB Run ID ===")
    _exec_ssh(ssh, f"cat {REMOTE_WORKSPACE}/run_id.txt 2>/dev/null || echo 'No run ID file'")

    # GPU utilization
    print("\n=== GPU Utilization ===")
    _exec_ssh(ssh, "nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available'")

    # Disk usage
    print("\n=== Disk Usage ===")
    _exec_ssh(ssh, f"df -h {REMOTE_WORKSPACE} 2>/dev/null; echo '---'; du -sh {REMOTE_WORKSPACE} 2>/dev/null")

    # Last few lines of log
    print("\n=== Recent Log (last 10 lines) ===")
    out, _ = _exec_ssh(
        ssh,
        f"ls -1t {REMOTE_WORKSPACE}/run_*.log 2>/dev/null | head -1",
        print_output=False,
    )
    log_file = out.strip()
    if log_file:
        _exec_ssh(ssh, f"tail -n 10 {log_file}")
    else:
        print("No log files found.")

    ssh.close()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vast.ai Deployment & Management for DeepScalper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search
  %(prog)s search --gpu RTX_4090 --sort dph
  %(prog)s create --offer_id 32194783 --disk 80
  %(prog)s deploy --instance_id 12345 --config configs/phase_k3_iqn_gc_5min.yaml --upload_data --data_file processed/gc_2025_3min_front.parquet
  %(prog)s status --instance_id 12345
  %(prog)s logs --instance_id 12345 --lines 100
  %(prog)s list
  %(prog)s destroy --instance_id 12345
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # ---- search ----
    sp_search = subparsers.add_parser("search", help="Search for suitable GPU instances")
    sp_search.add_argument("--gpu", default=None, help="GPU model (e.g., RTX_4090). Default: RTX_3090 or RTX_4090")
    sp_search.add_argument("--min_cpu", type=int, default=8, help="Minimum CPU cores (default: 8)")
    sp_search.add_argument("--min_ram", type=int, default=32, help="Minimum RAM in GB (default: 32)")
    sp_search.add_argument("--sort", default="dph_total", help="Sort field (default: dph_total)")
    sp_search.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    sp_search.set_defaults(func=cmd_search)

    # ---- create ----
    sp_create = subparsers.add_parser("create", help="Create an instance from an offer")
    sp_create.add_argument("--offer_id", required=True, type=int, help="Offer ID from search results")
    sp_create.add_argument("--image", default=DEFAULT_IMAGE, help=f"Docker image (default: {DEFAULT_IMAGE})")
    sp_create.add_argument("--disk", type=int, default=50, help="Disk space in GB (default: 50)")
    sp_create.set_defaults(func=cmd_create)

    # ---- deploy ----
    sp_deploy = subparsers.add_parser("deploy", help="Deploy code and launch training")
    sp_deploy.add_argument("--instance_id", required=True, type=int, help="Running instance ID")
    sp_deploy.add_argument("--config", required=True, help="Config YAML path (relative to project root)")
    sp_deploy.add_argument("--upload_data", action="store_true", help="Upload data file")
    sp_deploy.add_argument("--data_file", default=None, help="Data filename relative to data/ dir")
    sp_deploy.add_argument("--no_kill", action="store_true", help="Do not kill existing processes")
    sp_deploy.add_argument("--fresh_hpo", action="store_true", help="Wipe HPO database for fresh start")
    sp_deploy.add_argument("--collect", action="store_true", help="Poll WandB until completion then auto-collect (blocking)")
    sp_deploy.add_argument("--extra_args", default="", help="Extra args to pass to the training script")
    sp_deploy.set_defaults(func=cmd_deploy)

    # ---- list ----
    sp_list = subparsers.add_parser("list", help="List active instances")
    sp_list.set_defaults(func=cmd_list)

    # ---- destroy ----
    sp_destroy = subparsers.add_parser("destroy", help="Destroy an instance")
    sp_destroy.add_argument("--instance_id", required=True, type=int, help="Instance ID to destroy")
    sp_destroy.set_defaults(func=cmd_destroy)

    # ---- logs ----
    sp_logs = subparsers.add_parser("logs", help="Tail remote training logs")
    sp_logs.add_argument("--instance_id", required=True, type=int, help="Instance ID")
    sp_logs.add_argument("--lines", type=int, default=50, help="Number of lines to tail (default: 50)")
    sp_logs.set_defaults(func=cmd_logs)

    # ---- status ----
    sp_status = subparsers.add_parser("status", help="Check training status on an instance")
    sp_status.add_argument("--instance_id", required=True, type=int, help="Instance ID")
    sp_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
