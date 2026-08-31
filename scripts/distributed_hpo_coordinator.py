#!/usr/bin/env python
"""
Distributed HPO Coordinator — Orchestrates parallel HPO across multiple GPU instances.

Lifecycle:
    1. Create shared Optuna study (PostgreSQL-backed)
    2. Provision GPU instances (Vast.ai or GPUHub)
    3. Deploy code + data to each worker via SSH/SFTP
    4. Launch distributed_hpo_worker.py on each instance
    5. Monitor progress (Optuna poll + WandB health + SSH heartbeat)
    6. On completion: extract best params, route to config, save YAML
    7. Auto-teardown instances if --auto_teardown

Usage:
    python scripts/distributed_hpo_coordinator.py \\
        --config configs/gmgp1_sac_gc_15min.yaml \\
        --n_workers 10 --n_trials 50 \\
        --platform vastai --gpu_filter RTX_4090 \\
        --db_url postgresql://user:pass@host:5432/optuna \\
        --study_name gmgp1_hpo_v5 \\
        --auto_teardown
"""

import argparse
import atexit
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import wandb
import yaml

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DistributedHPO.Coordinator")

REMOTE_WORKSPACE = "/workspace/DeepScalper"

DEPLOY_EXCLUDES = [
    "mlruns", "logs", "wandb", "results", "checkpoints", ".git", ".venv",
    "venv", "__pycache__", "market_data.parquet", "btc_lob_jan2023.parquet",
    "sharpen.egg-info", "finrl_pro_ds.egg-info", "hpo.db", "hpo.db-journal",
]

WANDB_ENTITY = "bigcan-chiwin-technology"
WANDB_PROJECT = "FinRL-Pro-DS"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_distributed_config(path: str | None) -> dict:
    """Load optional distributed HPO overlay config.

    Reads the nested ``distributed_hpo:`` schema documented in
    ``configs/distributed_hpo.yaml`` and flattens it into the operational keys
    the coordinator consumes. Unknown top-level keys are also merged in
    (back-compat with flat overlays).
    """
    defaults = {
        "poll_interval_seconds": 300,
        "max_reprovision_attempts": 2,
        "ssh_timeout": 30,
        "setup_timeout": 600,
        "worker_launch_delay": 5,
        "cost_per_hour": {},      # GPU model -> $/hr overrides
        "baseline_trial": None,   # dict of params to enqueue
        "data_files": [],         # data file paths (relative to data/) to upload
        "cost_cap_usd": None,     # hard spend cap; None = no cap
        "auto_teardown": False,   # overlay default for --auto_teardown CLI flag
        "vastai_gpu_filter": None,
        "vastai_max_price_per_hour": None,
        "vastai_disk_gb": 50,
        "vastai_docker_image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        "gpuhub_instances": None,  # explicit list "name:gpu_idx", overrides instances.json
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_group_prefix": "dhpo",
        "trial_timeout_seconds": None,  # passed through to worker for visibility
    }
    if not (path and os.path.exists(path)):
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}

    nested = overlay.get("distributed_hpo")
    if isinstance(nested, dict):
        ft = nested.get("fault_tolerance", {}) or {}
        cost = nested.get("cost", {}) or {}
        data = nested.get("data", {}) or {}
        worker = nested.get("worker", {}) or {}
        vastai = nested.get("vastai", {}) or {}
        gpuhub = nested.get("gpuhub", {}) or {}
        wb = nested.get("wandb", {}) or {}

        if "monitor_poll_interval" in ft:
            defaults["poll_interval_seconds"] = int(ft["monitor_poll_interval"])
        if "max_worker_retries" in ft:
            defaults["max_reprovision_attempts"] = int(ft["max_worker_retries"])
        if "max_total_spend" in cost:
            defaults["cost_cap_usd"] = float(cost["max_total_spend"])
        if "auto_teardown" in cost:
            defaults["auto_teardown"] = bool(cost["auto_teardown"])
        if "data_files" in data:
            defaults["data_files"] = list(data["data_files"] or [])
        if "graceful_shutdown_timeout" in worker:
            defaults["worker_launch_delay"] = max(
                defaults["worker_launch_delay"], int(worker["graceful_shutdown_timeout"]) // 12 or 5
            )
        if "trial_timeout" in worker:
            defaults["trial_timeout_seconds"] = int(worker["trial_timeout"])
        if "gpu_filter" in vastai:
            gf = vastai["gpu_filter"]
            defaults["vastai_gpu_filter"] = ",".join(gf) if isinstance(gf, list) else str(gf)
        if "max_price_per_hour" in vastai:
            defaults["vastai_max_price_per_hour"] = float(vastai["max_price_per_hour"])
        if "disk_gb" in vastai:
            defaults["vastai_disk_gb"] = int(vastai["disk_gb"])
        if "docker_image" in vastai:
            defaults["vastai_docker_image"] = str(vastai["docker_image"])
        if "instances" in gpuhub:
            defaults["gpuhub_instances"] = list(gpuhub["instances"] or [])
        if "entity" in wb:
            defaults["wandb_entity"] = str(wb["entity"])
        if "project" in wb:
            defaults["wandb_project"] = str(wb["project"])
        if "group_prefix" in wb:
            defaults["wandb_group_prefix"] = str(wb["group_prefix"])

    # Back-compat: also accept flat top-level keys (override nested values)
    for k in list(defaults.keys()):
        if k in overlay:
            defaults[k] = overlay[k]
    return defaults


# ---------------------------------------------------------------------------
# Zip creation (mirrored from deploy scripts — not imported as they're scripts)
# ---------------------------------------------------------------------------

def create_filtered_zip(source_dir: Path, output_filename: str) -> str:
    """Create a filtered project zip for deployment, excluding heavy/transient dirs."""
    logger.info("Creating deployment package: %s", output_filename)
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
    logger.info("Package created: %.2f MB", size_mb)
    return output_filename


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_connect_gpuhub(host: str, port: int, password: str, timeout: int = 30):
    """Open paramiko SSH to a GPUHub instance (password auth)."""
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username="root", password=password, timeout=timeout)
    return ssh


def _ssh_connect_vastai(instance_id: str, timeout: int = 30):
    """Open paramiko SSH to a Vast.ai instance (key auth)."""
    import paramiko

    # Parse SSH URL from vastai CLI
    result = subprocess.run(
        ["vastai", "ssh-url", str(instance_id)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    raw = result.stdout.strip()
    m = re.match(r"ssh://(?:\w+@)?([^:]+):(\d+)", raw)
    if not m:
        m = re.match(r"(?:\w+@)?(\S+)\s+-p\s+(\d+)", raw)
    if not m:
        raise RuntimeError(f"Could not parse SSH URL from vastai: '{raw}'")

    host, port = m.group(1), int(m.group(2))

    # Find SSH key
    ssh_dir = Path.home() / ".ssh"
    key_path = None
    for name in ("id_ed25519", "id_rsa"):
        p = ssh_dir / name
        if p.exists():
            key_path = str(p)
            break
    if not key_path:
        raise FileNotFoundError("No SSH key found (~/.ssh/id_ed25519 or ~/.ssh/id_rsa)")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username="root", key_filename=key_path, timeout=timeout)
    return ssh


def _exec_ssh(ssh, cmd: str, timeout: int = 300) -> tuple[str, str]:
    """Execute a command over SSH and return (stdout, stderr)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err


def _exec_ssh_stdin(ssh, script: str, timeout: int = 30) -> tuple[str, str]:
    """Execute ``bash -s`` and pipe ``script`` via stdin.

    Use this for any launch that embeds secrets — argv only contains
    ``bash -s`` (the script body is on stdin, not on the command line), so
    secrets never appear in remote ``ps -ef`` output. The script's own
    ``export VAR=...`` lines put the values into the shell's environment;
    any forked python child inherits them via the env, not via argv.
    """
    stdin, stdout, stderr = ssh.exec_command("bash -s", timeout=timeout)
    stdin.write(script)
    stdin.flush()
    stdin.channel.shutdown_write()  # signal EOF to bash
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err


# ---------------------------------------------------------------------------
# Instance provisioning
# ---------------------------------------------------------------------------

def provision_vastai_instances(n_workers: int, gpu_filter: str | None) -> list[dict]:
    """Search and create Vast.ai instances. Returns list of worker dicts."""
    gpu_clause = f'gpu_name="{gpu_filter}"' if gpu_filter else 'gpu_name in ["RTX_3090","RTX_4090"]'
    filter_str = (
        f"{gpu_clause} cpu_cores>=8 cpu_ram>=32 disk_space>=50 "
        "cuda_vers>=12.0 reliability>0.95 num_gpus=1"
    )

    logger.info("Searching Vast.ai: %s (need %d instances)", filter_str, n_workers)
    result = subprocess.run(
        ["vastai", "search", "offers", "--raw", "--order", "dph_total", "--limit", str(n_workers + 5), filter_str],
        capture_output=True, text=True, check=True, timeout=120,
    )
    offers = json.loads(result.stdout)

    if len(offers) < n_workers:
        logger.warning("Only %d offers available (requested %d)", len(offers), n_workers)

    workers = []
    for i, offer in enumerate(offers[:n_workers]):
        offer_id = str(offer["id"])
        dph = offer.get("dph_total", 0)
        gpu_name = offer.get("gpu_name", "unknown")
        logger.info("Creating instance %d/%d from offer %s (%s, $%.3f/hr)",
                     i + 1, n_workers, offer_id, gpu_name, dph)

        create_result = subprocess.run(
            ["vastai", "create", "instance", offer_id,
             "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
             "--disk", "50", "--direct"],
            capture_output=True, text=True, check=True, timeout=120,
        )

        # Parse instance ID from output
        m = re.search(r"'new_contract':\s*(\d+)", create_result.stdout)
        if not m:
            m = re.search(r'"new_contract":\s*(\d+)', create_result.stdout)
        if not m:
            m = re.search(r"(\d{5,})", create_result.stdout)

        if m:
            instance_id = m.group(1)
            workers.append({
                "worker_id": f"vastai-{instance_id}",
                "platform": "vastai",
                "instance_id": instance_id,
                "gpu_name": gpu_name,
                "dph": dph,
                "status": "provisioning",
            })
            logger.info("  -> Instance %s created", instance_id)
        else:
            logger.error("  -> Failed to parse instance ID from: %s", create_result.stdout)

    # Wait for instances to reach 'running' state
    logger.info("Waiting for %d instances to start...", len(workers))
    for attempt in range(30):
        all_running = True
        for w in workers:
            if w["status"] == "running":
                continue
            try:
                show = subprocess.run(
                    ["vastai", "show", "instance", w["instance_id"], "--raw"],
                    capture_output=True, text=True, check=False, timeout=60,
                )
                if show.returncode == 0:
                    info = json.loads(show.stdout)
                    actual = info.get("actual_status", "")
                    if actual == "running":
                        w["status"] = "running"
                        logger.info("  Instance %s is RUNNING", w["instance_id"])
                    else:
                        all_running = False
                else:
                    all_running = False
            except Exception:
                all_running = False

        if all_running:
            break
        time.sleep(10)

    running = [w for w in workers if w["status"] == "running"]
    logger.info("%d/%d instances running", len(running), len(workers))
    return workers


def resolve_gpuhub_instances(
    gpu_filter: str | None,
    safe_slots: list[str] | None = None,
) -> list[dict]:
    """Read GPUHub instances from instances.json, optionally filtered.

    Args:
        gpu_filter: Comma-separated GPU name filter (e.g. "RTX_4090").
        safe_slots: List of "name:gpu_idx" strings (e.g. "gpuhub-1:0"). When
            provided, only listed slots are used; **duplicates spawn additional
            replica workers on the same physical GPU** (e.g. listing
            "gpuhub-1:0" three times → 3 workers with worker_ids
            ``gpuhub-1-gpu0-r0/-r1/-r2`` sharing one GPU). Each replica gets
            its own PID/log files so they don't collide. Useful for SAC HPO
            where one worker only utilizes ~25% of the GPU (CPU-bound on
            env.step) — co-locating 2-3 replicas typically gets 1.5-2.5×
            aggregate throughput per GPU.
    """
    instances_file = PROJECT_ROOT / "instances.json"
    if not instances_file.exists():
        raise FileNotFoundError(f"instances.json not found at {instances_file}")

    with open(instances_file, encoding="utf-8") as f:
        registry = json.load(f)

    # Count requested replicas per slot. None means "default 1 per GPU"
    # (legacy behavior when no overlay safe_slots passed).
    slot_counts: dict[tuple[str, int], int] = {}
    if safe_slots:
        for slot in safe_slots:
            try:
                name, idx = slot.rsplit(":", 1)
                key = (name.strip(), int(idx))
                slot_counts[key] = slot_counts.get(key, 0) + 1
            except (ValueError, AttributeError):
                logger.warning("Skipping malformed safe-slot entry: %r", slot)

    workers = []
    for name, inst in registry.get("instances", {}).items():
        gpus = inst.get("gpus", [])
        # Filter by GPU name if specified
        if gpu_filter:
            filters = [g.strip() for g in gpu_filter.split(",")]
            if not any(g in gpu for g in filters for gpu in gpus):
                continue

        for gpu_idx, gpu_name in enumerate(gpus):
            if safe_slots:
                replicas = slot_counts.get((name, gpu_idx), 0)
                if replicas == 0:
                    logger.info("Skipping %s:gpu%d (not in safe-slot list)", name, gpu_idx)
                    continue
            else:
                replicas = 1  # legacy: one worker per GPU
            for r in range(replicas):
                wid_suffix = f"-r{r}" if replicas > 1 else ""
                workers.append({
                    "worker_id": f"{name}-gpu{gpu_idx}{wid_suffix}",
                    "platform": "gpuhub",
                    "instance_name": name,
                    "host": inst["host"],
                    "port": inst["port"],
                    "password": inst["password"],
                    "gpu_idx": gpu_idx,
                    "gpu_name": gpu_name,
                    "replica": r,
                    "replicas_on_slot": replicas,
                    "status": "running",  # GPUHub instances are always-on
                })

    logger.info(
        "Resolved %d GPUHub worker slots (filter=%s, safe_slots=%s)",
        len(workers), gpu_filter, safe_slots,
    )
    return workers


# ---------------------------------------------------------------------------
# Deployment to a single worker
# ---------------------------------------------------------------------------

def deploy_to_worker(
    worker: dict,
    zip_path: str,
    data_files: list[str],
    config_path: str,
    db_url: str,
    study_name: str,
    target_trials: int,
    wandb_group: str,
    dist_config: dict,
    agent_type: str = "sac",
    wandb_run_id: str | None = None,
) -> bool:
    """Deploy code and launch distributed_hpo_worker.py on a single worker.
    Returns True on success."""
    worker_id = worker["worker_id"]
    logger.info("Deploying to worker %s...", worker_id)

    ssh_timeout = dist_config.get("ssh_timeout", 30)
    setup_timeout = dist_config.get("setup_timeout", 600)

    try:
        # Connect
        if worker["platform"] == "vastai":
            ssh = _ssh_connect_vastai(worker["instance_id"], timeout=ssh_timeout)
        else:
            ssh = _ssh_connect_gpuhub(
                worker["host"], worker["port"], worker["password"], timeout=ssh_timeout,
            )

        sftp = ssh.open_sftp()

        # Prepare remote workspace
        _exec_ssh(ssh, f"mkdir -p {REMOTE_WORKSPACE}")
        _exec_ssh(ssh, f"rm -rf {REMOTE_WORKSPACE}/*.zip")

        # Upload code zip
        logger.info("  [%s] Uploading code package...", worker_id)
        sftp.put(zip_path, f"{REMOTE_WORKSPACE}/{os.path.basename(zip_path)}")

        # Upload data files (with size-based caching)
        for data_file in data_files:
            local_path = PROJECT_ROOT / "data" / data_file
            if not local_path.exists():
                logger.warning("  [%s] Data file not found: %s", worker_id, local_path)
                continue

            remote_dir = f"{REMOTE_WORKSPACE}/data/{os.path.dirname(data_file)}".rstrip("/")
            remote_path = f"{REMOTE_WORKSPACE}/data/{data_file}"
            _exec_ssh(ssh, f"mkdir -p {remote_dir}")

            local_size = os.path.getsize(local_path)
            should_upload = True
            try:
                remote_attr = sftp.stat(remote_path)
                if remote_attr.st_size == local_size:
                    logger.info("  [%s] Cache hit for %s (%.2f MB)", worker_id, data_file, local_size / 1024 / 1024)
                    should_upload = False
            except IOError:
                pass

            if should_upload:
                logger.info("  [%s] Uploading %s (%.2f MB)...", worker_id, data_file, local_size / 1024 / 1024)
                sftp.put(str(local_path), remote_path)

        sftp.close()

        # Extract & install
        logger.info("  [%s] Setting up environment...", worker_id)
        zip_name = os.path.basename(zip_path)

        # Detect Python/pip path (GPUHub uses miniconda, Vast.ai uses system)
        if worker["platform"] == "gpuhub":
            pip_cmd = "/root/miniconda3/bin/pip"
            python_cmd = "/root/miniconda3/bin/python"
            path_export = "export PATH=/root/miniconda3/bin:$PATH"
        else:
            pip_cmd = "pip"
            python_cmd = "python"
            path_export = "true"  # no-op

        setup_cmds = [
            f"cd {REMOTE_WORKSPACE}",
            path_export,
            "ulimit -n 65536",
            f"{pip_cmd} uninstall sharpen finrl-pro-ds -y || true",
            "rm -rf sharpen.egg-info finrl_pro_ds.egg-info build dist",
            f"unzip -o {zip_name}",
            f"rm -f {zip_name}",
            f"{pip_cmd} install -q --upgrade -r requirements.txt",
            # [distributed] extra ships psycopg[binary] for Optuna RDBStorage
            f"{pip_cmd} install -q -e .[distributed]",
        ]
        setup_chain = " && ".join(setup_cmds) + " && echo SETUP_SUCCESS"
        out, err = _exec_ssh(ssh, setup_chain, timeout=setup_timeout)

        if "SETUP_SUCCESS" not in out:
            logger.error("  [%s] Setup failed. stdout: %s, stderr: %s", worker_id, out[:500], err[:500])
            ssh.close()
            return False
        logger.info("  [%s] Setup complete", worker_id)

        # Build launch script (multi-line bash, secrets via `export`, then
        # nohup'd python). DHP-CRED-LEAK fix: secrets go into bash env via
        # heredoc-style stdin (not into argv), so remote `ps -ef` shows only
        # `bash -s` — neither db_url nor WANDB_API_KEY is visible. The python
        # child inherits the env vars and reads DISTRIBUTED_HPO_DB_URL itself
        # (worker.py: --db_url is now optional, falls back to env).
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        log_file = f"worker_{worker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        export_lines = [
            f"export PATH={shlex.quote('/root/miniconda3/bin')}:$PATH"
            if worker["platform"] == "gpuhub" else "true",
            f"export CUDA_VISIBLE_DEVICES={worker.get('gpu_idx', 0)}",
            f"export DISTRIBUTED_HPO_DB_URL={shlex.quote(db_url)}",
        ]
        if wandb_key:
            export_lines.append(f"export WANDB_API_KEY={shlex.quote(wandb_key)}")
        if wandb_run_id:
            # S488 round-2 consolidation: worker attaches to coordinator's
            # parent run via init_wandb(). Namespace rotates per-trial inside
            # the objective — we do NOT set FINRL_WANDB_NAMESPACE here.
            export_lines.append(
                f"export FINRL_WANDB_RUN_ID={shlex.quote(wandb_run_id)}"
            )

        # `< /dev/null > log 2>&1 &` keeps the python child detached from the
        # SSH channel so bash exits cleanly. Even so, paramiko's stdout.read()
        # often hangs until our 30s timeout — that's tolerated below by
        # catching socket.timeout and verifying via PID file (DHP-FALSE-FAIL
        # fix: positive PID verification is the source of truth, not the
        # exec_command return).
        launch_script = "\n".join([
            f"cd {REMOTE_WORKSPACE}",
            *export_lines,
            "ulimit -n 65535 || true",
            (
                f"nohup {python_cmd} -u scripts/distributed_hpo_worker.py "
                f"--config {shlex.quote(config_path)} "
                f"--study_name {shlex.quote(study_name)} "
                f"--worker_id {shlex.quote(str(worker_id))} "
                f"--target_trials {int(target_trials)} "
                f"--wandb_group {shlex.quote(wandb_group)} "
                f"--agent_type {shlex.quote(agent_type)} "
                f"< /dev/null > {log_file} 2>&1 &"
            ),
            f"echo $! > worker_{worker_id}.pid",
            "",  # trailing newline
        ])

        logger.info("  [%s] Launching worker (target_trials=%d)...", worker_id, target_trials)
        try:
            _exec_ssh_stdin(ssh, launch_script, timeout=30)
        except (socket.timeout, TimeoutError) as e:
            # Expected when nohup'd python keeps the channel's stdout pipe
            # open. The PID file is the source of truth — verified next.
            logger.info(
                "  [%s] launch_cmd hit %s after 30s (expected for nohup; verifying via PID file)",
                worker_id, type(e).__name__,
            )

        # Positive verification: PID file written and process alive.
        time.sleep(5)
        try:
            pid_out, _ = _exec_ssh(
                ssh, f"cat {REMOTE_WORKSPACE}/worker_{worker_id}.pid 2>/dev/null", timeout=10,
            )
        except (socket.timeout, TimeoutError):
            ssh.close()
            ssh = (
                _ssh_connect_vastai(worker["instance_id"], timeout=ssh_timeout)
                if worker["platform"] == "vastai"
                else _ssh_connect_gpuhub(
                    worker["host"], worker["port"], worker["password"], timeout=ssh_timeout,
                )
            )
            pid_out, _ = _exec_ssh(
                ssh, f"cat {REMOTE_WORKSPACE}/worker_{worker_id}.pid 2>/dev/null", timeout=10,
            )

        pid = pid_out.strip()
        if pid and pid.isdigit():
            alive_out, _ = _exec_ssh(ssh, f"ps -p {pid} -o pid= 2>/dev/null", timeout=10)
            if pid in alive_out:
                worker["pid"] = pid
                worker["log_file"] = log_file
                worker["status"] = "running"
                worker["launched_at"] = datetime.now().isoformat()
                logger.info("  [%s] Worker launched. PID=%s", worker_id, pid)
                ssh.close()
                return True
            logger.error(
                "  [%s] PID %s in file but process not alive. Tail of log:",
                worker_id, pid,
            )
            tail, _ = _exec_ssh(ssh, f"tail -20 {REMOTE_WORKSPACE}/{log_file}", timeout=10)
            logger.error("  [%s] %s", worker_id, tail[:1000])
            ssh.close()
            return False

        logger.error("  [%s] Worker PID not found. Check remote logs.", worker_id)
        try:
            tail, _ = _exec_ssh(ssh, f"tail -20 {REMOTE_WORKSPACE}/{log_file}", timeout=10)
            logger.error("  [%s] Last log lines: %s", worker_id, tail[:1000])
        except Exception:
            pass
        ssh.close()
        return False

    except Exception as e:
        logger.error(
            "  [%s] Deployment failed: %s: %s",
            worker_id, type(e).__name__, e or "<no message>",
        )
        worker["status"] = "failed"
        return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_worker_health(worker: dict, ssh_timeout: int = 15) -> str:
    """Check if a worker is alive via SSH. Returns 'alive', 'dead', or 'unreachable'."""
    pid = worker.get("pid")
    if not pid:
        return "dead"

    try:
        if worker["platform"] == "vastai":
            ssh = _ssh_connect_vastai(worker["instance_id"], timeout=ssh_timeout)
        else:
            ssh = _ssh_connect_gpuhub(
                worker["host"], worker["port"], worker["password"], timeout=ssh_timeout,
            )

        out, _ = _exec_ssh(ssh, f"ps -p {pid} -o pid= 2>/dev/null", timeout=10)
        ssh.close()

        if pid in out.strip():
            return "alive"
        else:
            return "dead"

    except Exception:
        return "unreachable"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class CostTracker:
    """Track instance-hours and estimated spend."""

    def __init__(self):
        self._workers: dict[str, dict] = {}

    def register(self, worker_id: str, dph: float, start_time: float):
        self._workers[worker_id] = {
            "dph": dph,
            "start_time": start_time,
            "end_time": None,
        }

    def mark_done(self, worker_id: str):
        if worker_id in self._workers:
            self._workers[worker_id]["end_time"] = time.time()

    def summary(self) -> dict:
        now = time.time()
        total_hours = 0.0
        total_cost = 0.0
        for wid, info in self._workers.items():
            end = info["end_time"] or now
            hours = (end - info["start_time"]) / 3600
            total_hours += hours
            total_cost += hours * info["dph"]
        return {
            "total_instance_hours": round(total_hours, 2),
            "estimated_cost_usd": round(total_cost, 2),
            "n_workers": len(self._workers),
        }


# ---------------------------------------------------------------------------
# Best params extraction & routing (mirrors run_full_pipeline.py lines 152-198)
# ---------------------------------------------------------------------------

def extract_and_route_best_params(study, agent_type: str = "sac") -> dict:
    """Extract best trial params and route them to the correct config sections.

    Routing logic is identical to run_full_pipeline.py run_hpo() to ensure
    consistency between serial and distributed HPO.
    """
    best = study.best_trial
    agent_key = agent_type if agent_type in ("ppo", "iqn", "sac") else "bdq"
    best_params = {
        "env": {"reward": {}},
        "agents": {agent_key: {}},
        "training": {},
    }

    # SAC routing (primary use case for distributed HPO)
    if agent_type == "sac":
        reward_params = {"dsr_eta", "reward_mode"}
        agent_params = {
            "lr_actor", "lr_critic", "lr_alpha", "tau", "initial_alpha",
            "gamma", "gradient_clip", "batch_size",
            "learning_rate", "buffer_size",
            "cvar_alpha", "n_quantiles", "kappa",
        }
        env_params = {
            "deadband_threshold", "stop_loss_bps", "max_holding_bars",
            "reward_scaling", "lambda_delta",
        }
    elif agent_type == "ppo":
        reward_params = set()
        agent_params = {
            "learning_rate", "ent_coef", "gae_lambda", "n_epochs",
            "target_kl", "max_grad_norm", "clip_eps",
        }
        env_params = set()
    elif agent_type == "iqn":
        reward_params = {"crra_gamma", "stay_reward_weight"}
        agent_params = {
            "learning_rate", "num_quantiles", "noisy_sigma0", "tau",
            "gamma", "gamma_short", "gamma_long", "n_step", "buffer_size",
        }
        env_params = set()
    else:
        reward_params = set()
        agent_params = {"auxiliary_weight", "learning_rate", "epsilon_end", "tau", "gamma"}
        env_params = set()

    network_params = {"hidden_dim"}

    for key, val in best.params.items():
        if key == "reward_mode":
            best_params["env"]["reward"]["mode"] = val
        elif key in reward_params:
            best_params["env"]["reward"][key] = val
        elif key in agent_params:
            best_params["agents"][agent_key][key] = val
        elif agent_type == "sac" and key in env_params:
            best_params["env"][key] = val
        elif key in network_params:
            if "network" not in best_params:
                best_params["network"] = {"micro_config": {}, "macro_config": {}}
            best_params["network"]["micro_config"]["hidden_size"] = val
            best_params["network"]["macro_config"] = {"hidden_sizes": [val, val // 2]}
        else:
            # Mirror serial run_full_pipeline.run_hpo: fail loudly on unknown
            # params so silently mis-routed best_params YAMLs never reach
            # downstream training.
            raise ValueError(
                f"HPO param '{key}' has no routing rule for agent_type='{agent_type}'. "
                f"Add it to reward_params/agent_params/env_params in "
                f"extract_and_route_best_params() (and the matching block in "
                f"run_full_pipeline.run_hpo)."
            )

    return best_params


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------

def run_monitor_loop(
    study_name: str,
    db_url: str,
    n_trials: int,
    workers: list[dict],
    cost_tracker: CostTracker,
    poll_interval: int = 300,
    max_reprovision: int = 2,
    cost_cap_usd: float | None = None,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
):
    """Poll Optuna study and worker health until all trials complete.

    If ``cost_cap_usd`` is provided, exits the loop as soon as estimated spend
    exceeds the cap so the caller can tear down workers before further charges.
    """
    import optuna

    reprovision_counts: dict[str, int] = {}
    completed = 0

    while True:
        # --- Optuna progress ---
        try:
            study = optuna.load_study(study_name=study_name, storage=db_url)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
            failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

            best_value = None
            if completed > 0:
                best_value = study.best_value

            cost = cost_tracker.summary()
            logger.info(
                "Progress: %d/%d complete, %d running, %d failed | "
                "Best PF: %s | Cost: $%.2f (%.1f instance-hrs)",
                completed, n_trials, running, failed,
                f"{best_value:.4f}" if best_value is not None else "N/A",
                cost["estimated_cost_usd"],
                cost["total_instance_hours"],
            )

            if completed >= n_trials:
                logger.info("All %d trials complete. Exiting monitor loop.", n_trials)
                return study

            # Cost cap enforcement (Vast.ai mainly — GPUHub dph defaults to 0)
            if cost_cap_usd is not None and cost["estimated_cost_usd"] >= cost_cap_usd:
                logger.error(
                    "Cost cap hit: $%.2f >= $%.2f. Stopping monitor — caller should tear down.",
                    cost["estimated_cost_usd"], cost_cap_usd,
                )
                return study

        except Exception as e:
            logger.warning("Optuna poll failed: %s", e)

        # --- Worker health ---
        alive_count = 0
        for w in workers:
            if w["status"] not in ("running", "alive"):
                continue

            health = check_worker_health(w, ssh_timeout=15)
            if health == "alive":
                alive_count += 1
            elif health in ("dead", "unreachable"):
                logger.warning("Worker %s is %s", w["worker_id"], health)
                w["status"] = health
                cost_tracker.mark_done(w["worker_id"])

                # Fault tolerance: log but don't re-provision by default
                wid = w["worker_id"]
                reprovision_counts[wid] = reprovision_counts.get(wid, 0) + 1
                if reprovision_counts[wid] <= max_reprovision:
                    logger.info(
                        "Worker %s failed (attempt %d/%d). "
                        "Manual re-provision may be needed.",
                        wid, reprovision_counts[wid], max_reprovision,
                    )

        if alive_count == 0 and completed < n_trials:
            logger.error(
                "All workers are dead but only %d/%d trials complete. "
                "Manual intervention required.",
                completed, n_trials,
            )
            # Return whatever we have — don't loop forever
            try:
                return optuna.load_study(study_name=study_name, storage=db_url)
            except Exception:
                return None

        # --- WandB health (optional) ---
        # Under S488 consolidation the study has ONE parent run (workers attach
        # via resume=allow); a `group=study_name` query therefore returns 1,
        # which was previously misread as "1 worker alive". Query by tag to
        # get the real count of worker attach-sessions. Best-effort.
        try:
            api = wandb.Api()
            runs = api.runs(
                f"{wandb_entity}/{wandb_project}",
                filters={"tags": study_name, "state": "running"},
            )
            wandb_running = len(list(runs))
            logger.info(
                "WandB: %d run session(s) active tagged '%s' (1 = parent only)",
                wandb_running, study_name,
            )
        except Exception:
            pass  # WandB check is best-effort

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

def teardown_vastai_instances(workers: list[dict]):
    """Destroy all Vast.ai instances used by workers."""
    for w in workers:
        if w["platform"] != "vastai":
            continue
        instance_id = w.get("instance_id")
        if not instance_id:
            continue
        logger.info("Destroying Vast.ai instance %s (%s)...", instance_id, w["worker_id"])
        try:
            subprocess.run(
                ["vastai", "destroy", "instance", str(instance_id)],
                capture_output=True, text=True, check=False, timeout=60,
            )
            w["status"] = "destroyed"
            logger.info("  Instance %s destroyed", instance_id)
        except Exception as e:
            logger.error("  Failed to destroy %s: %s", instance_id, e)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed HPO Coordinator — orchestrate parallel HPO across GPU instances",
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config path")
    parser.add_argument("--n_workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--n_trials", type=int, default=50, help="Total HPO trials across all workers (default: 50)")
    parser.add_argument("--platform", choices=["vastai", "gpuhub"], default="vastai",
                        help="GPU platform (default: vastai)")
    parser.add_argument("--gpu_filter", default=None,
                        help="Comma-separated GPU names to filter (e.g., RTX_4090,RTX_3090)")
    parser.add_argument("--db_url", required=True,
                        help="PostgreSQL URL for shared Optuna storage (e.g., postgresql://user:pass@host:5432/optuna)")
    parser.add_argument("--study_name", required=True, help="Optuna study name")
    parser.add_argument("--auto_teardown", action="store_true",
                        help="Destroy Vast.ai instances on completion")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study (load_if_exists=True)")
    parser.add_argument("--distributed_config", default=None,
                        help="Path to distributed HPO overlay YAML (e.g., configs/distributed_hpo.yaml)")
    parser.add_argument("--agent_type", choices=["sac", "ppo", "iqn", "bdq"], default=None,
                        help="HPO agent type. If omitted, inferred from base_config.agents.* (sac > ppo > iqn > bdq).")

    args = parser.parse_args()

    # Load configs
    base_config = load_config(args.config)
    dist_config = load_distributed_config(args.distributed_config)
    poll_interval = dist_config.get("poll_interval_seconds", 300)
    max_reprovision = dist_config.get("max_reprovision_attempts", 2)
    cost_cap_usd = dist_config.get("cost_cap_usd")
    wandb_entity = dist_config.get("wandb_entity", WANDB_ENTITY)
    wandb_project = dist_config.get("wandb_project", WANDB_PROJECT)

    # Resolve agent_type once — used both for worker launch and best-param routing
    if args.agent_type:
        agent_type = args.agent_type
    else:
        agents_section = base_config.get("agents", {}) or {}
        for candidate in ("sac", "ppo", "iqn", "bdq"):
            if candidate in agents_section:
                agent_type = candidate
                break
        else:
            agent_type = "sac"
    logger.info("Resolved agent_type=%s", agent_type)

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY not set in coordinator environment — workers will run "
            "without WandB credentials and may fail to log metrics."
        )

    logger.info("=" * 70)
    logger.info("  Distributed HPO Coordinator")
    logger.info("  Config:     %s", args.config)
    logger.info("  Platform:   %s", args.platform)
    logger.info("  Workers:    %d", args.n_workers)
    logger.info("  Trials:     %d", args.n_trials)
    logger.info("  Study:      %s", args.study_name)
    logger.info("  DB:         %s", args.db_url.split("@")[-1] if "@" in args.db_url else args.db_url)
    logger.info("  Resume:     %s", args.resume)
    logger.info("  Teardown:   %s", args.auto_teardown)
    logger.info("=" * 70)

    # =========================================================================
    # Step 1: Create shared Optuna study
    # =========================================================================
    import optuna
    from sharpen.hpo.sampler import create_sampler

    logger.info("Creating Optuna study '%s' (PostgreSQL-backed)...", args.study_name)

    hpo_config = base_config.get("hpo", {})
    sampler = create_sampler(hpo_config, distributed=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db_url,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=args.resume,
    )

    existing = len(study.trials)
    if existing > 0:
        logger.info("Resuming study with %d existing trials", existing)

    # Optional: enqueue baseline trial from config or distributed_config
    baseline = dist_config.get("baseline_trial")
    if baseline is None and base_config.get("hpo", {}).get("enqueue_baseline", False):
        # Build baseline from config (same logic as run_full_pipeline.py)
        agents_cfg = base_config.get("agents", {})
        if "sac" in agents_cfg:
            bs = agents_cfg["sac"]
            be = base_config.get("env", {})
            be_r = be.get("reward", {})
            baseline = {
                "lr_actor": bs.get("lr_actor", 3e-4),
                "lr_critic": bs.get("lr_critic", 3e-4),
                "lr_alpha": bs.get("lr_alpha", 3e-4),
                "tau": bs.get("tau", 0.005),
                "gamma": bs.get("gamma", 0.99),
                "initial_alpha": bs.get("initial_alpha", 0.2),
                "deadband_threshold": be.get("deadband_threshold", 0.25),
                "dsr_eta": be_r.get("dsr_eta", 0.001),
                "gradient_clip": bs.get("gradient_clip", 1.0),
            }
            # B3 fix: anchor max_leverage=1.0 in trial 0 when the search
            # space declares it, so we have a within-study control point.
            # Without this, Optuna freely samples max_leverage even for the
            # baseline anchor (observed: trial 0 drew 0.77 in first launch).
            if "max_leverage" in (base_config.get("hpo", {}).get("search_space", {}) or {}):
                baseline["max_leverage"] = 1.0

    if baseline and existing == 0:
        study.enqueue_trial(baseline)
        logger.info("Enqueued baseline trial: %s", baseline)

    # =========================================================================
    # Step 2: Create deployment package
    # =========================================================================
    zip_name = f"deploy_dhpo_{os.getpid()}.zip"
    zip_path = str(PROJECT_ROOT / zip_name)
    create_filtered_zip(PROJECT_ROOT, zip_path)

    # =========================================================================
    # Step 3: Provision instances
    # =========================================================================
    if args.platform == "vastai":
        workers = provision_vastai_instances(args.n_workers, args.gpu_filter)
    else:
        safe_slots = dist_config.get("gpuhub_instances")
        workers = resolve_gpuhub_instances(args.gpu_filter, safe_slots=safe_slots)
        # Limit to requested n_workers
        workers = workers[:args.n_workers]

    running_workers = [w for w in workers if w["status"] == "running"]
    if not running_workers:
        logger.error("No running instances available. Aborting.")
        _cleanup_zip(zip_path)
        sys.exit(1)

    logger.info("%d workers ready for deployment", len(running_workers))

    # =========================================================================
    # Step 4: Deploy and launch workers
    # =========================================================================
    cost_tracker = CostTracker()

    # Per arch doc ADR-3: pass --target_trials = args.n_trials to every worker.
    # The worker's dynamic-pull loop checks GLOBAL completed trials against the
    # target (worker.py: `if completed_trials >= target_trials: break`), so all
    # workers naturally stop together when the study hits n_trials. Dividing
    # n_trials // n_workers (the previous behavior) was a semantic bug — workers
    # treated their share as the global cap and stopped collectively at
    # n_trials // n_workers, not n_trials. Existing trials from --resume already
    # count toward the global completion total in the worker's check.
    trial_assignments = [args.n_trials] * len(running_workers)

    wandb_group = args.study_name
    data_files = dist_config.get("data_files", [])
    launch_delay = dist_config.get("worker_launch_delay", 5)

    # ------------------------------------------------------------------
    # Parent WandB run (S488 round-2 consolidation)
    # ------------------------------------------------------------------
    # One run per study. Workers attach via FINRL_WANDB_RUN_ID +
    # init_wandb() in attach-only mode; per-trial namespace is set inside
    # the Optuna objective. Group is retained so historical `--group`
    # filters keep working in the UI.
    wcfg = base_config.get("wandb", {}) or {}
    wandb_entity = wcfg.get("entity", "bigcan-chiwin-technology")
    wandb_project = wcfg.get("project", "FinRL-Pro-DS")
    wandb_run_id = wandb.util.generate_id()
    # Name format matches multiseed + WF: <prefix>_<YYYYMMDD_HHMMSS>.
    # `group=study_name` is the cross-run semantic ID (stable across
    # --resume); `name` is the per-invocation display label.
    dhpo_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_run = wandb.init(
        id=wandb_run_id,
        entity=wandb_entity,
        project=wandb_project,
        name=f"{args.study_name}_{dhpo_timestamp}",
        group=wandb_group,
        job_type="dhpo_study",
        tags=list(wcfg.get("tags", []) or []) + ["dhpo", "consolidated"],
        config={
            "experiment": "dhpo",
            "study_name": args.study_name,
            "n_workers": len(running_workers),
            "n_trials": args.n_trials,
            "agent_type": agent_type,
            "platform": args.platform,
            "config_path": args.config,
        },
    )
    logger.info("Parent WandB run: %s (id=%s)", parent_run.url, wandb_run_id)

    # Crash-safety (F7 audit fix): if main() exits via KeyboardInterrupt or an
    # unhandled exception before reaching the explicit wandb.finish() below,
    # this atexit hook closes the run so WandB doesn't hold it in "running"
    # until the stale-run GC kicks in. Idempotent with the explicit finish.
    def _finish_parent_on_exit():
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
    atexit.register(_finish_parent_on_exit)

    for i, worker in enumerate(running_workers):
        success = deploy_to_worker(
            worker=worker,
            zip_path=zip_path,
            data_files=data_files,
            config_path=args.config,
            db_url=args.db_url,
            study_name=args.study_name,
            target_trials=trial_assignments[i],
            wandb_group=wandb_group,
            dist_config=dist_config,
            agent_type=agent_type,
            wandb_run_id=wandb_run_id,
        )
        if success:
            dph = worker.get("dph", 0.0)
            # For GPUHub, estimate from dist_config cost overrides
            if dph == 0.0:
                cost_overrides = dist_config.get("cost_per_hour", {})
                dph = cost_overrides.get(worker.get("gpu_name", ""), 0.0)
            cost_tracker.register(worker["worker_id"], dph, time.time())
        else:
            logger.warning("Worker %s failed to deploy — continuing with remaining workers", worker["worker_id"])

        # Stagger launches slightly to avoid SSH storms
        if i < len(running_workers) - 1:
            time.sleep(launch_delay)

    deployed = [w for w in running_workers if w["status"] == "running"]
    logger.info("%d/%d workers deployed successfully", len(deployed), len(running_workers))

    if not deployed:
        logger.error("No workers deployed successfully. Aborting.")
        _cleanup_zip(zip_path)
        # Close parent run cleanly so WandB doesn't leave a ghost "running" run
        # (F4 audit fix). Tag exit_code=1 so the crashed-run view picks it up.
        if wandb.run is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception as _e:  # pragma: no cover
                logger.warning("wandb.finish(exit_code=1) raised: %s", _e)
        sys.exit(1)

    # =========================================================================
    # Step 5: Monitor loop
    # =========================================================================
    logger.info("Entering monitor loop (poll_interval=%ds)...", poll_interval)

    try:
        final_study = run_monitor_loop(
            study_name=args.study_name,
            db_url=args.db_url,
            n_trials=args.n_trials,
            workers=deployed,
            cost_tracker=cost_tracker,
            poll_interval=poll_interval,
            max_reprovision=max_reprovision,
            cost_cap_usd=cost_cap_usd,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
    except KeyboardInterrupt:
        logger.info("Monitor interrupted by user (Ctrl+C). Collecting results so far...")
        final_study = optuna.load_study(study_name=args.study_name, storage=args.db_url)

    # =========================================================================
    # Step 6: Extract best params and save
    # =========================================================================
    if final_study is not None:
        completed_trials = [t for t in final_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info("Study complete. %d trials finished.", len(completed_trials))

        if completed_trials:
            best = final_study.best_trial
            logger.info(
                "Best trial #%d: PF=%.4f, params=%s",
                best.number, best.value, best.params,
            )

            # Route best params using the same agent_type the workers ran under
            best_routed = extract_and_route_best_params(final_study, agent_type)

            # Save best params to YAML
            output_path = PROJECT_ROOT / "results" / f"best_params_{args.study_name}.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "study_name": args.study_name,
                "best_trial_number": best.number,
                "best_profit_factor": best.value,
                "best_raw_params": dict(best.params),
                "best_routed_params": best_routed,
                "n_trials_completed": len(completed_trials),
                "timestamp": datetime.now().isoformat(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
            logger.info("Best params saved to %s", output_path)
        else:
            logger.warning("No completed trials found in study.")

    # Final cost summary
    cost = cost_tracker.summary()
    logger.info(
        "Cost summary: %.1f instance-hours, $%.2f estimated, %d workers used",
        cost["total_instance_hours"],
        cost["estimated_cost_usd"],
        cost["n_workers"],
    )

    # =========================================================================
    # Step 7: Auto-teardown
    # =========================================================================
    teardown_now = args.auto_teardown or bool(dist_config.get("auto_teardown"))
    if teardown_now:
        logger.info("Auto-teardown enabled. Destroying instances...")
        teardown_vastai_instances(workers)
    else:
        if args.platform == "vastai":
            alive = [w for w in workers if w["status"] == "running"]
            if alive:
                ids = ", ".join(w["instance_id"] for w in alive)
                logger.info(
                    "Instances still running (no --auto_teardown): %s. "
                    "Destroy manually: vastai destroy instance <id>",
                    ids,
                )

    # Cleanup local zip
    _cleanup_zip(zip_path)

    # Finish parent WandB run (workers keep their own sessions open until
    # they exit; they attach via resume="allow" so their logs continue to
    # land on the same run even after coordinator finishes).
    if wandb.run is not None:
        try:
            wandb.finish()
        except Exception as _e:  # pragma: no cover — best-effort cleanup
            logger.warning("wandb.finish() raised: %s", _e)

    logger.info("Coordinator finished.")


def _cleanup_zip(zip_path: str):
    """Remove local deployment zip if it exists."""
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
