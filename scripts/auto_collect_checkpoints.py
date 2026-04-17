#!/usr/bin/env python3
"""
Auto-Collect Checkpoints from GPUHub Instances
===============================================
Polls WandB for recently finished runs, finds the checkpoint on the correct
GPUHub instance via SFTP, and downloads it to local checkpoints/ directory.
The local folder is synced to Google Cloud automatically.

Idempotent: skips runs whose checkpoints already exist locally.

Usage:
    python scripts/auto_collect_checkpoints.py                    # Last 24h
    python scripts/auto_collect_checkpoints.py --hours 48         # Last 48h
    python scripts/auto_collect_checkpoints.py --run_id abc123    # Specific run
    python scripts/auto_collect_checkpoints.py --dry_run          # Preview only
    python scripts/auto_collect_checkpoints.py --all_instances    # Scan all (no WandB)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import paramiko
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_FILE = PROJECT_ROOT / "instances.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
COLLECTION_LOG = PROJECT_ROOT / "results" / "collection_log.json"
REMOTE_WORKSPACE = "/workspace/DeepScalper"
WANDB_ENTITY = "bigcan-chiwin-technology"
WANDB_PROJECT = "FinRL-Pro-DS"

# Named run prefixes — only these are worth auto-collecting.
# Timestamp-only dirs (e.g., 20260315_073204) are HPO trial intermediates.
NAMED_RUN_PREFIXES = (
    "gmgp1-", "sg1-", "signal-gate-", "mm-", "alphaseek-",
    "sync-", "funding-arb-", "DeepScalper_",
    "gmgp1v", "sac_run",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_collect")


def load_instances():
    """Load GPU instances from instances.json."""
    if not INSTANCES_FILE.exists():
        log.error(f"instances.json not found at {INSTANCES_FILE}")
        sys.exit(1)
    with open(INSTANCES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("instances", {})


def load_collection_log():
    """Load previously collected run IDs."""
    if COLLECTION_LOG.exists():
        with open(COLLECTION_LOG) as f:
            return json.load(f)
    return {"collected": {}}


def save_collection_log(data):
    """Save collection log."""
    COLLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COLLECTION_LOG, "w") as f:
        json.dump(data, f, indent=2)


def get_finished_runs(hours=24, run_id=None):
    """Query WandB for recently finished runs."""
    api = wandb.Api()

    if run_id:
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            if run.state == "finished":
                return [run]
            log.warning(f"Run {run_id} state={run.state}, not 'finished'")
            return [run]  # Collect anyway — might have useful checkpoint
        except wandb.errors.CommError:
            log.error(f"Run {run_id} not found on WandB")
            return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={
            "state": "finished",
            "updatedAt": {"$gte": cutoff.isoformat()},
        },
        order="-updated_at",
    )
    return list(runs)


def sftp_connect(instance):
    """Create SSH/SFTP connection to a GPUHub instance."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        instance["host"],
        port=instance["port"],
        username="root",
        password=instance["password"],
        timeout=15,
    )
    sftp = ssh.open_sftp()
    return ssh, sftp


def find_and_download_checkpoint(run_name, instances, dry_run=False):
    """
    Search all instances for a run's checkpoint and download it.

    Returns (instance_name, local_path, file_count) or (None, None, 0).
    """
    for inst_name, inst in instances.items():
        try:
            ssh, sftp = sftp_connect(inst)
        except Exception as e:
            log.warning(f"  {inst_name}: connection failed ({e})")
            continue

        try:
            # Pattern 1: checkpoints/{run_name}/checkpoint_final.pth
            ckpt_dir = f"{REMOTE_WORKSPACE}/checkpoints/{run_name}"
            try:
                sftp.stat(ckpt_dir)
            except FileNotFoundError:
                log.debug(f"  {inst_name}: no dir {ckpt_dir}")
                continue

            # Found the directory — list checkpoint files
            files = []
            for entry in sftp.listdir_attr(ckpt_dir):
                if entry.filename.endswith((".pth", ".zip")):
                    files.append(entry)

            if not files:
                log.debug(f"  {inst_name}: dir exists but no .pth/.zip files")
                continue

            # Check for checkpoint_final specifically
            final_files = [f for f in files if "final" in f.filename]
            if not final_files:
                log.info(
                    f"  {inst_name}: found {len(files)} checkpoint(s) but no final "
                    f"(run may still be in progress)"
                )
                # Still download the latest step checkpoint as a safety measure
                files.sort(key=lambda f: f.st_mtime, reverse=True)
                final_files = [files[0]]
                log.info(f"  {inst_name}: using latest: {final_files[0].filename}")

            # Download
            local_dir = CHECKPOINTS_DIR / run_name
            if dry_run:
                total_mb = sum(f.st_size for f in final_files) / 1024 / 1024
                log.info(
                    f"  {inst_name}: WOULD download {len(final_files)} file(s) "
                    f"({total_mb:.1f} MB) -> {local_dir}"
                )
                return inst_name, str(local_dir), len(final_files)

            local_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            for entry in final_files:
                remote_path = f"{ckpt_dir}/{entry.filename}"
                local_path = local_dir / entry.filename
                if local_path.exists() and local_path.stat().st_size == entry.st_size:
                    log.info(f"  {inst_name}: {entry.filename} already exists, skipping")
                    downloaded += 1
                    continue
                size_mb = entry.st_size / 1024 / 1024
                log.info(f"  {inst_name}: downloading {entry.filename} ({size_mb:.1f} MB)...")
                sftp.get(remote_path, str(local_path))
                downloaded += 1
                log.info(f"  {inst_name}: saved -> {local_path}")

            return inst_name, str(local_dir), downloaded

        except Exception as e:
            log.warning(f"  {inst_name}: error scanning ({e})")
        finally:
            try:
                sftp.close()
                ssh.close()
            except Exception:
                pass

    # Pattern 2: SB3 models in hpo_results/ (funding-arb, crypto)
    for inst_name, inst in instances.items():
        try:
            ssh, sftp = sftp_connect(inst)
        except Exception:
            continue

        try:
            hpo_dir = f"{REMOTE_WORKSPACE}/hpo_results"
            try:
                sftp.stat(hpo_dir)
            except FileNotFoundError:
                continue

            # Search subdirectories for .zip files
            zip_files = []
            for subdir in sftp.listdir(hpo_dir):
                sub_path = f"{hpo_dir}/{subdir}"
                try:
                    for entry in sftp.listdir_attr(sub_path):
                        if entry.filename.endswith(".zip"):
                            zip_files.append((subdir, entry))
                except Exception:
                    pass

            if not zip_files:
                continue

            if dry_run:
                total_mb = sum(e.st_size for _, e in zip_files) / 1024 / 1024
                log.info(
                    f"  {inst_name}: WOULD download {len(zip_files)} SB3 model(s) "
                    f"({total_mb:.1f} MB) from hpo_results/"
                )
                return inst_name, str(CHECKPOINTS_DIR / "hpo_results"), len(zip_files)

            downloaded = 0
            for subdir, entry in zip_files:
                local_sub = CHECKPOINTS_DIR / "hpo_results" / subdir
                local_sub.mkdir(parents=True, exist_ok=True)
                local_path = local_sub / entry.filename
                if local_path.exists() and local_path.stat().st_size == entry.st_size:
                    continue
                remote_path = f"{hpo_dir}/{subdir}/{entry.filename}"
                size_mb = entry.st_size / 1024 / 1024
                log.info(f"  {inst_name}: downloading {subdir}/{entry.filename} ({size_mb:.1f} MB)...")
                sftp.get(remote_path, str(local_path))
                downloaded += 1

            if downloaded > 0:
                return inst_name, str(CHECKPOINTS_DIR / "hpo_results"), downloaded

        except Exception as e:
            log.warning(f"  {inst_name}: hpo_results scan error ({e})")
        finally:
            try:
                sftp.close()
                ssh.close()
            except Exception:
                pass

    return None, None, 0


def is_named_run(dirname):
    """Check if a checkpoint dir is a named run (vs a timestamp-only HPO trial)."""
    return dirname.startswith(NAMED_RUN_PREFIXES)


def scan_all_instances(instances, dry_run=False, include_all=False):
    """Scan all instances for checkpoint_final.pth not yet collected locally.

    Args:
        include_all: If True, include timestamp-only HPO trial dirs too.
                     Default False = only named runs.
    """
    found = []
    for inst_name, inst in instances.items():
        log.info(f"Scanning {inst_name} ({inst['host']}:{inst['port']})...")
        try:
            ssh, sftp = sftp_connect(inst)
        except Exception as e:
            log.warning(f"  {inst_name}: connection failed ({e})")
            continue

        try:
            ckpt_base = f"{REMOTE_WORKSPACE}/checkpoints"
            try:
                dirs = sftp.listdir(ckpt_base)
            except FileNotFoundError:
                log.info(f"  {inst_name}: no checkpoints/ directory")
                continue

            skipped = 0
            for d in sorted(dirs):
                if not include_all and not is_named_run(d):
                    skipped += 1
                    continue

                remote_final = f"{ckpt_base}/{d}/checkpoint_final.pth"
                try:
                    attr = sftp.stat(remote_final)
                except FileNotFoundError:
                    continue

                local_final = CHECKPOINTS_DIR / d / "checkpoint_final.pth"
                size_mb = attr.st_size / 1024 / 1024
                if local_final.exists() and local_final.stat().st_size == attr.st_size:
                    continue  # Already collected

                found.append((inst_name, d, size_mb))
                if not dry_run:
                    local_final.parent.mkdir(parents=True, exist_ok=True)
                    log.info(f"  {inst_name}: downloading {d}/checkpoint_final.pth ({size_mb:.1f} MB)...")
                    sftp.get(remote_final, str(local_final))
                    log.info(f"  saved -> {local_final}")
                else:
                    log.info(f"  {inst_name}: WOULD download {d}/checkpoint_final.pth ({size_mb:.1f} MB)")

            if skipped:
                log.info(f"  {inst_name}: skipped {skipped} timestamp-only dirs (use --include_all to include)")

        except Exception as e:
            log.warning(f"  {inst_name}: scan error ({e})")
        finally:
            try:
                sftp.close()
                ssh.close()
            except Exception:
                pass

    return found


def main():
    parser = argparse.ArgumentParser(description="Auto-collect checkpoints from GPUHub instances")
    parser.add_argument("--hours", type=int, default=24, help="Look back N hours for finished runs (default: 24)")
    parser.add_argument("--run_id", type=str, help="Collect a specific WandB run ID")
    parser.add_argument("--dry_run", action="store_true", help="Preview only, don't download")
    parser.add_argument("--all_instances", action="store_true", help="Scan all instances for uncollected checkpoints (no WandB needed)")
    parser.add_argument("--include_all", action="store_true", help="With --all_instances: include timestamp-only HPO trial dirs too")
    args = parser.parse_args()

    instances = load_instances()
    log.info(f"Loaded {len(instances)} instance(s): {', '.join(instances.keys())}")

    if args.all_instances:
        log.info(f"Scanning all instances for uncollected checkpoints...")
        found = scan_all_instances(instances, dry_run=args.dry_run, include_all=args.include_all)
        if found:
            log.info(f"\n{'DRY RUN: ' if args.dry_run else ''}Found {len(found)} uncollected checkpoint(s):")
            for inst, name, mb in found:
                log.info(f"  {inst}: {name} ({mb:.1f} MB)")
        else:
            log.info("All checkpoints already collected.")
        return

    # WandB-based collection
    log.info(f"Querying WandB for finished runs (last {args.hours}h)...")
    runs = get_finished_runs(hours=args.hours, run_id=args.run_id)
    log.info(f"Found {len(runs)} finished run(s)")

    if not runs:
        log.info("Nothing to collect.")
        return

    collection_log = load_collection_log()
    collected_count = 0
    skipped_count = 0

    for run in runs:
        run_id = run.id
        run_name = run.name or run_id

        # Skip if already collected
        if run_id in collection_log["collected"]:
            log.debug(f"Skipping {run_id} ({run_name}) — already collected")
            skipped_count += 1
            continue

        # Skip if checkpoint already exists locally
        local_final = CHECKPOINTS_DIR / run_name / "checkpoint_final.pth"
        if local_final.exists():
            log.info(f"Skipping {run_id} ({run_name}) — already exists locally")
            collection_log["collected"][run_id] = {
                "run_name": run_name,
                "collected_at": datetime.now().isoformat(),
                "local_path": str(local_final),
                "source": "pre-existing",
            }
            skipped_count += 1
            continue

        log.info(f"Collecting {run_id} ({run_name})...")
        inst_name, local_path, count = find_and_download_checkpoint(
            run_name, instances, dry_run=args.dry_run
        )

        if inst_name:
            collected_count += 1
            if not args.dry_run:
                collection_log["collected"][run_id] = {
                    "run_name": run_name,
                    "collected_at": datetime.now().isoformat(),
                    "local_path": local_path,
                    "source": inst_name,
                    "files": count,
                }
                log.info(f"  Collected {count} file(s) from {inst_name}")
        else:
            log.warning(f"  Checkpoint not found on any instance for {run_name}")

    if not args.dry_run:
        save_collection_log(collection_log)

    log.info(f"\nSummary: {collected_count} collected, {skipped_count} skipped")


if __name__ == "__main__":
    main()
