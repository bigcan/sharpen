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

Eval/WF artifact preservation
-----------------------------
The collector defaults to pulling **only** model checkpoints (.pth/.zip), but
eval-stage and walk-forward outputs (trajectory parquets, verdict.json,
ensemble bundles) live under remote `results/<workstream>/...` and were not
mirrored historically. Loss of those artifacts when gpuhub-1 was released
post-FU-3 (2026-05-07) forced a single-fold drift-baseline fallback. Use
`--results <relpath>` to mirror the results tree for a finished WF/ensemble
run before its host is recycled. Whitelisted suffixes only: .parquet, .json,
.csv, .tar.gz; files larger than --max-size-mb (default 500 MB) are skipped.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import paramiko
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_FILE = PROJECT_ROOT / "instances.json"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
COLLECTION_LOG = PROJECT_ROOT / "results" / "collection_log.json"
REMOTE_WORKSPACE = "/workspace/DeepScalper"
WANDB_ENTITY = "bigcan-chiwin-technology"
WANDB_PROJECT = "FinRL-Pro-DS"

# Whitelisted file suffixes for --results pulls. Trajectory parquets,
# verdicts/manifests, summary CSVs, and bundled ensembles are tiny enough
# to mirror; large opaques (replay buffers, raw datasets) stay remote.
RESULTS_WHITELIST_SUFFIXES = (".parquet", ".json", ".csv", ".tar.gz")
RESULTS_DEFAULT_MAX_MB = 500

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


def _has_whitelisted_suffix(name: str) -> bool:
    return any(name.endswith(s) for s in RESULTS_WHITELIST_SUFFIXES)


def _walk_remote(sftp, remote_dir):
    """Yield (remote_path, attr) for every file under remote_dir (recursive).

    paramiko's SFTPClient lacks os.walk; we DIY via listdir_attr + stat-mode.
    """
    import stat as _stat
    pending = [remote_dir]
    while pending:
        cur = pending.pop()
        try:
            entries = sftp.listdir_attr(cur)
        except FileNotFoundError:
            continue
        except Exception as exc:
            log.debug(f"  walk skip {cur}: {exc}")
            continue
        for entry in entries:
            full = f"{cur}/{entry.filename}"
            if _stat.S_ISDIR(entry.st_mode):
                pending.append(full)
            else:
                yield full, entry


def collect_results_dir(
    relpath: str,
    instances: dict,
    *,
    dry_run: bool = False,
    max_mb: int = RESULTS_DEFAULT_MAX_MB,
):
    """Mirror /workspace/DeepScalper/results/<relpath>/ from the first
    instance that has it. Whitelisted suffixes only; files >max_mb skipped.

    Returns (instance_name, files_downloaded, total_mb).
    """
    rel = relpath.strip("/").rstrip("/")
    if not rel:
        log.error("--results requires a non-empty path under results/")
        return None, 0, 0.0

    remote_root = f"{REMOTE_WORKSPACE}/results/{rel}"
    local_root = RESULTS_DIR / rel

    for inst_name, inst in instances.items():
        try:
            ssh, sftp = sftp_connect(inst)
        except Exception as e:
            log.debug(f"  {inst_name}: connection failed ({e})")
            continue

        try:
            try:
                sftp.stat(remote_root)
            except FileNotFoundError:
                log.debug(f"  {inst_name}: no {remote_root}")
                continue

            log.info(f"  {inst_name}: walking {remote_root}")
            files = []
            for full, attr in _walk_remote(sftp, remote_root):
                fname = full.rsplit("/", 1)[-1]
                if not _has_whitelisted_suffix(fname):
                    continue
                size_mb = attr.st_size / 1024 / 1024
                if size_mb > max_mb:
                    log.warning(
                        f"  {inst_name}: skip {full} ({size_mb:.1f} MB > "
                        f"--max-size-mb={max_mb})"
                    )
                    continue
                files.append((full, attr, size_mb))

            if not files:
                log.warning(
                    f"  {inst_name}: {remote_root} exists but no whitelisted "
                    f"files ({RESULTS_WHITELIST_SUFFIXES})"
                )
                return None, 0, 0.0

            total_mb = sum(s for _, _, s in files)
            if dry_run:
                log.info(
                    f"  {inst_name}: WOULD download {len(files)} file(s) "
                    f"({total_mb:.1f} MB) -> {local_root}"
                )
                for full, _, sm in files[:10]:
                    log.info(f"    {full.replace(remote_root + '/', '')} "
                             f"({sm:.2f} MB)")
                if len(files) > 10:
                    log.info(f"    ... +{len(files) - 10} more")
                return inst_name, len(files), total_mb

            downloaded = 0
            skipped = 0
            for full, attr, size_mb in files:
                rel_path = full[len(remote_root) + 1:]
                local_path = local_root / rel_path
                if (local_path.exists()
                        and local_path.stat().st_size == attr.st_size):
                    skipped += 1
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                log.info(f"  {inst_name}: download {rel_path} ({size_mb:.2f} MB)")
                sftp.get(full, str(local_path))
                downloaded += 1

            log.info(
                f"  {inst_name}: results pull complete "
                f"({downloaded} new, {skipped} already-local, "
                f"{total_mb:.1f} MB total) -> {local_root}"
            )
            return inst_name, downloaded, total_mb

        except Exception as e:
            log.warning(f"  {inst_name}: results scan error ({e})")
        finally:
            try:
                sftp.close()
                ssh.close()
            except Exception:
                pass

    log.warning(f"results/{rel} not found on any instance")
    return None, 0, 0.0


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
    parser.add_argument(
        "--results", action="append", default=None,
        help="Mirror /workspace/DeepScalper/results/<relpath>/ from the first "
             "instance that has it (whitelist: .parquet/.json/.csv/.tar.gz). "
             "Repeat for multiple. Use to preserve WF trajectory parquets + "
             "verdicts before the host instance is recycled. Example: "
             "'--results sg1_btc_velotrade_extended_ensemble'.",
    )
    parser.add_argument(
        "--max-size-mb", type=int, default=RESULTS_DEFAULT_MAX_MB,
        help=f"Per-file size cap for --results pulls (default: {RESULTS_DEFAULT_MAX_MB} MB)",
    )
    args = parser.parse_args()

    instances = load_instances()
    log.info(f"Loaded {len(instances)} instance(s): {', '.join(instances.keys())}")

    if args.results:
        log.info(f"Collecting results subdir(s): {args.results}")
        any_found = False
        for rel in args.results:
            inst, n, mb = collect_results_dir(
                rel, instances,
                dry_run=args.dry_run, max_mb=args.max_size_mb,
            )
            if inst:
                any_found = True
        if not any_found:
            log.warning("No results subdir was found on any instance.")
        # Results pulls are a standalone mode — checkpoint flow only runs
        # if the operator also passes --hours/--run_id/--all_instances.
        if not (args.run_id or args.all_instances or args.hours != 24):
            return

    if args.all_instances:
        log.info("Scanning all instances for uncollected checkpoints...")
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
