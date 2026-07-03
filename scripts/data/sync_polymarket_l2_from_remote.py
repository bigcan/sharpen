"""Pull completed Polymarket L2 collector output from GPUHub to local, then prune remote.

The live collector (scripts/data/collect_polymarket_updown_l2.py) runs unattended on a
GPUHub instance's data disk (/root/autodl-tmp, 50GB quota) for the pm-updown-mm-v0 forward
paper-test (configs/polymarket_updown_mm_paper.gates.yaml). At the observed rate
(~141MB in <1hr covering CLOB L2 + RTDS for btc+eth), the 50GB quota fills in roughly
12 days if never pruned -- well inside the 28-day collection window. This script must
run on a cadence tighter than that (weekly is fine IF each run deletes what it already
safely pulled, which is what this script does) or the collector will start dropping data
when the remote disk fills.

Safety: a file is only deleted remotely after its local copy's size matches exactly. The
currently-open hour file (writer never closes it until the hour rolls) is left alone --
only files with an older hour-stamp than the current UTC hour are eligible, so we never
pull/delete a file mid-write.

Usage:
    python scripts/data/sync_polymarket_l2_from_remote.py --selftest
    python scripts/data/sync_polymarket_l2_from_remote.py --instance gpuhub-2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import paramiko
from dotenv import load_dotenv

log = logging.getLogger("sync_polymarket_l2")
REMOTE_DIR = "/root/autodl-tmp/polymarket_l2/data"
LOCAL_DIR = Path("data/polymarket_updown/l2")
HOUR_STAMP_RE = re.compile(r"_(\d{8})_(\d{2})\.jsonl\.gz$")
# Allowlist by construction: only exact collector outputs. Anything else at the top level
# (stray subdirs, partial/debug artifacts) is left alone, not blindly pulled or deleted.
DATA_FILE_RE = re.compile(r"^(clob|rtds)_\d{8}_\d{2}\.jsonl\.gz$|^markets_\d{8}\.jsonl$")


def _connect(instance: str) -> paramiko.SSHClient:
    load_dotenv()
    root = Path(__file__).resolve().parents[2]
    with open(root / "instances.json", encoding="utf-8") as f:
        inst = json.load(f)["instances"][instance]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(inst["host"], port=inst["port"], username="root",
                password=os.environ["GPUHUB_PASSWORD"], timeout=30)
    return ssh


def _current_hour_stamp() -> str:
    return time.strftime("%Y%m%d_%H", time.gmtime())


def sync(instance: str, dry_run: bool = False) -> dict:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    ssh = _connect(instance)
    sftp = ssh.open_sftp()
    current_hour = _current_hour_stamp()

    pulled, skipped_open, skipped_unrecognized, failed = [], [], [], []
    try:
        remote_files = sftp.listdir(REMOTE_DIR)
    except FileNotFoundError:
        log.error("remote dir %s does not exist -- is the collector running?", REMOTE_DIR)
        return {"pulled": [], "skipped_open": [], "skipped_unrecognized": [], "failed": []}

    for fname in remote_files:
        if not DATA_FILE_RE.match(fname):
            skipped_unrecognized.append(fname)
            log.warning("skipping unrecognized entry (not a known data-file pattern): %s", fname)
            continue
        m = HOUR_STAMP_RE.search(fname)
        stamp = f"{m.group(1)}_{m.group(2)}" if m else None
        if stamp is not None and stamp >= current_hour:
            skipped_open.append(fname)  # still being written this hour, or future -- skip
            continue
        remote_path = f"{REMOTE_DIR}/{fname}"
        local_path = LOCAL_DIR / fname
        try:
            remote_size = sftp.stat(remote_path).st_size
            if dry_run:
                log.info("[dry-run] would pull %s (%d bytes)", fname, remote_size)
                pulled.append(fname)
                continue
            sftp.get(remote_path, str(local_path))
            local_size = local_path.stat().st_size
            if local_size != remote_size:
                raise IOError(f"size mismatch: remote={remote_size} local={local_size}")
            sftp.remove(remote_path)
            pulled.append(fname)
            log.info("pulled + pruned %s (%d bytes)", fname, remote_size)
        except Exception as e:  # noqa: BLE001 - report and continue with remaining files
            log.error("failed on %s: %s", fname, e)
            failed.append(fname)

    sftp.close()
    ssh.close()
    log.info("sync done: pulled=%d skipped_open=%d skipped_unrecognized=%d failed=%d",
              len(pulled), len(skipped_open), len(skipped_unrecognized), len(failed))
    return {"pulled": pulled, "skipped_open": skipped_open,
            "skipped_unrecognized": skipped_unrecognized, "failed": failed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="gpuhub-2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="Verify SSH connectivity only, no transfer")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.selftest:
        ssh = _connect(args.instance)
        ssh.exec_command("echo ok")[1].read()
        ssh.close()
        log.info("selftest OK - SSH connectivity to %s confirmed", args.instance)
        return

    result = sync(args.instance, dry_run=args.dry_run)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
