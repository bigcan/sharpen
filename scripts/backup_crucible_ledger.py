"""Back up the Crucible ledgers to a durable (off-machine) location.

The Crucible DBs (``trial_ledger.db``, ``orchestrator.db``, ``catalog.db``) are git-ignored, so the
file-drawer N and the online-FDR wealth history live ONLY under ``results/crucible_orchestrator/``.
Lose that directory and the multiple-testing accounting silently restarts from zero. This script
mirrors it to a backed-up path (default: the Google-Drive folder) and is safe to call from the
nightly cron *after* a Crucible tick.

Why not ``cp``: a plain file copy of a live SQLite DB can capture a torn write mid-transaction. We
use the SQLite online-backup API (``Connection.backup``), which takes a transactionally-consistent
snapshot even while the orchestrator holds the DB open, then verify each copy with
``PRAGMA integrity_check``. Non-DB artifacts (manifests, scout reports, summaries) are plain-copied.

Layout at the destination::

    <dest>/snapshots/<UTC-timestamp>/   full point-in-time tree (dated; never overwritten)
    <dest>/latest/                      mirror of the most recent snapshot (stable path)

Snapshots are ~100 KB each, so a nightly cadence costs well under 50 MB/year — cheap insurance for
point-in-time recovery if a later run corrupts the working DBs.

Usage::

    python scripts/backup_crucible_ledger.py
    python scripts/backup_crucible_ledger.py --src results/crucible_orchestrator --dest "G:\\My Drive\\Crucible Alpha Mining"
    python scripts/backup_crucible_ledger.py --keep 90   # prune snapshots older than the newest 90
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backup_crucible_ledger")

DEFAULT_SRC = Path("results/crucible_orchestrator")
DEFAULT_DEST = Path(r"G:\My Drive\Crucible Alpha Mining")


def _sqlite_safe_backup(src_db: Path, dst_db: Path) -> None:
    """Transactionally-consistent snapshot of a live SQLite DB, then integrity-check the copy."""
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    # Open source read-only so we never disturb the orchestrator's writer.
    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(dst_db))
        try:
            src.backup(dst)  # online backup API — safe under concurrent writes
        finally:
            dst.close()
    finally:
        src.close()

    verify = sqlite3.connect(str(dst_db))
    try:
        result = verify.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        verify.close()
    if result != "ok":
        raise RuntimeError(f"integrity_check FAILED for {dst_db}: {result!r}")
    log.info("  db   %-24s -> ok (%d bytes)", src_db.name, dst_db.stat().st_size)


def backup(src_root: Path, dest_root: Path, keep: int | None) -> Path:
    if not src_root.exists():
        raise FileNotFoundError(f"source not found: {src_root.resolve()}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = dest_root / "snapshots" / stamp
    snapshot.mkdir(parents=True, exist_ok=True)
    log.info("snapshot -> %s", snapshot)

    n_db = n_file = 0
    for path in sorted(src_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src_root)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".db":
            _sqlite_safe_backup(path, target)
            n_db += 1
        else:
            shutil.copy2(path, target)
            log.info("  file %-24s -> copied (%d bytes)", rel.as_posix(), target.stat().st_size)
            n_file += 1

    # Refresh the stable 'latest' mirror.
    latest = dest_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(snapshot, latest)
    log.info("latest mirror refreshed -> %s", latest)

    if keep is not None and keep > 0:
        snaps = sorted((dest_root / "snapshots").iterdir())
        for old in snaps[:-keep]:
            shutil.rmtree(old)
            log.info("pruned old snapshot %s", old.name)

    log.info("DONE: %d db + %d artifact(s) backed up to %s", n_db, n_file, snapshot)
    return snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Crucible output root to back up")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="durable backup destination")
    ap.add_argument("--keep", type=int, default=None,
                    help="retain only the newest N snapshots (default: keep all)")
    args = ap.parse_args()
    try:
        backup(args.src, args.dest, args.keep)
    except Exception as exc:  # ops script: surface the failure loudly for the cron log
        log.error("BACKUP FAILED: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
