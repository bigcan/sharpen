#!/bin/sh
# Nightly tar+gzip of agent-memory LanceDB store with retention pruning.
# Output goes to container stdout (PID 1) so `docker logs agent-memory-backup`
# shows backup history.
set -eu

RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
SOURCE_DIR="${SOURCE_DIR:-/data/lancedb}"

TS=$(date -u +%Y-%m-%dT%H%M%SZ)
DEST="$BACKUP_DIR/agent-memory-$TS.tar.gz"
TMP="$DEST.tmp"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [agent-memory-backup] $*"; }

mkdir -p "$BACKUP_DIR"

if [ ! -d "$SOURCE_DIR/v1" ]; then
    log "FATAL source $SOURCE_DIR/v1 missing — abort" >&2
    exit 1
fi

log "start -> $DEST (source=$SOURCE_DIR retention=${RETENTION_DAYS}d)"

# tar from inside SOURCE_DIR so archive paths are relative ("v1/...").
# LanceDB is append-only with versioned manifests; a live tar may miss the
# very latest commit but prior versions remain consistent — restore-safe.
tar -czf "$TMP" -C "$SOURCE_DIR" v1
mv "$TMP" "$DEST"

SIZE=$(du -h "$DEST" | awk '{print $1}')
log "wrote $(basename "$DEST") ($SIZE)"

# Retention sweep
PRUNED=0
for old in $(find "$BACKUP_DIR" -maxdepth 1 -name 'agent-memory-*.tar.gz' \
                  -type f -mtime "+$RETENTION_DAYS" 2>/dev/null); do
    rm -f -- "$old"
    log "pruned $(basename "$old")"
    PRUNED=$((PRUNED+1))
done
log "retention sweep complete (pruned=$PRUNED, threshold=${RETENTION_DAYS}d)"

# Sanity: ensure the new archive still exists and is non-empty
if [ ! -s "$DEST" ]; then
    log "FATAL post-check: $DEST missing or empty" >&2
    exit 1
fi

# Inventory line for ops grepping
COUNT=$(find "$BACKUP_DIR" -maxdepth 1 -name 'agent-memory-*.tar.gz' -type f | wc -l)
TOTAL=$(du -sh "$BACKUP_DIR" 2>/dev/null | awk '{print $1}')
log "done backups=$COUNT total=$TOTAL"
