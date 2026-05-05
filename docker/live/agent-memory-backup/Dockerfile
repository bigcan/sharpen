# Nightly backup sidecar for agent-memory MCP LanceDB store.
#
# Build:
#   docker compose -f docker/live/docker-compose.yaml build agent-memory-backup
# (also auto-builds via `up` when image is missing)
#
# Runs busybox crond in the foreground, executing /usr/local/bin/backup.sh
# at 03:00 UTC daily. Each run tars /data/lancedb/v1 (mounted ro from the
# agent-memory-data volume) into /backups (mounted from agent-memory-backups
# named volume) and prunes archives older than BACKUP_RETENTION_DAYS (30).

FROM alpine:3.20

RUN apk add --no-cache tini bash tar gzip findutils coreutils tzdata \
 && mkdir -p /backups /var/log

COPY backup.sh /usr/local/bin/backup.sh
COPY crontab /etc/crontabs/root

RUN chmod +x /usr/local/bin/backup.sh \
 && chmod 0600 /etc/crontabs/root

ENV TZ=UTC \
    BACKUP_RETENTION_DAYS=30 \
    BACKUP_DIR=/backups \
    SOURCE_DIR=/data/lancedb

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["crond", "-f", "-l", "8"]
