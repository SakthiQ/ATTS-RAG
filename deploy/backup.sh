#!/usr/bin/env bash
# Encrypted, checked backup of the Docker installation: accounts and uploads (atts_data), the index (atts_chroma),
# logs including the hash-chained audit log (atts_logs) and the settings (.env). Models are not backed up.
#
#   bash backup.sh                 writes ~/atts-backups/atts-YYYYmmdd-HHMMSS.tar.gz.age
#
# The backup itself is made by scripts/backup.py inside a one-off backend container (see that file for the steps:
# consistent SQLite copies, checks, manifest, age encryption, retention 7 daily / 4 weekly / 6 monthly). Settings in
# deploy/.env:
#   BACKUP_AGE_RECIPIENT=age1...      required: the public key (the private key stays off this server)
#   BACKUP_HEARTBEAT_URL=http://uptime-kuma:3001/api/push/<token>    optional push monitor
#   BACKUP_REMOTE=gdrive:ATTS-RAG-backups   optional off-site copy with rclone on this host (rclone config first)
#
# Once:   sudo install -d -o 10001 -g 10001 -m 700 ~/atts-backups      (the container user writes the backups)
# Nightly with cron (crontab -e):   0 2 * * *  cd /home/ubuntu/ATTS-RAG/deploy && bash backup.sh >> ~/atts-backup.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"

DEST="${BACKUP_DIR:-$HOME/atts-backups}"
mkdir -p "$DEST"
# shellcheck disable=SC1091
REMOTE="$(grep -E '^BACKUP_REMOTE=' .env 2>/dev/null | tail -n 1 | cut -d= -f2- || true)"
BEAT="$(grep -E '^BACKUP_HEARTBEAT_URL=' .env 2>/dev/null | tail -n 1 | cut -d= -f2- || true)"

# The off-site copy runs here on the host (rclone and its sign-in stay out of the container), so the container
# gets no BACKUP_REMOTE; it reports to the push monitor only when there is no off-site step after it.
container_beat="$BEAT"
[[ -n "$REMOTE" ]] && container_beat=""

docker compose -f docker-compose.prod.yml run --rm --no-deps \
  -e BACKUP_REMOTE= -e BACKUP_HEARTBEAT_URL="$container_beat" -e BACKUP_DIR=/backups \
  -v "$DEST:/backups" -v "$PWD/.env:/app/.env:ro" \
  backend python scripts/backup.py --root /app

if [[ -n "$REMOTE" ]]; then
  latest="$(ls -1t "$DEST"/atts-*.tar.gz.age | head -n 1)"
  if rclone copyto "$latest" "$REMOTE/$(basename "$latest")" \
     && rclone lsf "$REMOTE" | grep -qx "$(basename "$latest")"; then
    # Off-site retention: never mirror deletions from here (an emptied local folder must not empty the remote);
    # only remove off-site backups older than the oldest one local retention keeps (6 months).
    rclone delete --min-age 190d --include "atts-*.tar.gz.age" "$REMOTE" || true
    [[ -n "$BEAT" ]] && curl -fsS -m 15 "$BEAT?status=up&msg=backup+and+off-site+copy+ok" > /dev/null || true
    echo "Off-site copy: $REMOTE/$(basename "$latest")"
  else
    [[ -n "$BEAT" ]] && curl -fsS -m 15 "$BEAT?status=down&msg=off-site+copy+failed" > /dev/null || true
    echo "OFF-SITE COPY FAILED" >&2
    exit 1
  fi
fi
