#!/usr/bin/env bash
# Backs up everything that cannot be re-created: accounts and uploads (atts_data), the index (atts_chroma),
# logs including the hash-chained audit log (atts_logs) and the settings (.env).
# The models (Ollama, Hugging Face) are not backed up; they download again.
#
#   bash backup.sh                       # writes ~/atts-backups/atts-YYYYmmdd-HHMMSS.tar.gz
#   BACKUP_S3=s3://my-bucket/atts bash backup.sh   # also copies it to S3 (needs the AWS CLI and an instance role)
#
# Run it from cron, for example every night:   0 2 * * *  cd /home/ubuntu/ATTS-RAG/deploy && bash backup.sh
# The data is consistent enough for a nightly copy; for a guaranteed-consistent copy stop the backend first:
#   docker compose -f docker-compose.prod.yml stop backend
set -euo pipefail
cd "$(dirname "$0")"

DEST="${BACKUP_DIR:-$HOME/atts-backups}"
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$DEST"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for v in atts_data atts_chroma atts_logs; do
  docker run --rm -v "${v}:/src:ro" -v "${TMP}:/out" alpine tar czf "/out/${v}.tar.gz" -C /src .
done
cp .env "${TMP}/env.backup"
chmod 600 "${TMP}/env.backup"

tar czf "${DEST}/atts-${STAMP}.tar.gz" -C "$TMP" .
chmod 600 "${DEST}/atts-${STAMP}.tar.gz"
echo "Backup written: ${DEST}/atts-${STAMP}.tar.gz"

if [[ -n "${BACKUP_S3:-}" ]]; then
  aws s3 cp "${DEST}/atts-${STAMP}.tar.gz" "${BACKUP_S3}/"
fi

# Keep the last 14 local backups.
ls -1t "${DEST}"/atts-*.tar.gz | tail -n +15 | xargs -r rm --
