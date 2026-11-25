#!/usr/bin/env bash
# Restores a backup made by backup.sh into the Docker volumes. This REPLACES the current data.
#   bash restore.sh ~/atts-backups/atts-20261120-020000.tar.gz
set -euo pipefail
cd "$(dirname "$0")"

FILE="${1:?usage: bash restore.sh <backup.tar.gz>}"
[[ -f "$FILE" ]] || { echo "No such file: $FILE"; exit 1; }
read -r -p "This replaces the accounts, uploads, index and logs on this server. Type YES to continue: " ok
[[ "$ok" == "YES" ]] || { echo "Cancelled."; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar xzf "$FILE" -C "$TMP"

docker compose -f docker-compose.prod.yml stop backend streamlit
for v in atts_data atts_chroma atts_logs; do
  # The application runs as user 10001 (see Dockerfile); backups made before that hold files owned by root.
  docker run --rm -v "${v}:/dst" -v "${TMP}:/in:ro" alpine sh -c "rm -rf /dst/* /dst/.[!.]* 2>/dev/null; tar xzf /in/${v}.tar.gz -C /dst && chown -R 10001:10001 /dst"
done
if [[ -f "$TMP/env.backup" && ! -f .env ]]; then cp "$TMP/env.backup" .env; chmod 600 .env; fi
docker compose -f docker-compose.prod.yml start backend streamlit
echo "Restored."
