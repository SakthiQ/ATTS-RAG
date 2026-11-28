#!/usr/bin/env bash
# Restores a backup into the Docker volumes. This REPLACES the current accounts, uploads, index and logs.
#
#   bash restore.sh --test  ~/atts-backups/atts-20261128-020000.tar.gz.age ~/backup-key.txt   restore test only
#   bash restore.sh         ~/atts-backups/atts-20261128-020000.tar.gz.age ~/backup-key.txt   real restore
#
# The key is the age private key file (bring it to the server only for the restore, then remove it).
# Both forms first run scripts/restore.py in a one-off container: it decrypts into a scratch folder and checks the
# files, databases, account and chunk counts, search and the audit chain against the backup's manifest. The real
# restore continues only if every check passes. Backups made by the earlier version of backup.sh (one tarball per
# volume, not encrypted) are still restored, without the checks.
set -euo pipefail
cd "$(dirname "$0")"

TEST_ONLY=0
if [[ "${1:-}" == "--test" ]]; then TEST_ONLY=1; shift; fi
FILE="${1:?usage: bash restore.sh [--test] <backup> [<age key file>]}"
KEY="${2:-}"
[[ -f "$FILE" ]] || { echo "No such file: $FILE"; exit 1; }
FILE="$(realpath "$FILE")"

restore_volumes() {   # $1: folder holding data/, chroma_db/, logs/
  docker compose -f docker-compose.prod.yml stop backend streamlit
  for pair in data:atts_data chroma_db:atts_chroma logs:atts_logs; do
    src="${pair%%:*}" vol="${pair##*:}"
    docker run --rm -v "${vol}:/dst" -v "$1/${src}:/in:ro" alpine sh -c \
      "rm -rf /dst/* /dst/.[!.]* 2>/dev/null; cp -a /in/. /dst/ && chown -R 10001:10001 /dst"
  done
  if [[ -f "$1/env.backup" && ! -f .env ]]; then cp "$1/env.backup" .env; chmod 600 .env; fi
  docker compose -f docker-compose.prod.yml start backend streamlit
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [[ "$FILE" == *.tar.gz.age || ( "$FILE" == *.tar.gz && -z "$(tar tzf "$FILE" | grep -x './atts_data.tar.gz' || true)" ) ]]; then
  # Current format: check in a container first. It runs as this host user, so it can read the key file.
  if [[ "$FILE" == *.age ]]; then
    [[ -f "$KEY" ]] || { echo "This backup is encrypted: give the age key file as the second argument."; exit 1; }
    mounts=(-v "$FILE:/in/backup.tar.gz.age:ro" -v "$(realpath "$KEY"):/in/key:ro")
    args=(/in/backup.tar.gz.age --identity /in/key)
  else
    mounts=(-v "$FILE:/in/backup.tar.gz:ro")
    args=(/in/backup.tar.gz)
  fi
  docker compose -f docker-compose.prod.yml run --rm --no-deps -u "$(id -u):$(id -g)" -e HF_HUB_OFFLINE=1 "${mounts[@]}" backend python scripts/restore.py "${args[@]}"
  [[ "$TEST_ONLY" == 1 ]] && exit 0

  read -r -p "The checks passed. Replace the accounts, uploads, index and logs on this server? Type YES: " ok
  [[ "$ok" == "YES" ]] || { echo "Cancelled."; exit 1; }
  if [[ "$FILE" == *.age ]]; then
    age -d -i "$KEY" "$FILE" | tar xz -C "$TMP"
  else
    tar xzf "$FILE" -C "$TMP"
  fi
  restore_volumes "$TMP"
else
  # Earlier format: atts_data.tar.gz, atts_chroma.tar.gz, atts_logs.tar.gz and env.backup in one tarball
  [[ "$TEST_ONLY" == 1 ]] && { echo "Backups in the earlier format cannot be restore-tested; restore them for real."; exit 1; }
  read -r -p "This replaces the accounts, uploads, index and logs on this server. Type YES to continue: " ok
  [[ "$ok" == "YES" ]] || { echo "Cancelled."; exit 1; }
  tar xzf "$FILE" -C "$TMP"
  for pair in data:atts_data chroma_db:atts_chroma logs:atts_logs; do
    mkdir -p "$TMP/${pair%%:*}" && tar xzf "$TMP/${pair##*:}.tar.gz" -C "$TMP/${pair%%:*}"
  done
  restore_volumes "$TMP"
fi
echo "Restored."
