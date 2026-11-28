# Operations runbook: monitoring, alerts, backups and restores

What is watched, what each alert means and what to do, and how backups are made, kept and restored. Covers the
Windows setup (one PC) and the Docker server (`deploy/`).

---

## 1. What is in place

| Part | Where | What it does |
|---|---|---|
| `GET /health` | public | `healthy` / `degraded` and an ok flag for the model and the index. No details. Always HTTP 200 while the API answers. |
| `GET /health/details` | `MONITOR_TOKEN` or a super admin | Seven checks with their reasons; **HTTP 503 when any fails** (section 3). |
| `GET /metrics` | `MONITOR_TOKEN` or a super admin | Prometheus text: the checks, disk, backup age, model queue, outcomes per layer, latency per stage, ratings (section 5). |
| `scripts/backup.py` | nightly task / cron | Consistent snapshot, checks, manifest, age encryption, retention, off-site copy, status and heartbeat (section 6). |
| `scripts/restore.py` | by hand, monthly | Restore test into a scratch folder with six checks and the time taken; `--apply` for a real restore. |
| Log rotation | `app/log_files.py` | JSON-lines logs move to `logs/archive/` at `LOG_MAX_MB` (default 20). The log-in audit chain runs on across files. |
| Uptime Kuma | `127.0.0.1:3001` | Dashboard, history and alerts. Windows: `scripts/windows/uptime_kuma.ps1`; server: `uptime-kuma` in `deploy/docker-compose.prod.yml`. |
| Self-restart | Docker only | `app/healthcheck_probe.py` stops a backend that fails 3 health checks in a row, and Docker starts it again. |

**What this does not do.** It does not change any layer's accuracy, precision, recall or F1. The per-layer outcome
rates in `/metrics` show changes in production (for example Layer 1 suddenly blocking far more questions), but they
are rates, not accuracy: there are no labels to compare against. Monitoring on the same machine cannot report that
the machine itself is off; an outside check (for example UptimeRobot's free plan on the public address) covers that.

---

## 2. Set-up

### Settings (`.env`)

| Setting | Purpose |
|---|---|
| `MONITOR_TOKEN` | Read-only secret for `/health/details` and `/metrics`, at least 24 characters (shorter values are refused). Generate: `python -c "import secrets; print(secrets.token_urlsafe(36))"` |
| `BACKUP_DIR` | Where backups go (Windows here: `D:/ATTS-RAG-backups/auto`) |
| `BACKUP_AGE_RECIPIENT` | The age **public** key. Backups are refused without one, unless `--allow-unencrypted`. |
| `AGE_BIN`, `RCLONE_BIN` | Paths of the programs when they are not on PATH |
| `BACKUP_REMOTE` | rclone destination for the off-site copy, e.g. `gdrive:ATTS-RAG-backups`; empty = none |
| `BACKUP_HEARTBEAT_URL` | Uptime Kuma push URL of the "Backup" monitor |
| `BACKUP_KEEP` | Retention, default `7,4,6` (daily, weekly, monthly) |
| `BACKUP_MAX_AGE_HOURS` | The `backup` check fails after this many hours without a successful backup (default 26) |
| `BACKUP_CHECK=off` | Turns the backup check off (a development copy without backups) |
| `DISK_MIN_FREE_GB`, `DISK_MIN_FREE_PCT` | Disk alarm limits (default 5 GB and 10 %) |
| `LOG_MAX_MB`, `LOG_ARCHIVE_DAYS` | Log rotation size (default 20) and optional deletion of plain-log archives (default: keep) |
| `METRICS_WINDOW_MINUTES` | Window for the traffic metrics (default 60) |

### The backup key

```powershell
D:\ATTS-RAG-tools\age\age-keygen.exe -o D:\ATTS-RAG-secrets\backup-age-key.txt
```

The file holds the **private** key; its `# public key:` line is what goes into `BACKUP_AGE_RECIPIENT`. Whoever has
the private key can read every backup; without it **no backup can be restored**. Keep a copy off this computer (a
password manager, or a USB stick kept elsewhere), never in the backup folder or the off-site remote, and never on a
server.

### Windows (this PC)

1. Start Uptime Kuma, and register it to start at log-on:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\uptime_kuma.ps1 -Start -Register
   ```
2. Open <http://127.0.0.1:3001>: choose SQLite, create the admin account, then add the monitors of section 4 and a
   notification (Settings → Notifications: Telegram via @BotFather, or e-mail via SMTP). Tick "Default enabled" and
   "Apply on all existing monitors".
3. Copy the Backup push monitor's URL into `BACKUP_HEARTBEAT_URL`.
4. Off-site copy (optional): `D:\ATTS-RAG-tools\rclone\rclone.exe config` → `n` (new remote) → name `gdrive` → type
   `drive` (or `onedrive`) → accept the defaults and sign in in the browser. Then set
   `BACKUP_REMOTE=gdrive:ATTS-RAG-backups`. Only encrypted files are uploaded.
5. Register the nightly backup (02:00, or as soon as the PC is on):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\backup_task.ps1 -DataRoot <ATTS-RAG folder>
   ```
6. Run one backup and one restore test by hand (section 6) and check that the Backup monitor turns green.

### Docker server

1. `BACKUP_AGE_RECIPIENT` and `MONITOR_TOKEN` in `deploy/.env`. Create the backup folder for the container user:
   `sudo install -d -o 10001 -g 10001 -m 700 ~/atts-backups`.
2. `docker compose -f docker-compose.prod.yml up -d` starts `uptime-kuma` on `127.0.0.1:3001`. Open it through an
   SSH tunnel: `ssh -L 3001:127.0.0.1:3001 ubuntu@<server>`, then <http://localhost:3001>. Inside the compose network
   the backend is `http://backend:8000`, Ollama `http://ollama:11434`, the console `http://streamlit:8501`.
3. Cron: `0 2 * * *  cd /home/ubuntu/ATTS-RAG/deploy && bash backup.sh >> ~/atts-backup.log 2>&1`. For the off-site
   copy install rclone on the host (`sudo apt install rclone`, `rclone config`) and set `BACKUP_REMOTE`.

---

## 3. Health checks: what a failure means and what to do

`/health/details` returns `{"ok", "status", "failing": [...], "checks": {name: {"ok", "detail"}}}`.

| Check | Fails when | What to do |
|---|---|---|
| `ollama` | Ollama does not answer, or `OLLAMA_MODEL` is not installed | Start Ollama (Windows: the Ollama app; Docker: `docker compose ... restart ollama`). Missing model: `ollama pull llama3`. Questions get no answers until fixed; uploads still index. |
| `index` | Chroma, BM25 and the registry disagree, or the index cannot be opened | `python scripts/check_index.py`; if it reports a mismatch, stop the app and run `python scripts/reingest_corpus.py --rebuild --registered-only`. |
| `disk` | Free space below `DISK_MIN_FREE_GB` or `DISK_MIN_FREE_PCT` where `data/`, `chroma_db/` or `logs/` live | Free space: old files in `logs/archive/` (plain logs, not `auth_audit-*`), old `chroma_db.backup-*` folders, backups beyond retention. A full disk stops uploads, logs and backups. |
| `writable` | The app cannot create files in `data/`, `chroma_db/` or `logs/` | Permissions (Docker: `chown -R 10001:10001` on the volume), a read-only disk, or a full disk. |
| `accounts_db` | `data/users.db` is missing or fails SQLite's quick_check | Nobody can log in. Stop the app and restore `data/` from the last backup (section 6). |
| `audit_chain` | The log-in audit's hash chain is broken (a line edited or deleted, or an archive missing) | Treat as a security incident: keep the file as it is, note the time, compare with the last backup's copy, and find out who had access. The detail names the file and line. Checked every `AUDIT_CHECK_SECONDS` (600). |
| `backup` | No successful backup for `BACKUP_MAX_AGE_HOURS`, or the last run failed | Read `logs/backup_status.json` (`last_error`) and run `python scripts/backup.py` by hand to see the error. |

---

## 4. Uptime Kuma monitors

All with a 60-second interval and 3 retries before an alert, except the Backup push monitor.

| Monitor | Type | Target (Windows / Docker) | Alerts when |
|---|---|---|---|
| API | HTTP | `http://127.0.0.1:8000/health` / `http://backend:8000/health` | The backend does not answer. Windows: start it again; Docker restarts it by itself after 3 failed checks. |
| Health details | HTTP, header `Authorization: Bearer <MONITOR_TOKEN>` | `.../health/details` | Any check of section 3 fails (HTTP 503). The alert shows the body's `failing` list. |
| Ollama | HTTP | `http://127.0.0.1:11434/api/tags` / `http://ollama:11434/api/tags` | Ollama is down. |
| Console | HTTP | `http://127.0.0.1:8501/_stcore/health` / `http://streamlit:8501/_stcore/health` | The Streamlit console is down. |
| Backup | Push, heartbeat interval 93,600 s (26 h) | (Kuma gives the URL: put it in `BACKUP_HEARTBEAT_URL`) | No backup reported for 26 hours, or a backup reported failure (`status=down`). |
| Certificate (server) | HTTP on `https://<domain>`, "Certificate expiry notification" on | the public address | The HTTPS certificate expires within 14 days (Caddy renews it automatically; this catches a failed renewal). |

The token goes into Kuma's header field by you; it is not stored anywhere else in Kuma's settings.

---

## 5. Metrics

`curl -H "Authorization: Bearer $MONITOR_TOKEN" http://127.0.0.1:8000/metrics`

| Series | Meaning |
|---|---|
| `atts_check_ok{check}` | 1/0 per health check |
| `atts_disk_free_bytes{folder}`, `atts_index_chunks`, `atts_backup_age_seconds` | Disk, index size, backup age |
| `atts_llm_calls{state="waiting"\|"running"}` | Model calls queued and running now |
| `atts_requests{outcome,window}` | Questions in the window by outcome: `answered`, `blocked_l1`, `no_relevant`, `blocked_l2`, `rejected_l3`, `error` |
| `atts_latency_ms{stage,quantile,window}` | Median and 95th percentile per stage (`total`, `layer1`, `retrieval`, `rerank`, `layer2`, `assembly`, `layer3`, ...) |
| `atts_llm_wait_ms{quantile,window}` | Time model calls waited for a slot |
| `atts_feedback{rating,window}` | Thumbs up / down given |

The traffic series read the last 8 MB of `logs/traces.jsonl` within `METRICS_WINDOW_MINUTES`. For long-term charts,
point Prometheus at `/metrics` with the token, and add Grafana; Uptime Kuma alone keeps up/down history.

---

## 6. Backups and restores

**Schedule and retention.** Nightly at 02:00 (Windows task "ATTS-RAG nightly backup"; cron on the server). Kept: the
newest of each of the last 7 days, 4 ISO weeks and 6 months; at most 17 files. Off-site, files older than about
190 days are removed (Docker), or the same 7/4/6 rule is applied (Windows, `scripts/backup.py`). Data that can be
lost: at most one day.

**Contents.** `data/` (accounts, keys, uploads), `chroma_db/` (index), `logs/` (including the audit archives),
`.env` as `env.backup`, and `manifest.json` (every file's SHA-256, the account, chunk, document and audit counts,
five sample chunks). Everything is inside one age-encrypted file.

**Run by hand.**
```powershell
python scripts\backup.py --root <ATTS-RAG folder>
```

**Restore test (monthly, and after any change to the backup tools).**
```powershell
python scripts\restore.py D:\ATTS-RAG-backups\auto\atts-<time>.tar.gz.age --identity D:\ATTS-RAG-secrets\backup-age-key.txt
```
It decrypts into a temporary folder and must print `PASS` for `files`, `databases`, `accounts`, `index`, `search`
and `audit`, then the total time. Docker: `bash restore.sh --test <file> <key file>`.

**Real restore.** Stop the app first.
- Windows: the same command with `--root <ATTS-RAG folder> --apply --yes`. The current `data/`, `chroma_db/` and
  `logs/` are kept as `<name>.before-restore-<time>`; nothing is deleted.
- Docker: `bash restore.sh <file> <key file>` (asks for `YES`). Old-format backups (before 28 November 2026) still
  restore, without the checks.

Then log in, ask one question you know the answer to, and run `python scripts/check_index.py`.

---

## 7. Failure drills

Each failure is caused on purpose and must produce an alert. Repeat after changes to monitoring.

Run on 28 November 2026 on a restored copy of this PC's data (so the live installation was never touched), reading
`/health/details` after each failure was caused and undone:

| # | Failure caused | Expected | Result |
|---|---|---|---|
| 0 | none (baseline, and again after all drills) | HTTP 200, nothing failing | 200, nothing failing, both times |
| 1 | Ollama unreachable (`OLLAMA_HOST` pointed at a closed port; the real Ollama was left running) | `ollama` fails | 503, only `ollama` failing: "All connection attempts failed" |
| 2 | Model not installed (`OLLAMA_MODEL` set to a missing name) | `ollama` fails with "not installed" | 503, only `ollama`: "model ... is not installed; run: ollama pull ..." |
| 3 | Disk limit above the free space (`DISK_MIN_FREE_GB=100000`) | `disk` fails | 503, only `disk`: "low disk space: data (56.7 GB, 21.1% free) ..." |
| 4 | Index stores disagree (one chunk id removed from the registry in memory) | `index` fails | 503, only `index`: "the stores disagree: run scripts/check_index.py" |
| 5 | Account database damaged (4 KB overwritten) | `accounts_db` fails | 503, only `accounts_db`: "cannot read the account database: DatabaseError" |
| 6 | Audit log edited (the actor of record 2 changed) | `audit_chain` fails | 503, only `audit_chain`: "chain broken in auth_audit.jsonl at line 2" |
| 7 | No backup within the limit (`BACKUP_MAX_AGE_HOURS=0.0001`) | `backup` fails | 503, only `backup`: "last successful backup 0.1 h ago (limit 0.0001 h)" |
| 8 | Last backup run failed (status file with a newer failure) | `backup` fails | 503, only `backup`: "the last backup run failed: RuntimeError: drill" |
| 9 | Backend stopped (a real uvicorn on port 8010, killed) | `/health` unreachable, so the API monitor goes down | no connection after the kill |
| 10 | Backend hung (Docker only) | restarted after 3 failed checks, about 1.5 minutes | not run: needs the Docker server. The probe's logic is in `app/healthcheck_probe.py`, and CI checks the image imports it |

**Detected: 9 of 9 drills that can run on this PC**, each by exactly the check meant to catch it. Drill 10 is
still to run on a server. With Uptime Kuma's 60-second interval and 3 retries, an alert follows a failure within
about 3 to 4 minutes (Kuma itself was not yet configured when the drills ran, so this is the configured interval, not a
measured alert time).

---

## 8. Measured results (28 November 2026, this PC)

| What | Result |
|---|---|
| First backup of the live data | 26.3 MB encrypted, 35 files, 1 account, 372 chunks in 7 documents, 13 audit records; 45 s |
| Backup from the scheduled task | exit code 0, 42 s; retention then removed the earlier backup of the same day (newest per day kept) |
| Restore test on that backup | all 6 checks PASS (35-36 files, 2 databases, 1 account, 372 chunks in 7 documents, 5 of 5 sample chunks found, 13 audit records): decrypt and unpack 1 s, checks 39-42 s, **total 39-42 s** |
| Data that can be lost | at most 24 hours (nightly backup) |
| `/health` and `/metrics` response time | about 0.35 s each (3 calls each); `/health` does the same two probes as before this change |
| `/metrics` without the token | HTTP 401 |
| Off-site copy | not yet set up (needs the Google or Microsoft sign-in in `rclone config`) |

The 39 to 42 seconds cover the data only. A full recovery on a new machine also needs the code, Python packages and
models installed (section "Install" in the README), which takes much longer the first time.
