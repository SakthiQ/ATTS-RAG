"""Makes one encrypted, checked backup of everything ATTS-RAG cannot re-create.

    python scripts/backup.py                       settings from .env (below)
    python scripts/backup.py --out D:/ATTS-RAG-backups/auto

What is backed up (from --root, default: the current folder):
  data/        accounts (users.db), the sign-in and 2FA keys, uploaded files
  chroma_db/   the index (Chroma, BM25, registry)
  logs/        logs, including the hash-chained log-in audit and its archives
  .env         the settings, stored in the archive as env.backup
Models are not backed up: they download again.

Steps, and what makes each one safe:
  1. Snapshot. SQLite files (users.db, chroma.sqlite3) are copied with SQLite's own backup method, which gives a
     consistent copy while the app keeps writing; other files are copied as they are.
  2. Check the snapshot before archiving: every database passes integrity_check, the three index stores agree,
     and the audit chain is intact. A torn copy of the index (an upload finished mid-copy) is retried twice.
  3. Write manifest.json: every file's size and SHA-256, the account and chunk counts, and five sample chunks that
     scripts/restore.py searches for after a restore.
  4. Encrypt with age to a public key (BACKUP_AGE_RECIPIENT). The archive is streamed into age, so no unencrypted
     archive is ever written; only the snapshot folder exists unencrypted for a moment, next to the output, and it is
     deleted at the end. The private key is needed only to restore and must not be kept with the backups.
  5. Keep 7 daily, 4 weekly and 6 monthly backups (newest of each day, ISO week and month); delete the rest.
  6. Copy to an rclone remote (BACKUP_REMOTE, e.g. gdrive:ATTS-RAG-backups) and apply the same retention there.
  7. Record the result in logs/backup_status.json (read by /health/details) and send it to a push monitor
     (BACKUP_HEARTBEAT_URL, e.g. an Uptime Kuma push URL): status=up on success, status=down on failure.

Settings (environment or .env in --root; command-line options win):
  BACKUP_DIR              output folder (default: <root>/backups)
  BACKUP_AGE_RECIPIENT    age public key(s), comma-separated (age1...). Required unless --allow-unencrypted.
  BACKUP_REMOTE           rclone destination; empty = no off-site copy
  BACKUP_HEARTBEAT_URL    push monitor URL; empty = none
  BACKUP_KEEP             daily,weekly,monthly counts (default 7,4,6)
  AGE_BIN, RCLONE_BIN     paths of the age and rclone programs (default: found on PATH)
Exit code 0 on success, 1 on failure (the status file and the push monitor say why).
"""
import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

SOURCES = ("data", "chroma_db", "logs")
ARCHIVE_PREFIX = "atts-"
ARCHIVE_SUFFIX = ".tar.gz.age"
PLAIN_SUFFIX = ".tar.gz"
STAMP_FORMAT = "%Y%m%d-%H%M%S"
STATUS_FILE = os.path.join("logs", "backup_status.json")
SQLITE_HEADER = b"SQLite format 3\x00"
SKIP_SUFFIXES = ("-wal", "-shm", "-journal")   # part of the SQLite backup copy, not copied separately
SKIP_PREFIXES = (".write-check-",)              # health-check probe files
SAMPLE_CHUNKS = 5


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def load_env_file(root: str) -> None:
    """Reads <root>/.env into os.environ without overriding variables that are already set."""
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for key, value in dotenv_values(path).items():
        if value is not None:
            os.environ.setdefault(key, value)


def find_program(explicit: Optional[str], env_name: str, name: str) -> Optional[str]:
    candidate = explicit or os.getenv(env_name)
    if candidate:
        return candidate if os.path.exists(candidate) or shutil.which(candidate) else None
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def is_sqlite(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(16) == SQLITE_HEADER
    except OSError:
        return False


def copy_sqlite(src: str, dst: str) -> None:
    """A consistent copy of a live SQLite database (SQLite's online backup)."""
    source = sqlite3.connect(f"file:{os.path.abspath(src)}?mode=ro", uri=True, timeout=30)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def snapshot_folder(src_root: str, dst_root: str) -> int:
    """Copies one folder; returns the number of files. Missing folders give an empty one."""
    os.makedirs(dst_root, exist_ok=True)
    count = 0
    if not os.path.isdir(src_root):
        return 0
    for folder, dirs, files in os.walk(src_root):
        rel = os.path.relpath(folder, src_root)
        out_dir = os.path.join(dst_root, rel) if rel != "." else dst_root
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            if name.endswith(SKIP_SUFFIXES) or name.startswith(SKIP_PREFIXES):
                continue
            src, dst = os.path.join(folder, name), os.path.join(out_dir, name)
            try:
                if is_sqlite(src):
                    copy_sqlite(src, dst)
                else:
                    shutil.copy2(src, dst)
                count += 1
            except FileNotFoundError:
                continue   # removed while copying (a rotated log, a finished upload's temp file)
    return count


def take_snapshot(root: str, staging: str, only: Sequence[str] = SOURCES) -> None:
    for name in only:
        target = os.path.join(staging, name)
        if os.path.exists(target):
            shutil.rmtree(target)
        snapshot_folder(os.path.join(root, name), target)
    env = os.path.join(root, ".env")
    if os.path.exists(env) and "data" in only:
        shutil.copy2(env, os.path.join(staging, "env.backup"))


# ---------------------------------------------------------------------------
# Checks (shared with scripts/restore.py)
# ---------------------------------------------------------------------------

def sqlite_files(folder: str) -> List[str]:
    out = []
    for base, _, files in os.walk(folder):
        out.extend(os.path.join(base, n) for n in files if is_sqlite(os.path.join(base, n)))
    return sorted(out)


def integrity_problems(folder: str) -> List[str]:
    problems = []
    for path in sqlite_files(folder):
        try:
            con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
            try:
                result = con.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error as exc:
            result = type(exc).__name__
        if result != "ok":
            problems.append(f"{os.path.relpath(path, folder)}: {result}")
    return problems


def account_count(folder: str) -> Optional[int]:
    path = os.path.join(folder, "data", "users.db")
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)
    try:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    except sqlite3.Error:
        return None
    finally:
        con.close()


def audit_chain(folder: str) -> Dict[str, Any]:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from app import audit
    result = audit.verify_all(os.path.join(folder, "logs", "auth_audit.jsonl"))
    if result.get("broken"):
        result["broken"]["file"] = os.path.relpath(result["broken"]["file"], folder)
    return result


def index_report(folder: str, samples: Optional[List[Dict[str, str]]] = None, k: int = 5) -> Dict[str, Any]:
    """Index consistency (and, with samples, whether each sample chunk is found again by search), computed in a
    child process: Chroma keeps its files open until the process ends, and Windows cannot delete open files."""
    chroma = os.path.join(folder, "chroma_db")
    if not os.path.isdir(chroma) or not os.listdir(chroma):
        return {"consistent": True, "chunks": 0, "documents": 0, "samples": [], "found": None, "empty": True}
    request = json.dumps({"persist": chroma, "samples": samples, "k": k})
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--_index-report"], input=request,
                          capture_output=True, text=True, encoding="utf-8",
                          cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError("index check failed: " + " | ".join(tail))
    return json.loads(lines[-1])


def _index_report_child() -> None:
    """Runs in the child process started by index_report()."""
    request = json.loads(sys.stdin.read())
    sys.path.insert(0, os.getcwd())
    import logging
    logging.disable(logging.WARNING)
    from loguru import logger
    logger.remove()
    from app.rag.vectorstore import VectorStoreManager
    vsm = VectorStoreManager(persist_directory=request["persist"])
    report = vsm.consistency_report()
    out: Dict[str, Any] = {"consistent": report["consistent"], "chunks": report["chroma_chunks"],
                           "documents": len(vsm.registry), "empty": False}
    if request.get("samples") is None:   # pick samples for the manifest
        cache = [c for c in vsm.chunks_cache if c["metadata"].get("chunk_uid") and len(c["content"]) > 80]
        step = max(1, len(cache) // SAMPLE_CHUNKS)
        out["samples"] = [{"chunk_uid": c["metadata"]["chunk_uid"],
                           "tenant_id": c["metadata"].get("tenant_id", "default_tenant"),
                           "query": c["content"][:300]} for c in cache[::step][:SAMPLE_CHUNKS]]
    else:   # look for the manifest's samples
        found = []
        for s in request["samples"]:
            hits = vsm.search(s["query"], k=request.get("k", 5), filter={"tenant_id": s["tenant_id"]})
            found.append(any(h["metadata"].get("chunk_uid") == s["chunk_uid"] for h in hits))
        out["found"] = found
    print(json.dumps(out))


def file_hashes(folder: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for base, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(base, name)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            if rel == "manifest.json":
                continue
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            out[rel] = {"size": os.path.getsize(path), "sha256": h.hexdigest()}
    return dict(sorted(out.items()))


def git_commit(root: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10)
        return proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# Archive and encryption
# ---------------------------------------------------------------------------

def write_archive(staging: str, target: str, age_bin: Optional[str], recipients: List[str]) -> None:
    """tar.gz of the staging folder, streamed through age when recipients are given."""
    names = sorted(os.listdir(staging))
    if not recipients:
        with tarfile.open(target, "w:gz") as tar:
            for name in names:
                tar.add(os.path.join(staging, name), arcname=name)
        return
    cmd = [age_bin]
    for r in recipients:
        cmd += ["-r", r]
    cmd += ["-o", target]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|gz") as tar:
            for name in names:
                tar.add(os.path.join(staging, name), arcname=name)
        proc.stdin.close()
    except BaseException:
        proc.kill()
        raise
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"age failed: {err.strip()[:300]}")


def check_age_file(path: str) -> None:
    with open(path, "rb") as f:
        head = f.read(64)
    if not head.startswith(b"age-encryption.org/v1"):
        raise RuntimeError(f"{os.path.basename(path)} is not an age file")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def parse_stamp(name: str) -> Optional[datetime]:
    base = os.path.basename(name)
    if not base.startswith(ARCHIVE_PREFIX):
        return None
    for suffix in (ARCHIVE_SUFFIX, PLAIN_SUFFIX):
        if base.endswith(suffix):
            try:
                return datetime.strptime(base[len(ARCHIVE_PREFIX):-len(suffix)], STAMP_FORMAT)
            except ValueError:
                return None
    return None


def select_keep(names: Iterable[str], now: datetime, daily: int = 7, weekly: int = 4, monthly: int = 6) -> Set[str]:
    """The backups to keep: the newest of each of the last `daily` days, `weekly` ISO weeks and `monthly` months
    (counted back from `now`, including the current one). Names that are not backups are always kept."""
    dated = sorted(((parse_stamp(n), n) for n in names), key=lambda x: (x[0] is None, x[0] or datetime.min),
                   reverse=True)
    keep = {n for t, n in dated if t is None}
    newest_per: Dict[Any, str] = {}
    for t, n in dated:
        if t is None:
            continue
        for key in (("d", t.date()), ("w", tuple(t.isocalendar()[:2])), ("m", (t.year, t.month))):
            newest_per.setdefault(key, n)   # dated is newest first
    today = now.date()
    days = {today - timedelta(days=i) for i in range(daily)}
    weeks = {tuple((today - timedelta(weeks=i)).isocalendar()[:2]) for i in range(weekly)}
    months, y, m = set(), today.year, today.month
    for _ in range(monthly):
        months.add((y, m))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    for (kind, key), n in newest_per.items():
        if (kind == "d" and key in days) or (kind == "w" and key in weeks) or (kind == "m" and key in months):
            keep.add(n)
    return keep


def prune_local(folder: str, now: datetime, keep_counts: Sequence[int]) -> List[str]:
    names = [n for n in os.listdir(folder) if parse_stamp(n)]
    keep = select_keep(names, now, *keep_counts)
    removed = []
    for n in names:
        if n not in keep:
            os.remove(os.path.join(folder, n))
            removed.append(n)
    return removed


def rclone(rclone_bin: str, *args: str) -> str:
    proc = subprocess.run([rclone_bin, *args], capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"rclone {args[0]} failed: {proc.stderr.strip()[-300:]}")
    return proc.stdout


def copy_offsite(rclone_bin: str, remote: str, path: str, now: datetime, keep_counts: Sequence[int]) -> List[str]:
    remote = remote.rstrip("/")
    rclone(rclone_bin, "copyto", path, f"{remote}/{os.path.basename(path)}")
    listed = rclone(rclone_bin, "lsf", "--files-only", remote).split()
    if os.path.basename(path) not in listed:
        raise RuntimeError("the off-site copy is not listed after the upload")
    names = [n for n in listed if parse_stamp(n)]
    keep = select_keep(names, now, *keep_counts)
    removed = []
    for n in names:
        if n not in keep:
            rclone(rclone_bin, "deletefile", f"{remote}/{n}")
            removed.append(n)
    return removed


# ---------------------------------------------------------------------------
# Status and heartbeat
# ---------------------------------------------------------------------------

def write_status(root: str, **fields: Any) -> None:
    path = os.getenv("BACKUP_STATUS_FILE") or os.path.join(root, STATUS_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            status = json.load(f)
    except (OSError, ValueError):
        status = {}
    status.update(fields)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def heartbeat(url: Optional[str], ok: bool, message: str, seconds: float) -> None:
    if not url:
        return
    query = urllib.parse.urlencode({"status": "up" if ok else "down", "msg": message[:200],
                                    "ping": int(seconds * 1000)})
    try:
        urllib.request.urlopen(f"{url}{'&' if '?' in url else '?'}{query}", timeout=15).read()
    except Exception as exc:   # a monitor that cannot be reached must not fail the backup itself
        print(f"warning: could not reach the push monitor: {type(exc).__name__}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="the ATTS-RAG folder holding data/, chroma_db/, logs/ and .env")
    ap.add_argument("--out", help="output folder (BACKUP_DIR)")
    ap.add_argument("--recipient", action="append", help="age public key (repeatable; BACKUP_AGE_RECIPIENT)")
    ap.add_argument("--remote", help="rclone destination (BACKUP_REMOTE)")
    ap.add_argument("--heartbeat", help="push monitor URL (BACKUP_HEARTBEAT_URL)")
    ap.add_argument("--keep", help="daily,weekly,monthly (BACKUP_KEEP, default 7,4,6)")
    ap.add_argument("--age", help="path of age (AGE_BIN)")
    ap.add_argument("--rclone", help="path of rclone (RCLONE_BIN)")
    ap.add_argument("--allow-unencrypted", action="store_true",
                    help="write a plain .tar.gz when no recipient is set (not for anything that leaves the machine)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    load_env_file(root)
    out = os.path.abspath(args.out or os.getenv("BACKUP_DIR") or os.path.join(root, "backups"))
    recipients = args.recipient or [r.strip() for r in os.getenv("BACKUP_AGE_RECIPIENT", "").split(",") if r.strip()]
    remote = args.remote if args.remote is not None else os.getenv("BACKUP_REMOTE", "")
    beat = args.heartbeat if args.heartbeat is not None else os.getenv("BACKUP_HEARTBEAT_URL", "")
    keep_counts = [int(x) for x in (args.keep or os.getenv("BACKUP_KEEP", "7,4,6")).split(",")]
    started = time.time()
    now = datetime.now()
    stamp = now.strftime(STAMP_FORMAT)
    staging = os.path.join(out, f".staging-{stamp}")

    try:
        if len(keep_counts) != 3 or min(keep_counts) < 1:
            raise ValueError("BACKUP_KEEP must be three positive numbers: daily,weekly,monthly")
        age_bin = None
        if recipients:
            age_bin = find_program(args.age, "AGE_BIN", "age")
            if not age_bin:
                raise RuntimeError("age was not found: install it or set AGE_BIN")
        elif not args.allow_unencrypted:
            raise RuntimeError("no BACKUP_AGE_RECIPIENT set: refusing to write an unencrypted backup "
                               "(use --allow-unencrypted only for a copy that never leaves this machine)")
        if remote and not recipients:
            raise RuntimeError("an off-site copy must be encrypted: set BACKUP_AGE_RECIPIENT")
        rclone_bin = None
        if remote:
            rclone_bin = find_program(args.rclone, "RCLONE_BIN", "rclone")
            if not rclone_bin:
                raise RuntimeError("rclone was not found: install it or set RCLONE_BIN")

        os.makedirs(out, exist_ok=True)
        take_snapshot(root, staging)
        problems = integrity_problems(staging)
        index = index_report(staging)
        for _ in range(2):   # a copy taken while an upload was being indexed can be torn; take the index again
            if index["consistent"]:
                break
            time.sleep(5)
            take_snapshot(root, staging, only=("chroma_db",))
            index = index_report(staging)
        chain = audit_chain(staging)
        if problems:
            raise RuntimeError("database check failed: " + "; ".join(problems))
        if not index["consistent"]:
            raise RuntimeError("the index stores disagree in three snapshots in a row: run scripts/check_index.py")
        if not chain["intact"]:
            raise RuntimeError(f"the audit chain is broken in {chain['broken']['file']} at line {chain['broken']['line']}")

        manifest = {
            "format": 1, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": git_commit(root), "encrypted": bool(recipients),
            "accounts": account_count(staging), "index": {k: index[k] for k in ("chunks", "documents")},
            "samples": index.get("samples", []), "audit_records": chain["records"],
            "files": file_hashes(staging),
        }
        with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        target = os.path.join(out, f"{ARCHIVE_PREFIX}{stamp}{ARCHIVE_SUFFIX if recipients else PLAIN_SUFFIX}")
        write_archive(staging, target, age_bin, recipients)
        if recipients:
            check_age_file(target)
        digest = sha256_file(target)
        removed = prune_local(out, now, keep_counts)
        removed_remote = copy_offsite(rclone_bin, remote, target, now, keep_counts) if remote else []

        seconds = time.time() - started
        size_mb = os.path.getsize(target) / 1e6
        message = (f"{os.path.basename(target)}: {size_mb:.1f} MB, {manifest['accounts']} account(s), "
                   f"{index['chunks']} chunks, {len(manifest['files'])} files"
                   + (", copied off-site" if remote else ", no off-site copy"))
        write_status(root, last_success=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     last_file=target, last_sha256=digest, last_size=os.path.getsize(target),
                     last_seconds=round(seconds, 1), offsite=bool(remote), encrypted=bool(recipients))
        heartbeat(beat, True, message, seconds)
        print(message)
        print(f"SHA-256 {digest}; {seconds:.0f} s; removed {len(removed)} old local"
              + (f" and {len(removed_remote)} old off-site" if remote else "") + " backup(s)")
        return 0
    except Exception as exc:
        seconds = time.time() - started
        error = f"{type(exc).__name__}: {exc}"
        try:
            write_status(root, last_failure=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         last_error=error[:500])
        except OSError:
            pass
        heartbeat(beat, False, error, seconds)
        print(f"BACKUP FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_index-report":
        _index_report_child()
        sys.exit(0)
    sys.exit(run())
