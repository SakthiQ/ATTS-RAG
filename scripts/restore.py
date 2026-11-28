"""Restores a backup made by scripts/backup.py, and proves that the restored copy works.

    python scripts/restore.py BACKUP --identity KEY            restore test: into a scratch folder, then checks
    python scripts/restore.py BACKUP --identity KEY --apply    real restore into --root (stop the app first)

BACKUP is an atts-*.tar.gz.age file (or a plain .tar.gz made with --allow-unencrypted). KEY is the age private key
file (AGE_IDENTITY): the only thing that can open the backups, kept away from them.

The checks, all against the manifest written at backup time:
  files        every file is present with the recorded size and SHA-256
  databases    every SQLite database passes integrity_check
  accounts     the account count matches
  index        the three index stores agree, with the recorded chunk and document counts
  search       each of the manifest's sample chunks is found again by search (top 5, its own tenant)
  audit        the log-in audit chain is intact, with the recorded number of records
It prints how long the restore took (decrypt, unpack, check). Exit code 0 only if every check passes.

--apply also needs --yes. It runs the same checks on the scratch copy first; only if they pass does it move the
current data/, chroma_db/ and logs/ aside (to <name>.before-restore-<time>, nothing is deleted) and move the
restored folders into place. .env is restored only if --root has none.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backup  # noqa: E402  (scripts/backup.py: snapshot checks shared by both tools)

SOURCES = backup.SOURCES


def unpack(archive: str, target: str, identity: Optional[str], age_bin: Optional[str]) -> None:
    os.makedirs(target, exist_ok=True)
    if archive.endswith(backup.ARCHIVE_SUFFIX):
        if not identity:
            raise RuntimeError("this backup is encrypted: pass --identity (the age private key file)")
        if not age_bin:
            raise RuntimeError("age was not found: install it or set AGE_BIN")
        proc = subprocess.Popen([age_bin, "-d", "-i", identity, archive], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
                tar.extractall(target, filter="data")
        finally:
            err = proc.stderr.read().decode("utf-8", "replace")
            code = proc.wait()
        if code != 0:
            raise RuntimeError(f"age could not decrypt the backup: {err.strip()[:300]}")
    else:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target, filter="data")


def check(folder: str) -> List[Dict[str, Any]]:
    """[{name, ok, detail}] for the restored folder."""
    results = []

    def add(name: str, ok: bool, detail: str):
        results.append({"name": name, "ok": bool(ok), "detail": detail})

    path = os.path.join(folder, "manifest.json")
    if not os.path.exists(path):
        add("manifest", False, "manifest.json is missing: not a backup made by scripts/backup.py")
        return results
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Hashes first: opening the index below lets Chroma rewrite some of its own files
    actual = backup.file_hashes(folder)
    expected = manifest["files"]
    missing = sorted(set(expected) - set(actual))
    changed = sorted(n for n in expected if n in actual and actual[n]["sha256"] != expected[n]["sha256"])
    add("files", not missing and not changed,
        f"{len(expected)} files match" if not missing and not changed
        else f"{len(missing)} missing, {len(changed)} changed (first: {(missing + changed)[0]})")

    problems = backup.integrity_problems(folder)
    add("databases", not problems, f"{len(backup.sqlite_files(folder))} database(s) pass integrity_check"
        if not problems else "; ".join(problems)[:300])

    accounts = backup.account_count(folder)
    add("accounts", accounts == manifest.get("accounts"), f"{accounts} account(s), expected {manifest.get('accounts')}")

    samples = manifest.get("samples") or []
    try:
        index = backup.index_report(folder, samples=samples)
        want = manifest.get("index", {})
        same = index["chunks"] == want.get("chunks") and index["documents"] == want.get("documents")
        add("index", index["consistent"] and same,
            f"{index['chunks']} chunks in {index['documents']} document(s), stores "
            + ("agree" if index["consistent"] else "DISAGREE")
            + ("" if same else f"; expected {want.get('chunks')} chunks in {want.get('documents')} document(s)"))
        found = index.get("found") or []
        add("search", all(found) and len(found) == len(samples),
            f"{sum(found)} of {len(samples)} sample chunks found again" if samples else "no sample chunks (empty index)")
    except RuntimeError as exc:
        add("index", False, str(exc)[:300])
        add("search", False, "not run: the index could not be opened")

    chain = backup.audit_chain(folder)
    add("audit", chain["intact"] and chain["records"] == manifest.get("audit_records"),
        f"{chain['records']} records, chain " + ("intact" if chain["intact"] else f"BROKEN at {chain['broken']}")
        + ("" if chain["records"] == manifest.get("audit_records") else f"; expected {manifest.get('audit_records')}"))
    return results


def apply(scratch: str, root: str) -> List[str]:
    stamp = datetime.now().strftime(backup.STAMP_FORMAT)
    moved = []
    for name in SOURCES:
        current = os.path.join(root, name)
        if os.path.exists(current):
            aside = f"{current}.before-restore-{stamp}"
            os.replace(current, aside)
            moved.append(aside)
        shutil.move(os.path.join(scratch, name), current)
    env = os.path.join(scratch, "env.backup")
    if os.path.exists(env) and not os.path.exists(os.path.join(root, ".env")):
        shutil.copy2(env, os.path.join(root, ".env"))
    return moved


def run(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backup", help="an atts-*.tar.gz.age (or .tar.gz) file")
    ap.add_argument("--identity", help="age private key file (AGE_IDENTITY)")
    ap.add_argument("--age", help="path of age (AGE_BIN)")
    ap.add_argument("--root", default=".", help="the ATTS-RAG folder to restore into with --apply")
    ap.add_argument("--apply", action="store_true", help="really restore into --root (after the checks pass)")
    ap.add_argument("--yes", action="store_true", help="confirm --apply")
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder after a restore test")
    args = ap.parse_args(argv)
    if args.apply and not args.yes:
        ap.error("--apply replaces data/, chroma_db/ and logs/ in --root: add --yes to confirm (stop the app first)")

    root = os.path.abspath(args.root)
    backup.load_env_file(root)
    identity = args.identity or os.getenv("AGE_IDENTITY")
    age_bin = backup.find_program(args.age, "AGE_BIN", "age")
    started = time.time()
    scratch = tempfile.mkdtemp(prefix="atts-restore-", dir=root if args.apply else None)
    kept = False   # the decrypted copy is kept only after a passed test with --keep
    try:
        unpack(os.path.abspath(args.backup), scratch, identity, age_bin)
        unpacked = time.time() - started
        results = check(scratch)
        total = time.time() - started
        for r in results:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['name']:<10} {r['detail']}")
        ok = all(r["ok"] for r in results)
        print(f"\nRestore {'test passed' if ok else 'test FAILED'}: decrypt and unpack {unpacked:.0f} s, "
              f"checks {total - unpacked:.0f} s, total {total:.0f} s.")
        if not ok:
            return 1
        if args.apply:
            moved = apply(scratch, root)
            print(f"Restored into {root}. The previous folders were kept: " + (", ".join(moved) or "none existed"))
        elif args.keep:
            kept = True
            print(f"Restored copy kept in {scratch} (unencrypted: delete it when done)")
        return 0
    except Exception as exc:
        print(f"RESTORE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if not kept:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run())
