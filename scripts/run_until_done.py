#!/usr/bin/env python3
"""Runs a command until it succeeds, retrying after a pause (for long evaluations that a model-server restart can interrupt).

Every LLM reply is cached on disk, so a retried evaluation resumes where the previous attempt stopped.
Usage: python scripts/run_until_done.py --done eval/results/NAME/report.md --tries 12 -- python scripts/evaluate_pipeline.py ...
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/tags", timeout=5).read(10)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--done", required=True, help="file whose existence means the run finished")
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--pause", type=int, default=90)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    for attempt in range(1, args.tries + 1):
        while not ollama_up():
            print(f"[run_until_done] waiting for the model server ...", flush=True)
            time.sleep(30)
        print(f"[run_until_done] attempt {attempt}/{args.tries}", flush=True)
        code = subprocess.call(cmd)
        if code == 0 and os.path.exists(args.done):
            print("[run_until_done] finished", flush=True)
            return 0
        print(f"[run_until_done] exit code {code}; retrying in {args.pause}s", flush=True)
        time.sleep(args.pause)
    return 1


if __name__ == "__main__":
    sys.exit(main())
