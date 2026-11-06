#!/usr/bin/env python3
"""Manage ATTS-RAG accounts from the command line (works while the backend is stopped or running).

    python scripts/manage_users.py create --username alice --tenant default_tenant
    python scripts/manage_users.py create --username root --role admin
    python scripts/manage_users.py list

The password is asked for interactively, or read from the environment variable named by --password-env
(so it never appears in the shell history). Uses the same database as the API (USER_DB_PATH, default data/users.db).
"""
import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import auth  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create an account")
    create.add_argument("--username", required=True)
    create.add_argument("--role", choices=auth.ROLES, default="user")
    create.add_argument("--tenant", default=auth.DEFAULT_TENANT)
    create.add_argument("--password-env", help="read the password from this environment variable instead of prompting")
    sub.add_parser("list", help="list accounts")
    args = ap.parse_args()

    store = auth.get_store()
    if args.command == "list":
        for user in store.list_users():
            print(f"{user.username:30s} {user.role:6s} {user.tenant_id:20s} {user.status}")
        return

    if args.password_env:
        password = os.environ.get(args.password_env)
        if not password:
            sys.exit(f"Environment variable {args.password_env} is empty.")
    else:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            sys.exit("The passwords do not match.")
    try:
        user = store.create_user(args.username, password, tenant_id=args.tenant, role=args.role)
    except auth.AuthError as exc:
        sys.exit(str(exc))
    print(f"Created {user.role} '{user.username}' in tenant '{user.tenant_id}'.")


if __name__ == "__main__":
    main()
