#!/usr/bin/env python3
"""Manage ATTS-RAG accounts from the command line (works while the backend is stopped or running).

    python scripts/manage_users.py create --username alice --tenant default_tenant
    python scripts/manage_users.py create --username root --role super_admin
    python scripts/manage_users.py list
    python scripts/manage_users.py set-role --username alice --role tenant_admin
    python scripts/manage_users.py set-clearance --username alice --clearance 2
    python scripts/manage_users.py disable --username alice        (enable, delete work the same way)
    python scripts/manage_users.py reset-password --username alice  (prints a one-time reset code)
    python scripts/manage_users.py audit-verify                     (checks the audit log's hash chain)

Roles: viewer, user, tenant_admin, super_admin ('admin' is accepted as the old name of super_admin).
The password is asked for interactively, or read from the environment variable named by --password-env
(so it never appears in the shell history). Uses the same database as the API (USER_DB_PATH, default data/users.db).
Every change is written to the audit log with actor "cli".
"""
import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import audit, auth  # noqa: E402

ROLE_CHOICES = auth.ROLES + tuple(auth.ROLE_ALIASES)


def _account(store: auth.UserStore, username: str) -> auth.User:
    user = store.get_by_username(username)
    if user is None:
        sys.exit(f"No account named '{username}'.")
    return user


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create an account")
    create.add_argument("--username", required=True)
    create.add_argument("--role", choices=ROLE_CHOICES, default="user")
    create.add_argument("--tenant", default=auth.DEFAULT_TENANT)
    create.add_argument("--clearance", type=int, choices=auth.CLEARANCE_LEVELS, default=1)
    create.add_argument("--password-env", help="read the password from this environment variable instead of prompting")
    sub.add_parser("list", help="list accounts")
    set_role = sub.add_parser("set-role", help="change an account's role")
    set_role.add_argument("--username", required=True)
    set_role.add_argument("--role", choices=ROLE_CHOICES, required=True)
    set_clearance = sub.add_parser("set-clearance", help="change an account's clearance (1-3)")
    set_clearance.add_argument("--username", required=True)
    set_clearance.add_argument("--clearance", type=int, choices=auth.CLEARANCE_LEVELS, required=True)
    for name, text in (("disable", "disable an account"), ("enable", "enable an account"),
                       ("delete", "delete an account"), ("reset-password", "issue a one-time password-reset code")):
        sub.add_parser(name, help=text).add_argument("--username", required=True)
    sub.add_parser("audit-verify", help="check the audit log's hash chain")
    args = ap.parse_args(argv)

    if args.command == "audit-verify":
        intact, checked, broken = audit.verify()
        print(f"{audit.log_path()}: {checked} records, " + ("chain intact." if intact else f"chain BROKEN at line {broken}."))
        sys.exit(0 if intact else 1)

    store = auth.get_store()
    if args.command == "list":
        for user in store.list_users():
            print(f"{user.username:30s} {user.role:12s} {user.tenant_id:20s} clearance {user.clearance}  {user.status}")
        return

    try:
        if args.command == "create":
            if args.password_env:
                password = os.environ.get(args.password_env)
                if not password:
                    sys.exit(f"Environment variable {args.password_env} is empty.")
            else:
                password = getpass.getpass("Password: ")
                if password != getpass.getpass("Repeat password: "):
                    sys.exit("The passwords do not match.")
            user = store.create_user(args.username, password, tenant_id=args.tenant, role=args.role,
                                     clearance=args.clearance)
            audit.audit("account_created", actor="cli", tenant=user.tenant_id, target=user.username, role=user.role)
            print(f"Created {user.role} '{user.username}' in tenant '{user.tenant_id}'.")
            return

        target = _account(store, args.username)
        if args.command == "set-role":
            role = auth.normalize_role(args.role)
            auth.check_not_last_super_admin(store, target, removing=role != "super_admin")
            store.update_user(target.id, role=role)
            audit.audit("account_updated", actor="cli", tenant=target.tenant_id, target=target.username,
                        before={"role": target.role}, after={"role": role})
            print(f"'{target.username}' is now {role}; their sessions have ended.")
        elif args.command == "set-clearance":
            store.update_user(target.id, clearance=args.clearance)
            audit.audit("account_updated", actor="cli", tenant=target.tenant_id, target=target.username,
                        before={"clearance": target.clearance}, after={"clearance": args.clearance})
            print(f"'{target.username}' now has clearance {args.clearance}; their sessions have ended.")
        elif args.command in ("disable", "enable"):
            new_status = "disabled" if args.command == "disable" else "active"
            auth.check_not_last_super_admin(store, target, removing=new_status == "disabled")
            store.update_user(target.id, account_status=new_status)
            audit.audit("account_updated", actor="cli", tenant=target.tenant_id, target=target.username,
                        before={"status": target.status}, after={"status": new_status})
            print(f"'{target.username}' is now {new_status}.")
        elif args.command == "delete":
            auth.check_not_last_super_admin(store, target, removing=True)
            store.delete_user(target.id)
            audit.audit("account_deleted", actor="cli", tenant=target.tenant_id, target=target.username, role=target.role)
            print(f"Deleted '{target.username}'.")
        elif args.command == "reset-password":
            code, lifetime = store.issue_reset_code(target.id)
            audit.audit("reset_code_issued", actor="cli", tenant=target.tenant_id, target=target.username)
            print(f"Reset code for '{target.username}' (valid {lifetime // 60} minutes, single use): {code}")
            print("They set a new password with POST /auth/password/reset, or on the app's log-in screen.")
    except auth.AuthError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
