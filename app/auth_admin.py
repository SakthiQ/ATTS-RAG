"""Account administration and self-service: password change and reset, account create/list/update/delete, audit log.

Who may do what (see the role list in app/auth.py):
  * everyone logged in     changes their own password
  * tenant_admin           manages accounts of their own tenant, gives roles up to tenant_admin, reads the
                           tenant's audit records
  * super_admin            manages every account, including super_admins, and reads the whole audit log

Guardrails: an admin gets 404 for an account they do not manage (so other tenants' usernames are not revealed);
a tenant_admin cannot change a super_admin; nobody changes their own role or status or deletes themselves
(use POST /auth/password for your own password); the last active super_admin cannot be demoted, disabled or
deleted. Every change ends the account's sessions and is written to the audit log.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from . import audit as audit_log
from .audit import audit
from . import totp
from .auth import (AuthError, User, check_not_last_super_admin, client_ip, get_current_user,
                   get_current_user_enrolling, get_store, http_error, normalize_role, normalize_tenant,
                   require_tenant_admin, token_response, validate_clearance)

router = APIRouter(prefix="/auth", tags=["Accounts"])


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    username: str
    code: str
    new_password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    tenant_id: Optional[str] = None   # default: the admin's own tenant
    role: str = "user"
    clearance: int = 1


class TwoFactorCode(BaseModel):
    code: str


class DisableTwoFactorRequest(BaseModel):
    password: str
    code: str


class CreateInviteRequest(BaseModel):
    tenant_id: Optional[str] = None   # default: the admin's own tenant
    role: str = "user"
    max_uses: int = 1
    expires_in_hours: float = 168


class UpdateUserRequest(BaseModel):
    role: Optional[str] = None
    status: Optional[str] = None
    clearance: Optional[int] = None


def _row(user: User) -> dict:
    # "clearance" is what the account can see; "assigned_clearance" what was set (admins see every tier anyway)
    return user.public() | {"status": user.status, "assigned_clearance": user.clearance, "two_factor": user.two_factor}


def managed_account(admin: User, username: str) -> User:
    """The account `admin` may administer, or 404 (also for accounts that exist in another tenant)."""
    target = get_store().get_by_username(username)
    if target is None or not admin.manages(target.tenant_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such account.")
    if target.is_super_admin and not admin.is_super_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only a super admin can change a super admin.")
    return target


def _keep_one_super_admin(target: User, removing: bool) -> None:
    try:
        check_not_last_super_admin(get_store(), target, removing)
    except AuthError as exc:
        raise http_error(exc)


# ---------------------------------------------------------------------------
# Self-service
# ---------------------------------------------------------------------------

@router.post("/password")
def change_password(body: ChangePasswordRequest, request: Request, user: User = Depends(get_current_user)):
    """Changes your own password. Every other session ends; the response carries a fresh token for this one."""
    store = get_store()
    try:
        # Throttled like a log-in; no second factor, the session already passed it
        store.authenticate(user.username, body.old_password, client_ip=client_ip(request), second_factor=False)
        updated = store.set_password(user.id, body.new_password)
    except AuthError as exc:
        audit("password_changed", actor=user.username, tenant=user.tenant_id, outcome=exc.reason)
        if exc.status_code == 401:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "The current password is wrong.")
        raise http_error(exc)
    audit("password_changed", actor=user.username, tenant=user.tenant_id)
    return token_response(updated)


@router.post("/password/reset")
def reset_password(body: ResetPasswordRequest, request: Request):
    """Sets a new password with a one-time reset code issued by an admin. No login needed."""
    store = get_store()
    try:
        user = store.reset_password_with_code(body.username, body.code, body.new_password, client_ip=client_ip(request))
    except AuthError as exc:
        known = store.get_by_username(body.username)
        audit("password_reset", actor=known.username if known else None, tenant=known.tenant_id if known else None,
              outcome=exc.reason)
        raise http_error(exc)
    audit("password_reset", actor=user.username, tenant=user.tenant_id)
    return token_response(user)


# ---------------------------------------------------------------------------
# Two-factor log-in (self-service)
# ---------------------------------------------------------------------------

@router.post("/2fa/setup")
def two_factor_setup(user: User = Depends(get_current_user_enrolling)):
    """Starts 2FA set-up: returns a new secret and the otpauth:// link for an authenticator app (scan it as a QR
    code, or type the secret). Nothing changes until POST /auth/2fa/enable confirms a code."""
    if user.totp_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Two-factor log-in is already on. Turn it off first to change phones.")
    secret = get_store().begin_totp_setup(user.id)
    return {"secret": secret, "otpauth_uri": totp.provisioning_uri(secret, user.username), "issuer": totp.ISSUER}


@router.post("/2fa/enable")
def two_factor_enable(body: TwoFactorCode, user: User = Depends(get_current_user_enrolling)):
    """Turns 2FA on with the first code from the app; returns one-time recovery codes, shown only here."""
    if user.totp_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Two-factor log-in is already on.")
    try:
        codes = get_store().enable_totp(user.id, body.code)
    except AuthError as exc:
        raise http_error(exc)
    audit("two_factor_enabled", actor=user.username, tenant=user.tenant_id)
    return {"enabled": True, "recovery_codes": codes,
            "note": "Keep these codes somewhere safe. Each one logs you in once if you lose your phone."}


@router.post("/2fa/disable")
def two_factor_disable(body: DisableTwoFactorRequest, request: Request, user: User = Depends(get_current_user)):
    """Turns your 2FA off; needs your password and a current code. Not allowed when your role requires 2FA."""
    if user.two_factor_required:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Your role requires two-factor log-in.")
    if not user.totp_enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Two-factor log-in is not on.")
    try:
        get_store().authenticate(user.username, body.password, client_ip=client_ip(request), otp=body.code)
    except AuthError as exc:
        audit("two_factor_disabled", actor=user.username, tenant=user.tenant_id, outcome=exc.reason)
        raise http_error(exc) if exc.status_code == 429 else HTTPException(
            status.HTTP_400_BAD_REQUEST, "Wrong password or code.")
    get_store().disable_totp(user.id)
    audit("two_factor_disabled", actor=user.username, tenant=user.tenant_id)
    return {"enabled": False}


# ---------------------------------------------------------------------------
# Administration
# ---------------------------------------------------------------------------

@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUserRequest, admin: User = Depends(require_tenant_admin)):
    """Creates an account. A tenant_admin creates accounts in their own tenant only, with roles up to tenant_admin."""
    try:
        tenant = normalize_tenant(body.tenant_id or admin.tenant_id)
        role = normalize_role(body.role)
        clearance = validate_clearance(body.clearance)
    except AuthError as exc:
        raise http_error(exc)
    if not admin.manages(tenant):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only create accounts in your own tenant.")
    if not admin.can_assign(role):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only a super admin can create a super admin.")
    try:
        user = get_store().create_user(body.username, body.password, tenant_id=tenant, role=role, clearance=clearance)
    except AuthError as exc:
        raise http_error(exc)
    audit("account_created", actor=admin.username, tenant=user.tenant_id, target=user.username, role=user.role,
          clearance=user.clearance)
    return user.public()


@router.get("/users")
def list_users(admin: User = Depends(require_tenant_admin)):
    """Accounts you manage (no password data): your tenant's, or every tenant's for a super admin."""
    users = get_store().list_users(None if admin.is_super_admin else admin.tenant_id)
    return [_row(u) for u in users]


@router.patch("/users/{username}")
def update_user(username: str, body: UpdateUserRequest, admin: User = Depends(require_tenant_admin)):
    """Changes an account's role, status (active, disabled) and/or clearance (1-3). Its sessions end at once."""
    target = managed_account(admin, username)
    if target.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot change your own role or status.")
    try:
        role = normalize_role(body.role) if body.role is not None else None
    except AuthError as exc:
        raise http_error(exc)
    if role is not None and not admin.can_assign(role):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only a super admin can make a super admin.")
    _keep_one_super_admin(target, removing=(role not in (None, "super_admin")) or body.status == "disabled")
    try:
        updated = get_store().update_user(target.id, role=role, account_status=body.status, clearance=body.clearance)
    except AuthError as exc:
        raise http_error(exc)
    changes = {k: v for k, v in (("role", role), ("status", body.status), ("clearance", body.clearance)) if v is not None}
    audit("account_updated", actor=admin.username, tenant=target.tenant_id, target=target.username,
          before={"role": target.role, "status": target.status, "clearance": target.clearance}, after=changes)
    return _row(updated)


@router.delete("/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(username: str, admin: User = Depends(require_tenant_admin)):
    """Deletes an account. Its documents stay (they belong to the tenant)."""
    target = managed_account(admin, username)
    if target.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot delete your own account.")
    _keep_one_super_admin(target, removing=True)
    get_store().delete_user(target.id)
    audit("account_deleted", actor=admin.username, tenant=target.tenant_id, target=target.username, role=target.role)


@router.delete("/users/{username}/2fa", status_code=status.HTTP_204_NO_CONTENT)
def reset_two_factor(username: str, admin: User = Depends(require_tenant_admin)):
    """Turns off an account's 2FA (lost phone, no recovery codes) and ends its sessions. The owner sets it up
    again at the next log-in if their role requires it."""
    target = managed_account(admin, username)
    if target.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Use POST /auth/2fa/disable for your own account.")
    get_store().disable_totp(target.id, end_sessions=True)
    audit("two_factor_reset", actor=admin.username, tenant=target.tenant_id, target=target.username)


@router.post("/users/{username}/reset")
def issue_reset_code(username: str, admin: User = Depends(require_tenant_admin)):
    """Issues a one-time password-reset code for the account; give it to its owner over a trusted channel.

    The owner sets a new password with POST /auth/password/reset. The code is shown only in this response."""
    target = managed_account(admin, username)
    if target.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Use POST /auth/password to change your own password.")
    code, lifetime = get_store().issue_reset_code(target.id)
    audit("reset_code_issued", actor=admin.username, tenant=target.tenant_id, target=target.username)
    return {"username": target.username, "reset_code": code, "expires_in": lifetime}


@router.post("/invites", status_code=status.HTTP_201_CREATED)
def create_invite(body: CreateInviteRequest, admin: User = Depends(require_tenant_admin)):
    """Creates an invite code (role viewer or user) for your tenant, or any tenant for a super admin.

    The code is shown only in this response; it expires after `expires_in_hours` and works `max_uses` times."""
    try:
        tenant = normalize_tenant(body.tenant_id or admin.tenant_id)
    except AuthError as exc:
        raise http_error(exc)
    if not admin.manages(tenant):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only create invites for your own tenant.")
    try:
        invite = get_store().create_invite(tenant, role=body.role, max_uses=body.max_uses,
                                           expires_in_hours=body.expires_in_hours, created_by=admin.username)
    except AuthError as exc:
        raise http_error(exc)
    audit("invite_created", actor=admin.username, tenant=tenant, invite_id=invite["id"], role=invite["role"],
          max_uses=invite["max_uses"], expires_at=invite["expires_at"])
    return invite


@router.get("/invites")
def list_invites(admin: User = Depends(require_tenant_admin)):
    """Invites of your tenant (every tenant for a super admin), without their codes."""
    return get_store().list_invites(None if admin.is_super_admin else admin.tenant_id)


@router.delete("/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_invite(invite_id: str, admin: User = Depends(require_tenant_admin)):
    """Revokes an invite so its code stops working."""
    invite = get_store().get_invite(invite_id)
    if invite is None or not admin.manages(invite["tenant_id"]):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such invite.")
    get_store().revoke_invite(invite_id)
    audit("invite_revoked", actor=admin.username, tenant=invite["tenant_id"], invite_id=invite_id)


@router.get("/audit")
def read_audit(limit: int = 200, event: Optional[str] = None, admin: User = Depends(require_tenant_admin)):
    """Newest audit records (oldest first): your tenant's, or all of them for a super admin, plus whether the
    hash chain of the whole file is intact."""
    records = audit_log.read(tenant=None if admin.is_super_admin else admin.tenant_id, event=event,
                             limit=max(1, min(limit, 2000)))
    intact, checked, broken_at = audit_log.verify()
    return {"records": records, "chain_intact": intact, "records_checked": checked, "first_broken_line": broken_at}
