# Security policy

ATTS-RAG is a security research project: its three gates are meant to stop prompt injection, poisoned documents and
unsupported answers. Reports of ways around them are welcome.

## Reporting a vulnerability

Please **do not open a public issue** for a vulnerability. Report it privately through GitHub:
**Security** tab of this repository → **Report a vulnerability**. Include what you did, what happened, and what you
expected; a minimal input that reproduces it is the most useful thing you can send.

You should get a first reply within a week. When the fix is published, the reporter is credited unless they ask
otherwise.

## What counts

- Getting a question, document or answer past a gate that should have stopped it, beyond the known limits below.
- Reading another tenant's documents, or documents above your clearance.
- Getting around log-in, two-factor log-in, roles, rate limits or the audit log.
- Secrets or personal data leaking into answers, logs or the Docker image.

## Known limits (not new reports)

These are measured and documented, and are being worked on:

- A patient attacker who rewrites an attack many times finds an unflagged version of most attacks within
  20 tries (`docs/Adaptive_Suspicion.md`).
- Search and answer checking are tuned for English.
- Conflicting facts are detected by wording, not meaning.
- Suspicion scores are kept in memory, so a restart resets them, and only one backend instance is supported.

## Supported versions

Only the latest commit on `main` receives fixes.
