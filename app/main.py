import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.routes import router, vsm
from app import auth, auth_admin, monitoring

# Configure Logger. BACKEND_LOG names the file (default logs/backend.log); BACKEND_LOG=off writes no file
_backend_log = os.getenv("BACKEND_LOG", os.path.join("logs", "backend.log"))
if _backend_log.lower() != "off":
    logger.add(_backend_log, rotation="10 MB", retention="10 days", level="INFO")

# ---------------------------------------------------------------------------
# OpenAPI / Swagger metadata
# ---------------------------------------------------------------------------
_DESCRIPTION = """
## ATTS-RAG — Adaptive Threat and Trust Security for Retrieval-Augmented Generation

A **zero-trust RAG API** with three hardened security layers:

| Layer | Gate | Purpose |
|-------|------|---------|
| **1** | Threat Gate | Blocks prompt injection, jailbreaks, PII exfiltration |
| **2** | Trust Gate | Verifies source provenance and cross-tenant isolation |
| **3** | Output Gate | NLI-verified, claim-by-claim answer grounding |

### Quick Start
1. **Sign up** via `POST /auth/signup` or **log in** via `POST /auth/login` (or the *Authorize* button above)
2. **Upload** a document via `POST /upload`
3. **Query** it via `POST /query`
4. Inspect the `threat_gate`, `layer2_gate`, `layer3_gate` fields in the response

### Auth
Every endpoint except `/`, `/health` and the `/auth` sign-up / log-in endpoints needs
`Authorization: Bearer <token>`. `/health/details` and `/metrics` take the monitoring token (`MONITOR_TOKEN`) or a
super admin's token. The token fixes the user, tenant and role: identity headers sent by the
client are ignored. Admin-only actions (elevated source tiers, uploads into another tenant, new document
versions, deletion, unmasked personal data, creating accounts) need an account with role `admin`.
"""

_TAGS = [
    {
        "name": "Auth",
        "description": "Sign up, log in, log out and (admins) manage accounts.",
    },
    {
        "name": "Documents",
        "description": "Upload, list, preview and delete ingested documents.",
    },
    {
        "name": "Query",
        "description": "Run a security-gated RAG query through all three verification layers.",
    },
    {
        "name": "System",
        "description": "Health checks and API metadata.",
    },
]

app = FastAPI(
    title="ATTS-RAG Security API",
    description=_DESCRIPTION,
    version="1.0.0",
    openapi_tags=_TAGS,
    contact={
        "name": "ATTS-RAG Engineering",
        "url": "https://github.com/your-org/ATTS-RAG",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",        # Swagger UI   → http://localhost:8000/docs
    redoc_url="/redoc",      # ReDoc UI     → http://localhost:8000/redoc
    openapi_url="/openapi.json",  # raw schema  → http://localhost:8000/openapi.json
)

logger.info("Starting ATTS-RAG API...")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catcher for all unhandled exceptions."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "An unexpected error occurred on the server.",
        },
    )

# Enable CORS for the local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes
app.include_router(auth.router)
app.include_router(auth_admin.router)
app.include_router(router)
app.include_router(monitoring.build_router(vsm))


@app.on_event("startup")
def _bootstrap_admin():
    """Creates the admin from ATTS_ADMIN_USERNAME / ATTS_ADMIN_PASSWORD on first start (no-op if unset)."""
    try:
        auth.bootstrap_admin()
    except Exception as exc:
        logger.error(f"Admin bootstrap failed: {exc}")

@app.get("/", tags=["System"])
async def root():
    return {"message": "ATTS-RAG Security API v1.0 — visit /docs for Swagger UI."}

@app.get("/health", tags=["System"])
async def health_check():
    """Public status: "healthy" or "degraded" and an ok flag for the model and the index, with no details.

    Always HTTP 200 while the API answers, so Docker restarts only a backend that stops answering, never one that is
    waiting for Ollama. The reasons, and every other check, are at /health/details (MONITOR_TOKEN)."""
    ollama = await monitoring.check_ollama()
    index = monitoring.check_index(vsm)
    return {
        "status": "healthy" if (ollama["ok"] and index["ok"]) else "degraded",
        "version": "1.0.0",
        "details": {"api": "online", "ollama": {"ok": ollama["ok"]}, "vector_db": {"ok": index["ok"]}},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
