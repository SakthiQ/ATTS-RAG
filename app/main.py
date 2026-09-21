import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.routes import router, vsm

# Configure Logger
logger.add("logs/backend.log", rotation="10 MB", retention="10 days", level="INFO")

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
1. **Upload** a document via `POST /upload`
2. **Query** it via `POST /query`
3. Inspect the `layer1_gate`, `layer2_gate`, `layer3_gate` fields in the response

### Auth
Upload endpoints accept an optional `X-Admin-Token` header for elevated source tiers.
"""

_TAGS = [
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
app.include_router(router)

@app.get("/", tags=["System"])
async def root():
    return {"message": "ATTS-RAG Security API v1.0 — visit /docs for Swagger UI."}

@app.get("/health", tags=["System"])
async def health_check():
    """Probes every subsystem and returns a granular health status."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # --- Probe Ollama ---
    ollama_ok = False
    ollama_detail = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ollama_host}/api/tags")
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
            ollama_detail = f"{len(models)} model(s) loaded"
        else:
            ollama_detail = f"HTTP {r.status_code}"
    except Exception as exc:
        ollama_detail = str(exc)[:120]

    # --- Probe ChromaDB (VectorStore) ---
    vector_db_ok = False
    vector_db_detail = "unreachable"
    try:
        report = vsm.consistency_report()
        vector_db_ok = True
        vector_db_detail = (
            f"{report['chroma_chunks']} chunks indexed"
            + ("" if report["consistent"] else " [index inconsistency detected]")
        )
    except Exception as exc:
        vector_db_detail = str(exc)[:120]

    overall = "healthy" if (ollama_ok and vector_db_ok) else "degraded"

    return {
        "status": overall,
        "version": "1.0.0",
        "details": {
            "api": "online",
            "ollama": {"ok": ollama_ok, "detail": ollama_detail},
            "vector_db": {"ok": vector_db_ok, "detail": vector_db_detail},
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
