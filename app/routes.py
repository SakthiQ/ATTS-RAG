import os
import hmac
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from loguru import logger
from .rag.loader import DocumentLoader
from .rag.chunker import DocumentChunker
from .rag.vectorstore import VectorStoreManager
from .rag.engine import RAGEngine
from .rag.ingestion_guard import IngestionGuard
from .rag.ingestion import ingest_file, slugify_document_id, DOCUMENT_ID_PATTERN
from .rag.trust_policy import SOURCE_TIERS, DEFAULT_TIER

router = APIRouter()

# Initialize components
loader = DocumentLoader()
chunker = DocumentChunker()
vsm = VectorStoreManager()
engine = RAGEngine(vsm=vsm)
guard = IngestionGuard()

# Ensure upload directory exists
UPLOAD_DIR = "data"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"
    tenant_id: Optional[str] = "default_tenant"
    user_id: Optional[str] = "default_user"

class QueryResponse(BaseModel):
    answer: str
    citations: list
    reasoning_log: list = []
    threat_gate: Optional[dict] = None
    layer2_gate: Optional[dict] = None
    layer3_gate: Optional[dict] = None
    telemetry: Optional[dict] = None



def _is_admin(token: Optional[str]) -> bool:
    """True only when ADMIN_TOKEN is configured and the supplied token matches it."""
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))

def process_document_background(file_path: str, filename: str, source_tier: str, document_id: str, uploaded_by: str):
    """Worker function to process document in the background."""
    try:
        summary = ingest_file(
            file_path, vsm, guard, loader, chunker,
            source_tier=source_tier, document_id=document_id, uploaded_by=uploaded_by,
        )
        logger.info(f"Background Task: Ingested {filename}: {summary}")
    except Exception as e:
        logger.error(f"Background Task Error for {filename}: {e}")

@router.post("/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_tier: str = Form(DEFAULT_TIER),
    document_id: Optional[str] = Form(None),
    admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    """Handles file upload and triggers background ingestion.

    Uploads default to the 'unknown' source tier. Assigning any other tier, or adding a new
    version under an existing document ID, requires a valid X-Admin-Token header.
    """
    safe_filename = os.path.basename(file.filename or "")
    ext = os.path.splitext(safe_filename)[1].lower()

    if not safe_filename or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported or invalid filename: {file.filename}")

    if source_tier not in SOURCE_TIERS:
        raise HTTPException(status_code=400, detail=f"Unknown source tier '{source_tier}'. Allowed: {', '.join(SOURCE_TIERS)}.")

    resolved_id = (document_id or "").strip().lower() or slugify_document_id(safe_filename)
    if not DOCUMENT_ID_PATTERN.match(resolved_id):
        raise HTTPException(status_code=400, detail="document_id may contain only lowercase letters, digits, '_' and '-' (max 100 characters).")

    # All permission checks happen before anything is written to disk
    is_admin = _is_admin(admin_token)
    if source_tier != DEFAULT_TIER and not is_admin:
        raise HTTPException(status_code=403, detail="Only an admin can assign a source tier other than 'unknown'. Send a valid X-Admin-Token header.")

    # Without this, an anonymous upload named after a real document would become its "latest version"
    existing_ids = {entry.get("document_id") for entry in vsm.registry.values()}
    if resolved_id in existing_ids and not is_admin:
        raise HTTPException(status_code=409, detail=f"Document ID '{resolved_id}' already exists. Only an admin can upload a new version of it.")

    uploaded_by = "admin" if is_admin else "anonymous"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Save the file locally (fast)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger the heavy AI work (including the poisoning scan) in the background
        background_tasks.add_task(process_document_background, file_path, safe_filename, source_tier, resolved_id, uploaded_by)

        return {
            "status": "processing",
            "filename": safe_filename,
            "document_id": resolved_id,
            "source_tier": source_tier,
            "message": f"File '{safe_filename}' uploaded successfully. Scanning and indexing started in the background."
        }

    except Exception as e:
        logger.error(f"Upload failed for '{safe_filename}': {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

@router.get("/documents", tags=["Documents"])
async def list_documents():
    """Returns a list of all ingested documents from the registry."""
    return vsm.registry

@router.delete("/documents/{content_hash}", tags=["Documents"])
async def delete_document(content_hash: str):
    """Deletes a document from the system using its hash."""
    if content_hash not in vsm.registry:
        raise HTTPException(status_code=404, detail=f"Document {content_hash} not found.")
    try:
        vsm.delete_document(content_hash)
        return {"status": "success", "message": f"Document {content_hash} deleted."}
    except Exception as e:
        logger.error(f"Delete failed for '{content_hash}': {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document.")

@router.get("/documents/{content_hash}/preview", tags=["Documents"])
async def preview_document(content_hash: str, max_chars: int = 1000):
    """Returns a short plaintext preview of a document's indexed content.

    Fetches the first 3 chunks from the Chroma vector store for the given
    document hash and concatenates their text up to `max_chars` characters.
    """
    if content_hash not in vsm.registry:
        raise HTTPException(status_code=404, detail=f"Document {content_hash} not found.")

    entry = vsm.registry[content_hash]
    chunk_ids: list = entry.get("ids", [])[:3]  # preview first 3 chunks only

    if not chunk_ids:
        return {"preview": "(No indexed content available for this document.)"}

    try:
        result = vsm.vector_store.get(ids=chunk_ids, include=["documents"])
        texts = result.get("documents", [])
        combined = "\n\n".join(t for t in texts if t).strip()
        return {"preview": combined[:max_chars]}
    except Exception as e:
        logger.error(f"Preview fetch failed for '{content_hash}': {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document preview.")

@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(
    request_data: QueryRequest,
    req: Request,
    session_id_hdr: Optional[str] = Header(None, alias="X-Session-ID"),
    tenant_id_hdr: Optional[str] = Header(None, alias="X-Tenant-ID"),
    user_id_hdr: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Processes a natural language query and returns an answer with citations and Layer 1-3 security metadata."""
    try:
        session_id = session_id_hdr or request_data.session_id or "default_session"
        tenant_id = tenant_id_hdr or request_data.tenant_id or "default_tenant"
        user_id = user_id_hdr or request_data.user_id or "default_user"
        client_ip = req.client.host if req.client else "127.0.0.1"

        response = engine.query(
            question=request_data.question,
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            client_ip=client_ip
        )
        return response
    except Exception as e:
        logger.error(f"Query failed for '{request_data.question}': {e}")
        raise HTTPException(status_code=500, detail="Failed to process query.")



