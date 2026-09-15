import os
import hmac
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
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

class QueryResponse(BaseModel):
    answer: str
    citations: list
    reasoning_log: list = []

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

@router.post("/upload")
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

@router.get("/documents")
async def list_documents():
    """Returns a list of all ingested documents from the registry."""
    return vsm.registry

@router.delete("/documents/{content_hash}")
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

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Processes a natural language query and returns an answer with citations."""
    try:
        response = engine.query(request.question)
        return response
    except Exception as e:
        logger.error(f"Query failed for '{request.question}': {e}")
        raise HTTPException(status_code=500, detail="Failed to process query.")
