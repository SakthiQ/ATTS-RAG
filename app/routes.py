import os
import shutil
import asyncio
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel
from loguru import logger
from .rag.loader import DocumentLoader
from .rag.chunker import DocumentChunker
from .rag.vectorstore import ACL_TIERS, DEFAULT_ACL_TIER, VectorStoreManager
from .rag.engine import RAGEngine
from .rag.ingestion_guard import IngestionGuard
from .rag.ingestion import ingest_file, slugify_document_id, DOCUMENT_ID_PATTERN
from .rag.trust_policy import SOURCE_TIERS, DEFAULT_TIER
from .auth import User, get_current_user, require_tenant_admin
from .audit import audit

router = APIRouter()

# Initialize components
loader = DocumentLoader()
chunker = DocumentChunker()
vsm = VectorStoreManager()
engine = RAGEngine(vsm=vsm)
guard = IngestionGuard()

# Ensure upload directory exists
UPLOAD_DIR = "data"
UPLOAD_SUBDIR = "uploads"   # data/uploads/<tenant>/<random id>/<filename>
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DEFAULT_TENANT_ID = "default_tenant"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    # The user and tenant come from the login token; identity fields sent by the client are ignored.
    question: str
    session_id: Optional[str] = "default_session"   # groups a conversation in logs only; never an identity
    admin_view: Optional[bool] = False   # ask for unmasked personal data; admins only

class QueryResponse(BaseModel):
    answer: str
    admin_view: bool = False
    intent: Optional[str] = None
    chart: Optional[dict] = None
    citations: list
    citation_details: Optional[list] = []
    reasoning_log: list = []
    threat_gate: Optional[dict] = None
    layer2_gate: Optional[dict] = None
    layer3_gate: Optional[dict] = None
    telemetry: Optional[dict] = None



def upload_path(tenant_id: str, filename: str) -> str:
    """Where an upload is saved: its own new folder inside the tenant's folder.

    Saving every upload as data/<filename> let two uploads with the same name (from different tenants, or at the
    same time) overwrite each other before the background indexer read them. The file keeps its name inside the
    folder, because citations show it."""
    folder = os.path.join(UPLOAD_DIR, UPLOAD_SUBDIR, tenant_id, uuid.uuid4().hex)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

class DocumentUpdate(BaseModel):
    acl_tier: int


def _visible(entry: dict, user: User) -> bool:
    """A super admin sees every tenant's documents; everyone else (tenant admins too) only their own tenant's,
    and only documents whose clearance tier is at most the user's clearance."""
    in_tenant = user.is_super_admin or entry.get("tenant_id", DEFAULT_TENANT_ID) == user.tenant_id
    return in_tenant and int(entry.get("acl_tier", DEFAULT_ACL_TIER)) <= user.effective_clearance

# Outcome of recent background ingestions, keyed by (tenant, document ID). An upload returns before the file is
# scanned and indexed, so without this a failed or still-running ingestion was invisible to the user. Kept in
# memory for the newest INGESTION_STATUS_LIMIT uploads; a restart forgets it (the registry is the lasting record).
INGESTION_STATUS_LIMIT = 1000
_ingestion_status: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
_ingestion_lock = threading.Lock()


def _set_ingestion_status(tenant_id: str, document_id: str, **fields: Any) -> None:
    with _ingestion_lock:
        key = (tenant_id, document_id)
        _ingestion_status[key] = {**_ingestion_status.get(key, {}), **fields}
        _ingestion_status.move_to_end(key)
        while len(_ingestion_status) > INGESTION_STATUS_LIMIT:
            _ingestion_status.popitem(last=False)


def process_document_background(file_path: str, filename: str, source_tier: str, document_id: str, uploaded_by: str,
                                tenant_id: str = "default_tenant", acl_tier: int = DEFAULT_ACL_TIER):
    """Worker function to process document in the background."""
    try:
        summary = ingest_file(
            file_path, vsm, guard, loader, chunker,
            source_tier=source_tier, document_id=document_id, uploaded_by=uploaded_by, tenant_id=tenant_id,
            acl_tier=acl_tier,
        )
        logger.info(f"Background Task: Ingested {filename}: {summary}")
        _set_ingestion_status(tenant_id, document_id, state=summary.get("status", "indexed"),   # indexed | duplicate | empty
                              indexed=summary.get("indexed"), flagged=summary.get("flagged"),
                              quarantined=summary.get("quarantined"), finished_at=datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Background Task Error for {filename}: {e}")
        _set_ingestion_status(tenant_id, document_id, state="failed", error=str(e)[:300],
                              finished_at=datetime.now().isoformat())

@router.post("/upload", tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_tier: str = Form(DEFAULT_TIER),
    document_id: Optional[str] = Form(None),
    tenant_id: Optional[str] = Form(None),
    acl_tier: int = Form(DEFAULT_ACL_TIER),
    user: User = Depends(get_current_user),
):
    """Handles file upload and triggers background ingestion. Requires login.

    The document joins the uploader's tenant. Uploads default to the 'unknown' source tier. Viewers cannot
    upload. Assigning any other tier or adding a new version under an existing document ID needs an admin of
    the target tenant (tenant_admin or super_admin); uploading into another tenant needs a super admin.
    `acl_tier` is the document's clearance level (1 general, 2 confidential, 3 restricted); above 1 is admin only.
    """
    if not user.can_upload:
        raise HTTPException(status_code=403, detail="Viewers cannot upload documents.")
    safe_filename = os.path.basename(file.filename or "")
    ext = os.path.splitext(safe_filename)[1].lower()

    if not safe_filename or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported or invalid filename: {file.filename}")

    if source_tier not in SOURCE_TIERS:
        raise HTTPException(status_code=400, detail=f"Unknown source tier '{source_tier}'. Allowed: {', '.join(SOURCE_TIERS)}.")
    if acl_tier not in ACL_TIERS:
        raise HTTPException(status_code=400, detail=f"acl_tier must be one of {', '.join(map(str, ACL_TIERS))}.")

    resolved_id = (document_id or "").strip().lower() or slugify_document_id(safe_filename)
    if not DOCUMENT_ID_PATTERN.match(resolved_id):
        raise HTTPException(status_code=400, detail="document_id may contain only lowercase letters, digits, '_' and '-' (max 100 characters).")

    tenant_id = (tenant_id or user.tenant_id).strip().lower()
    if not DOCUMENT_ID_PATTERN.match(tenant_id):
        raise HTTPException(status_code=400, detail="tenant_id may contain only lowercase letters, digits, '_' and '-' (max 100 characters).")

    # All permission checks happen before anything is written to disk
    is_admin = user.manages(tenant_id)
    # Without this, a user could plant documents into another tenant's knowledge base
    if tenant_id != user.tenant_id and not user.is_super_admin:
        raise HTTPException(status_code=403, detail="Only a super admin can upload a document for another tenant.")
    if source_tier != DEFAULT_TIER and not is_admin:
        raise HTTPException(status_code=403, detail="Only an admin can assign a source tier other than 'unknown'.")
    if acl_tier != DEFAULT_ACL_TIER and not is_admin:
        raise HTTPException(status_code=403, detail="Only an admin can give a document a clearance tier above 1.")

    # Without this, a user's upload named after a real document would become its "latest version".
    # Only the target tenant's IDs count: checking every tenant's would tell a user which IDs other tenants use.
    existing_ids = {entry.get("document_id") for entry in vsm.registry.values()
                    if entry.get("tenant_id", DEFAULT_TENANT_ID) == tenant_id}
    if resolved_id in existing_ids and not is_admin:
        raise HTTPException(status_code=409, detail=f"Document ID '{resolved_id}' already exists. Only an admin can upload a new version of it.")

    uploaded_by = user.username
    try:
        file_path = upload_path(tenant_id, safe_filename)
        # Save the file locally (fast)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Trigger the heavy AI work (including the poisoning scan) in the background
        _set_ingestion_status(tenant_id, resolved_id, state="processing", filename=safe_filename, acl_tier=acl_tier,
                              uploaded_by=uploaded_by, started_at=datetime.now().isoformat())
        background_tasks.add_task(process_document_background, file_path, safe_filename, source_tier, resolved_id, uploaded_by,
                                  tenant_id, acl_tier)

        return {
            "status": "processing",
            "filename": safe_filename,
            "document_id": resolved_id,
            "source_tier": source_tier,
            "tenant_id": tenant_id,
            "acl_tier": acl_tier,
            "message": f"File '{safe_filename}' uploaded successfully. Scanning and indexing started in the background."
        }

    except Exception as e:
        logger.error(f"Upload failed for '{safe_filename}': {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

@router.get("/ingestion/{document_id}", tags=["Documents"])
async def ingestion_status(document_id: str, tenant_id: Optional[str] = None, user: User = Depends(get_current_user)):
    """Whether a recent upload is still being scanned and indexed, finished (indexed, duplicate, empty) or failed.

    Only the user's own tenant is looked up (a super admin may name another with `tenant_id`), and an upload above
    the user's clearance is reported as unknown, like any other document they may not see."""
    tenant = (tenant_id or user.tenant_id).strip().lower()
    if tenant != user.tenant_id and not user.is_super_admin:
        raise HTTPException(status_code=403, detail="Only a super admin can look at another tenant's uploads.")
    with _ingestion_lock:
        status = dict(_ingestion_status.get((tenant, document_id), {}))
    if not status or int(status.get("acl_tier", DEFAULT_ACL_TIER)) > user.effective_clearance:
        raise HTTPException(status_code=404, detail=f"No recent upload with document ID '{document_id}'.")
    return {"document_id": document_id, "tenant_id": tenant, **status}

@router.get("/documents", tags=["Documents"])
async def list_documents(user: User = Depends(get_current_user)):
    """Returns the ingested documents the user may see: their own tenant's (all tenants for a super admin)."""
    return {h: e for h, e in vsm.registry.items() if _visible(e, user)}

@router.delete("/documents/{content_hash}", tags=["Documents"])
async def delete_document(content_hash: str, admin: User = Depends(require_tenant_admin)):
    """Deletes a document from the system using its hash. Needs an admin of the document's tenant."""
    entry = vsm.registry.get(content_hash)
    # Another tenant's document is reported as missing, so its existence is not revealed
    if entry is None or not admin.manages(entry.get("tenant_id", DEFAULT_TENANT_ID)):
        raise HTTPException(status_code=404, detail=f"Document {content_hash} not found.")
    try:
        vsm.delete_document(content_hash)
        audit("document_deleted", actor=admin.username, tenant=entry.get("tenant_id", DEFAULT_TENANT_ID),
              document_id=entry.get("document_id"), content_hash=content_hash)
        return {"status": "success", "message": f"Document {content_hash} deleted."}
    except Exception as e:
        logger.error(f"Delete failed for '{content_hash}': {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document.")

@router.patch("/documents/{content_hash}", tags=["Documents"])
async def update_document(content_hash: str, body: DocumentUpdate, admin: User = Depends(require_tenant_admin)):
    """Changes a document's clearance tier (1-3). Needs an admin of the document's tenant."""
    entry = vsm.registry.get(content_hash)
    if entry is None or not admin.manages(entry.get("tenant_id", DEFAULT_TENANT_ID)):
        raise HTTPException(status_code=404, detail=f"Document {content_hash} not found.")
    if body.acl_tier not in ACL_TIERS:
        raise HTTPException(status_code=400, detail=f"acl_tier must be one of {', '.join(map(str, ACL_TIERS))}.")
    before = int(entry.get("acl_tier", DEFAULT_ACL_TIER))
    try:
        updated = await asyncio.to_thread(vsm.set_acl_tier, content_hash, body.acl_tier)
    except Exception as e:
        logger.error(f"Clearance change failed for '{content_hash}': {e}")
        raise HTTPException(status_code=500, detail="Failed to change the document's clearance tier.")
    audit("document_clearance_changed", actor=admin.username, tenant=entry.get("tenant_id", DEFAULT_TENANT_ID),
          document_id=entry.get("document_id"), content_hash=content_hash, before=before, after=body.acl_tier)
    return {"content_hash": content_hash, "document_id": entry.get("document_id"), "acl_tier": body.acl_tier,
            "chunks_updated": updated}

@router.get("/documents/{content_hash}/preview", tags=["Documents"])
async def preview_document(content_hash: str, max_chars: int = 1000, user: User = Depends(get_current_user)):
    """Returns a short plaintext preview of a document's indexed content.

    Fetches the first 3 chunks from the Chroma vector store for the given
    document hash and concatenates their text up to `max_chars` characters.
    """
    # Another tenant's document, or one above the user's clearance, is reported as missing, so its existence is not revealed
    if content_hash not in vsm.registry or not _visible(vsm.registry[content_hash], user):
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
    user: User = Depends(get_current_user),
):
    """Processes a natural language query and returns an answer with citations and Layer 1-3 security metadata.

    Requires login; the query only ever reaches the user's own tenant's documents.

    Personal data in the evidence (SSN, card numbers, e-mail) is masked for everyone. An admin (tenant_admin or super_admin) who sends
    `"admin_view": true` gets it unmasked (and the access is logged). Being an admin alone does not unmask
    anything, and an admin view request from a non-admin is refused, not downgraded.
    """
    if request_data.admin_view and not user.is_admin:
        raise HTTPException(status_code=403, detail="admin_view is for admins only.")
    try:
        session_id = session_id_hdr or request_data.session_id or "default_session"
        tenant_id = user.tenant_id   # from the verified token, never from the client
        user_id = user.username      # unique and never reassigned; readable in logs and the admin audit
        client_ip = req.client.host if req.client else "127.0.0.1"

        response = await asyncio.to_thread(
            engine.query,
            question=request_data.question,
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            client_ip=client_ip,
            admin_view=bool(request_data.admin_view),
            authenticated=True,
            role=user.role,
            clearance=user.effective_clearance,
        )
        return response
    except Exception as e:
        logger.error(f"Query failed for '{request_data.question}': {e}")
        raise HTTPException(status_code=500, detail="Failed to process query.")



