import os
import pickle
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from .embedder import DocumentEmbedder
from .trust_policy import SOURCE_TIERS, DEFAULT_TIER

DEFAULT_TENANT = "default_tenant"
# Document clearance levels (acl_tier): 1 general, 2 confidential, 3 restricted. A user sees a document only if
# their clearance is at least its tier (app/auth.py gives each account a clearance; admins have the highest).
ACL_TIERS = (1, 2, 3)
DEFAULT_ACL_TIER = 1
_METADATA_DEFAULTS = {"tenant_id": DEFAULT_TENANT, "acl_tier": DEFAULT_ACL_TIER}


def metadata_matches(meta: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    """Evaluates a Chroma-style metadata filter on one chunk's metadata, for the keyword (BM25) index.

    Supports equality, $and / $or, and $eq $ne $lt $lte $gt $gte $in $nin. A missing tenant or tier counts as
    its default, as the start-up migration would have set it."""
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(metadata_matches(meta, f) for f in cond):
                return False
            continue
        if key == "$or":
            if not any(metadata_matches(meta, f) for f in cond):
                return False
            continue
        value = meta.get(key, _METADATA_DEFAULTS.get(key))
        if not isinstance(cond, dict):
            if value != cond:
                return False
            continue
        for op, ref in cond.items():
            if op == "$eq":
                ok = value == ref
            elif op == "$ne":
                ok = value != ref
            elif op == "$in":
                ok = value in ref
            elif op == "$nin":
                ok = value not in ref
            elif value is None:
                ok = False
            elif op == "$lt":
                ok = value < ref
            elif op == "$lte":
                ok = value <= ref
            elif op == "$gt":
                ok = value > ref
            elif op == "$gte":
                ok = value >= ref
            else:
                raise ValueError(f"Unsupported filter operator {op}")
            if not ok:
                return False
    return True


class VectorStoreManager:
    """Manages Hybrid Search (Vector + BM25) with Atomic Transactions, Hashing, and Trust Metadata."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.registry_path = os.path.join(persist_directory, "document_registry.json")
        self.bm25_path = os.path.join(persist_directory, "bm25_index.pkl")
        self.embedder = DocumentEmbedder()
        self.strategy_version = "v1.2" # Current chunking/embedding/scanning strategy

        # 1. Initialize Vector Store (Chroma)
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedder.client,
            collection_name="document_collection"
        )

        # 2. Initialize Hashing & BM25 members
        self.registry = self._load_registry()
        self.bm25 = None
        self.chunks_cache = []
        self.tokenized_corpus = []
        self._load_bm25()

        # 2b. Documents indexed before tenants (or clearance tiers) were recorded get the defaults
        self._tag_untagged_chunks()

        # 3. Warn if the registry, BM25 and Chroma disagree (e.g. stale chunks from older code)
        report = self.consistency_report()
        if not report["consistent"]:
            logger.warning(f"Index stores disagree; rebuild with 'python scripts/reingest_corpus.py --rebuild'. {report}")

    def _tag_untagged_chunks(self) -> int:
        """One-time, idempotent migration: chunks and registry entries without a tenant get the default tenant,
        and those without a clearance tier get tier 1.

        Without it a tenant or tier filter would hide every document indexed before these were stored (Chroma
        filters drop chunks that lack the key). Tier 1 is what every existing document effectively had."""
        def missing(meta):
            return [k for k in _METADATA_DEFAULTS if (meta or {}).get(k) in (None, "")]

        changed = 0
        try:
            got = self.vector_store.get(include=["metadatas"])
            todo = [(i, m) for i, m in zip(got["ids"], got["metadatas"]) if missing(m)]
            if todo:
                metas = [{**dict(m or {}), **{k: _METADATA_DEFAULTS[k] for k in missing(m)}} for _, m in todo]
                self.vector_store._collection.update(ids=[i for i, _ in todo], metadatas=metas)
                changed += len(todo)
        except Exception as e:
            logger.warning(f"Could not tag existing vectors with a tenant and tier: {e}")
        untagged_bm25 = [c for c in self.chunks_cache if missing(c.get("metadata"))]
        for c in untagged_bm25:
            meta = c.setdefault("metadata", {})
            for k in missing(meta):
                meta[k] = _METADATA_DEFAULTS[k]
        registry_changed = 0
        for entry in self.registry.values():
            for k, v in _METADATA_DEFAULTS.items():
                if k not in entry:
                    entry[k] = v
                    registry_changed += 1
        if untagged_bm25:
            self._save_bm25()
        if changed or untagged_bm25 or registry_changed:
            self._save_registry()
            logger.info(f"Tagged {changed} vectors and {len(untagged_bm25)} keyword-index chunks with the default "
                        f"tenant / clearance tier.")
        return changed + len(untagged_bm25)

    def consistency_report(self) -> Dict[str, Any]:
        """Compares chunk ids across the registry, the BM25 cache and Chroma. All three must match."""
        registry_ids = {i for entry in self.registry.values() for i in entry.get("ids", [])}
        bm25_ids = [c.get("id") for c in self.chunks_cache]
        bm25_id_set = {i for i in bm25_ids if i}
        chroma_ids = set(self.vector_store.get(include=[])["ids"])
        problems = {
            "bm25_without_id": sum(1 for i in bm25_ids if not i),
            "bm25_duplicate_ids": len([i for i in bm25_ids if i]) - len(bm25_id_set),
            "bm25_not_in_registry": len(bm25_id_set - registry_ids),
            "chroma_not_in_registry": len(chroma_ids - registry_ids),
            "registry_missing_from_bm25": len(registry_ids - bm25_id_set),
            "registry_missing_from_chroma": len(registry_ids - chroma_ids),
        }
        return {
            "consistent": not any(problems.values()),
            "registry_chunks": len(registry_ids),
            "bm25_chunks": len(bm25_ids),
            "chroma_chunks": len(chroma_ids),
            **problems,
        }

    def _calculate_hash(self, text: str) -> str:
        """Generates SHA-256 hash of the content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def document_hash(self, chunks: List[Dict[str, Any]], tenant_id: str = DEFAULT_TENANT) -> str:
        """Registry key of a document: identical text in the same tenant always maps to the same key.

        The key includes the tenant (default-tenant keys are the plain content hash, as before), so the same file
        uploaded by two tenants is two documents: otherwise the second tenant's upload would be skipped as a
        duplicate of a document it cannot see."""
        text = "".join(c["content"] for c in chunks)
        if tenant_id == DEFAULT_TENANT:
            return self._calculate_hash(text)
        return self._calculate_hash(f"tenant:{tenant_id}|{text}")

    def _load_registry(self) -> Dict[str, Any]:
        """Loads the document registry from disk."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        """Saves the document registry to disk."""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=4)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _load_bm25(self):
        """Loads BM25 index and tokenized corpus from disk."""
        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data["index"]
                    self.chunks_cache = data["chunks"]
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                logger.info(f"Loaded BM25 index with {len(self.chunks_cache)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.chunks_cache = []
                self.tokenized_corpus = []

    def _save_bm25(self):
        """Saves BM25 index and tokenized corpus to disk."""
        try:
            with open(self.bm25_path, "wb") as f:
                pickle.dump({
                    "index": self.bm25,
                    "chunks": self.chunks_cache,
                    "tokenized_corpus": self.tokenized_corpus
                }, f)
            logger.debug("BM25 index saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: Optional[str] = None,
                   source_tier: str = DEFAULT_TIER, uploaded_by: str = "anonymous",
                   tenant_id: str = DEFAULT_TENANT, acl_tier: int = DEFAULT_ACL_TIER) -> Dict[str, Any]:
        """Indexes one document's chunks atomically across Chroma, BM25 and the registry.

        `tenant_id` says whose document this is; Layer 2 only lets a query see documents of its own tenant (or "global").
        `acl_tier` is its clearance level (1-3): only users with at least that clearance retrieve it.

        Chunks the ingestion scan marked 'quarantined' are recorded in the registry for review
        but never indexed. Every indexed chunk carries trust metadata for Layer 2.
        """
        if not chunks:
            return {"status": "empty"}
        if source_tier not in SOURCE_TIERS:
            raise ValueError(f"Unknown source tier: {source_tier}")
        if acl_tier not in ACL_TIERS:
            raise ValueError(f"Unknown clearance tier: {acl_tier}")

        # 1. Calculate Content Hash for the entire document (all chunks belong to the same document)
        source_name = chunks[0]["metadata"].get("source", "unknown")
        content_hash = self.document_hash(chunks, tenant_id)

        # 2. Check Registry for Duplicates
        if content_hash in self.registry:
            logger.info(f"Document '{source_name}' (Hash: {content_hash[:8]}) already exists. Skipping.")
            return {"status": "duplicate", "content_hash": content_hash, "filename": source_name}

        # 3. Versioning: a new upload under an existing document ID of the same tenant becomes its next version
        #    (document IDs are per tenant: another tenant's document of the same name is a different document)
        document_id = document_id or source_name
        version = 1 + max(
            (entry.get("version", 1) for entry in self.registry.values()
             if entry.get("document_id") == document_id and entry.get("tenant_id", DEFAULT_TENANT) == tenant_id),
            default=0,
        )
        source_score = SOURCE_TIERS[source_tier]

        # 4. Trust metadata on every chunk (Chroma accepts only scalar metadata values)
        ids = [f"{content_hash}_{i}" for i in range(len(chunks))]
        for chunk, chunk_uid in zip(chunks, ids):
            chunk["metadata"].update({
                "chunk_uid": chunk_uid,
                "document_id": document_id,
                "version": version,
                "source_tier": source_tier,
                "source_score": source_score,
                "tenant_id": tenant_id,
                "acl_tier": acl_tier,
            })

        indexed = [(c, uid) for c, uid in zip(chunks, ids) if c["metadata"].get("scan_status") != "quarantined"]
        quarantined = [(c, uid) for c, uid in zip(chunks, ids) if c["metadata"].get("scan_status") == "quarantined"]
        flagged_count = sum(1 for c, _ in indexed if c["metadata"].get("scan_status") == "flagged")
        texts = [c["content"] for c, _ in indexed]
        metadatas = [c["metadata"] for c, _ in indexed]
        index_ids = [uid for _, uid in indexed]

        # 5. Attempt Atomic Ingestion
        try:
            # PHASE 1: Vector Ingestion (Heavy AI Work)
            if index_ids:
                self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=index_ids)

            try:
                # PHASE 2: Logic Ingestion (BM25 & Registry)
                # Tag each cached chunk with its Chroma id so deletes can target
                # exact chunks instead of matching on filename (which is not unique).
                for chunk, chunk_uid in indexed:
                    chunk["id"] = chunk_uid
                new_tokens = [self._tokenize(t) for t in texts]

                # Update memory
                self.chunks_cache.extend(c for c, _ in indexed)
                self.tokenized_corpus.extend(new_tokens)

                # Rebuild and Persist
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                self._save_bm25()

                self.registry[content_hash] = {
                    "filename": source_name,
                    "document_id": document_id,
                    "version": version,
                    "source_tier": source_tier,
                    "source_score": source_score,
                    "uploaded_by": uploaded_by,
                    "tenant_id": tenant_id,
                    "acl_tier": acl_tier,
                    "chunk_count": len(index_ids),
                    "total_chunks": len(chunks),
                    "flagged_count": flagged_count,
                    "quarantined": [
                        {
                            "chunk_uid": uid,
                            "anomaly_score": c["metadata"].get("anomaly_score"),
                            "notes": c["metadata"].get("scan_notes", ""),
                            "preview": c["content"][:200],
                        }
                        for c, uid in quarantined
                    ],
                    "ingested_at": datetime.now().isoformat(),
                    "strategy_version": self.strategy_version,
                    "ids": index_ids,
                    # Integrity reference for Layer 2, kept apart from the Chroma metadata it verifies
                    "chunk_hashes": {uid: self._calculate_hash(c["content"]) for c, uid in indexed},
                }
                self._save_registry()
                logger.info(
                    f"Atomic success: '{source_name}' registered as {document_id} v{version} "
                    f"({len(index_ids)} indexed, {flagged_count} flagged, {len(quarantined)} quarantined)."
                )

            except Exception as e:
                # PHASE 3: ROLLBACK (Chroma partial sync)
                logger.error(f"Logic failure during ingestion. Rolling back Vector Store: {e}")
                if index_ids:
                    self.vector_store.delete(ids=index_ids)
                raise e

        except Exception as e:
            logger.error(f"Incomplete Ingestion for '{source_name}': {e}")
            raise e

        return {
            "status": "indexed",
            "content_hash": content_hash,
            "filename": source_name,
            "document_id": document_id,
            "version": version,
            "indexed": len(index_ids),
            "flagged": flagged_count,
            "quarantined": len(quarantined),
        }

    def set_acl_tier(self, content_hash: str, acl_tier: int) -> int:
        """Changes an indexed document's clearance tier in Chroma, the keyword index and the registry.
        Returns the number of indexed chunks updated."""
        if acl_tier not in ACL_TIERS:
            raise ValueError(f"Unknown clearance tier: {acl_tier}")
        entry = self.registry[content_hash]
        ids = list(entry.get("ids", []))
        if ids:
            got = self.vector_store.get(ids=ids, include=["metadatas"])
            metas = [{**dict(m or {}), "acl_tier": acl_tier} for m in got["metadatas"]]
            self.vector_store._collection.update(ids=got["ids"], metadatas=metas)
        id_set = set(ids)
        for chunk in self.chunks_cache:
            if chunk.get("id") in id_set:
                chunk.setdefault("metadata", {})["acl_tier"] = acl_tier
        entry["acl_tier"] = acl_tier
        self._save_bm25()
        self._save_registry()
        logger.info(f"Document {entry.get('document_id')} ({content_hash[:8]}) is now clearance tier {acl_tier}.")
        return len(ids)

    def delete_document(self, content_hash: str):
        """Removes a document from both Chroma and BM25 using its hash."""
        if content_hash not in self.registry:
            logger.error(f"Hash {content_hash} not found in registry.")
            return

        doc_data = self.registry[content_hash]
        ids_to_remove = doc_data["ids"]

        # 1. Remove from Chroma (a fully quarantined document has nothing indexed)
        if ids_to_remove:
            try:
                self.vector_store.delete(ids=ids_to_remove)
            except Exception as e:
                logger.error(f"Failed to delete from Chroma: {e}")

        # 2. Clean BM25 Cache and Corpus
        # We find indices of chunks that AREN'T in the deleted set
        ids_to_remove_set = set(ids_to_remove)
        remaining_chunks = []
        remaining_tokens = []
        for chunk, tokens in zip(self.chunks_cache, self.tokenized_corpus):
            # If chunk belongs to this specific ingestion (by id), skip it
            if chunk.get("id") in ids_to_remove_set:
                 continue
            remaining_chunks.append(chunk)
            remaining_tokens.append(tokens)

        self.chunks_cache = remaining_chunks
        self.tokenized_corpus = remaining_tokens

        # 3. Rebuild BM25 from clean corpus
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

        self._save_bm25()

        # 4. Remove from Registry
        del self.registry[content_hash]
        self._save_registry()
        logger.info(f"Document '{doc_data['filename']}' successfully purged from system.")

    def delete_document_by_id(self, document_id: str, tenant_id: Optional[str] = None) -> int:
        """Deletes all chunks of a document matching document_id and optional tenant_id."""
        matching_hashes = [
            h for h, meta in self.registry.items()
            if meta.get("document_id") == document_id and (tenant_id is None or meta.get("tenant_id", DEFAULT_TENANT) == tenant_id)
        ]
        total_removed = 0
        for h in matching_hashes:
            ids = self.registry[h].get("ids", [])
            total_removed += len(ids)
            self.delete_document(h)
        return total_removed


    def hybrid_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Combines Vector and BM25 results using Reciprocal Rank Fusion (RRF)."""
        # 1. Get Vector Results (Filtered)
        vector_results = self.vector_store.similarity_search(query, k=k*2, filter=filter)
        vector_chunks = [{"content": res.page_content, "metadata": res.metadata} for res in vector_results]

        # 2. Get BM25 Results
        # Note: rank_bm25 doesn't natively support filtering easily without custom logic.
        # We will filter the chunks_cache first if filter is provided.
        tokenized_query = self._tokenize(query)

        target_chunks = self.chunks_cache
        target_tokens = self.tokenized_corpus

        if filter:
            # The same filter Chroma applied, evaluated on the keyword index
            filtered_indices = [i for i, c in enumerate(self.chunks_cache) if metadata_matches(c["metadata"], filter)]
            target_chunks = [self.chunks_cache[i] for i in filtered_indices]
            target_tokens = [self.tokenized_corpus[i] for i in filtered_indices]

            if not target_chunks:
                # Fallback to vector results only if filter matches nothing in BM25
                return vector_chunks[:k]

            # Re-run BM25 on filtered subset
            temp_bm25 = BM25Okapi(target_tokens)
            bm25_scores = temp_bm25.get_scores(tokenized_query)
        else:
            bm25_scores = self.bm25.get_scores(tokenized_query) if self.bm25 else []

        bm25_chunks = [
            c for c, _ in sorted(zip(target_chunks, bm25_scores), key=lambda x: x[1], reverse=True)
        ][:k*2] if len(bm25_scores) > 0 and any(bm25_scores) else []

        # 3. Reciprocal Rank Fusion: each list contributes independently, so a
        # miss in one ranking (e.g. no keyword overlap) never zeroes out the other.
        scores: Dict[str, float] = {}
        content_map: Dict[str, Dict[str, Any]] = {}

        for rank, chunk in enumerate(vector_chunks):
            key = chunk["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            content_map[key] = chunk

        for rank, chunk in enumerate(bm25_chunks):
            key = chunk["content"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            content_map[key] = chunk

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        # Copies carrying the fused score (for traces); copying also keeps later stages (the reranker adds
        # rerank_score) from writing into the shared keyword-index cache
        return [{**content_map[key], "retrieval_score": round(scores[key], 6)} for key in ranked_keys[:k]]

    def search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fallback to hybrid search with filtering."""
        if self.bm25:
            return self.hybrid_search(query, k, filter)
        return self.search_vector_only(query, k, filter)

    def search_vector_only(self, query: str, k: int = 4, filter: Dict[str, Any] = None):
        results = self.vector_store.similarity_search(query, k=k, filter=filter)
        return [{"content": res.page_content, "metadata": res.metadata} for res in results]
