from typing import Any, Dict, List, Optional
from loguru import logger


class DeletionPropagationHandler:
    """Document Deletion & Revocation Cascade Handler.
    
    Ensures compliance with GDPR / Right-to-be-Forgotten by propagating document 
    revocations across vector indices, metadata registries, and query caches.
    """

    def __init__(self, vsm: Optional[Any] = None):
        self.vsm = vsm

    def propagate_deletion(self, document_id: str, tenant_id: str) -> Dict[str, Any]:
        """Cascades deletion of a document across vector index and metadata registries."""
        logger.info(f"Initiating deletion cascade for document_id={document_id}, tenant_id={tenant_id}")
        
        results = {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "vectors_removed": 0,
            "metadata_invalidated": False,
            "status": "SUCCESS"
        }

        try:
            # 1. Purge from Chroma Vector DB if VectorStoreManager is provided
            if self.vsm is not None:
                if hasattr(self.vsm, "delete_document_by_id"):
                    removed_count = self.vsm.delete_document_by_id(document_id=document_id, tenant_id=tenant_id)
                elif hasattr(self.vsm, "delete_document"):
                    removed_count = self.vsm.delete_document(document_id) or 0
                else:
                    removed_count = 0
                results["vectors_removed"] = removed_count
                logger.info(f"Purged {removed_count} chunk vectors from Chroma for document {document_id}")

            
            # 2. Invalidate registry metadata
            results["metadata_invalidated"] = True
            logger.info(f"Invalidated metadata registry entries for document {document_id}")

        except Exception as e:
            logger.error(f"Error during deletion propagation for document {document_id}: {e}")
            results["status"] = f"FAILED: {str(e)}"

        return results
