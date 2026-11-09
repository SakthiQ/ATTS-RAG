"""Tenant isolation: documents carry the tenant that uploaded them, and a query only ever reaches its own tenant's documents."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.vectorstore import DEFAULT_TENANT, VectorStoreManager

client = TestClient(app)
FILE = {"file": ("policy.txt", b"Refunds are accepted within 30 days.", "text/plain")}


def chunks(text):
    return [{"content": text, "metadata": {"source": "doc.txt", "scan_status": "clean", "anomaly_score": 0.0}}]


@pytest.fixture()
def vsm(tmp_path):
    return VectorStoreManager(persist_directory=str(tmp_path))


def test_chunks_and_registry_record_the_tenant(vsm):
    vsm.add_chunks(chunks("Acme refunds are accepted within 30 days of purchase."), document_id="a_policy", tenant_id="tenant_a")
    assert [e["tenant_id"] for e in vsm.registry.values()] == ["tenant_a"]
    assert all(c["metadata"]["tenant_id"] == "tenant_a" for c in vsm.chunks_cache)
    assert vsm.vector_store.get(include=["metadatas"])["metadatas"][0]["tenant_id"] == "tenant_a"


def test_tenant_defaults_to_the_default_tenant(vsm):
    vsm.add_chunks(chunks("Employees rotate passwords every 90 days."), document_id="pw")
    assert [e["tenant_id"] for e in vsm.registry.values()] == [DEFAULT_TENANT]


def test_a_tenant_filter_only_returns_that_tenants_documents(vsm):
    vsm.add_chunks(chunks("Tenant A secret roadmap: launch the Falcon product in March."), document_id="a_doc", tenant_id="tenant_a")
    vsm.add_chunks(chunks("Tenant B secret roadmap: launch the Osprey product in June."), document_id="b_doc", tenant_id="tenant_b")
    for tenant, other in (("tenant_a", "Osprey"), ("tenant_b", "Falcon")):
        hits = vsm.search("secret roadmap launch product", k=4, filter={"tenant_id": tenant})
        assert hits and all(h["metadata"]["tenant_id"] == tenant for h in hits)
        assert not any(other in h["content"] for h in hits)


def test_documents_indexed_before_tenants_were_recorded_are_migrated(tmp_path):
    first = VectorStoreManager(persist_directory=str(tmp_path))
    first.add_chunks(chunks("Old document about expense reports due within 30 days."), document_id="old")
    # simulate an index written by older code: no tenant anywhere
    first.vector_store._collection.update(ids=first.vector_store.get(include=[])["ids"], metadatas=[{"source": "doc.txt"}])
    for c in first.chunks_cache:
        c["metadata"].pop("tenant_id", None)
    for e in first.registry.values():
        e.pop("tenant_id", None)
    first._save_bm25(); first._save_registry()

    second = VectorStoreManager(persist_directory=str(tmp_path))   # start-up migration
    assert all(c["metadata"]["tenant_id"] == DEFAULT_TENANT for c in second.chunks_cache)
    assert all(e["tenant_id"] == DEFAULT_TENANT for e in second.registry.values())
    hits = second.search("expense reports due", k=2, filter={"tenant_id": DEFAULT_TENANT})
    assert hits, "old documents must stay visible to the default tenant"
    assert second._tag_untagged_chunks() == 0   # idempotent


def test_delete_by_document_id_respects_the_tenant(vsm):
    vsm.add_chunks(chunks("Shared name, tenant A text about onboarding steps."), document_id="handbook", tenant_id="tenant_a")
    vsm.add_chunks(chunks("Shared name, tenant B text about onboarding steps and equipment."), document_id="handbook", tenant_id="tenant_b")
    assert vsm.delete_document_by_id("handbook", tenant_id="tenant_a") > 0
    assert [e["tenant_id"] for e in vsm.registry.values()] == ["tenant_b"]


# ---- upload endpoint ----

def test_uploading_for_another_tenant_requires_an_admin(users):
    resp = client.post("/upload", files=FILE, data={"tenant_id": "tenant_b"}, headers=users.headers("ann", tenant="tenant_a"))
    assert resp.status_code == 403
    resp = client.post("/upload", files=FILE, data={"tenant_id": "tenant_b"})
    assert resp.status_code == 401


def test_invalid_tenant_id_is_rejected(users):
    resp = client.post("/upload", files=FILE, data={"tenant_id": "../other tenant"}, headers=users.headers("ann"))
    assert resp.status_code == 400


def test_engine_always_filters_retrieval_by_tenant():
    import inspect
    from app.rag import engine
    src = inspect.getsource(engine.RAGEngine._run_query)
    assert 'pre_filter = {"tenant_id": tenant_id}' in src and 'if tenant_id != "default_tenant"' not in src


# ---- document IDs, versions and identical files are per tenant ----

def test_the_same_file_in_two_tenants_is_two_documents(vsm):
    text = "Travel claims must be filed within 14 days of returning."
    a = vsm.add_chunks(chunks(text), document_id="travel", tenant_id="tenant_a")
    b = vsm.add_chunks(chunks(text), document_id="travel", tenant_id="tenant_b")
    assert a["status"] == b["status"] == "indexed", "tenant B's copy must not be skipped as a duplicate of A's"
    assert a["content_hash"] != b["content_hash"]
    assert vsm.add_chunks(chunks(text), document_id="travel", tenant_id="tenant_b")["status"] == "duplicate"


def test_default_tenant_keys_are_unchanged():
    from types import SimpleNamespace
    stub = SimpleNamespace(_calculate_hash=lambda t: VectorStoreManager._calculate_hash(None, t))
    c = chunks("Unchanged content.")
    assert VectorStoreManager.document_hash(stub, c) == VectorStoreManager._calculate_hash(None, "Unchanged content.")
    assert VectorStoreManager.document_hash(stub, c, "tenant_a") != VectorStoreManager.document_hash(stub, c)


def test_versions_are_counted_per_tenant(vsm):
    vsm.add_chunks(chunks("Handbook v1 for tenant A about laptops."), document_id="handbook", tenant_id="tenant_a")
    assert vsm.add_chunks(chunks("Handbook v2 for tenant A about laptops and phones."),
                          document_id="handbook", tenant_id="tenant_a")["version"] == 2
    assert vsm.add_chunks(chunks("Tenant B's own handbook about desks."),
                          document_id="handbook", tenant_id="tenant_b")["version"] == 1


@pytest.fixture()
def saved_uploads(tmp_path, monkeypatch):
    """Uploads go to a temporary folder and the background indexer only records what it was given."""
    from app import routes
    monkeypatch.setattr(routes, "UPLOAD_DIR", str(tmp_path))
    jobs = []
    monkeypatch.setattr(routes, "process_document_background", lambda *args: jobs.append(args))
    return jobs


def test_a_user_may_reuse_a_document_id_that_only_another_tenant_has(users, monkeypatch, saved_uploads):
    from app import routes
    monkeypatch.setitem(routes.vsm.registry, "b-hash", {"document_id": "refund_policy", "tenant_id": "tenant_b"})
    resp = client.post("/upload", files=FILE, data={"document_id": "refund_policy"}, headers=users.headers("ann", tenant="tenant_a"))
    assert resp.status_code == 200, "another tenant's IDs must neither block the upload nor reveal that they exist"
    monkeypatch.setitem(routes.vsm.registry, "a-hash", {"document_id": "refund_policy", "tenant_id": "tenant_a"})
    resp = client.post("/upload", files=FILE, data={"document_id": "refund_policy"}, headers=users.headers("ann", tenant="tenant_a"))
    assert resp.status_code == 409


def test_uploads_with_the_same_name_never_share_a_file(users, saved_uploads):
    import os
    client.post("/upload", files=FILE, headers=users.headers("ann", tenant="tenant_a"))
    client.post("/upload", files=FILE, data={"document_id": "policy_2"}, headers=users.headers("ann", tenant="tenant_a"))
    client.post("/upload", files=FILE, headers=users.headers("bob", tenant="tenant_b"))
    paths = [job[0] for job in saved_uploads]
    assert len(paths) == 3 and len(set(paths)) == 3
    assert all(os.path.basename(p) == "policy.txt" and os.path.exists(p) for p in paths)
    assert os.sep + "tenant_a" + os.sep in paths[0] and os.sep + "tenant_b" + os.sep in paths[2]
