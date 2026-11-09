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
    assert 'pre_filter = {"$and": [{"tenant_id": tenant_id}, {"acl_tier": {"$lte": int(clearance)}}]}' in src
    assert 'if tenant_id != "default_tenant"' not in src


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


# ---- clearance tiers ----

def tiered_store(vsm):
    vsm.add_chunks(chunks("Lunch menu: the canteen serves pasta on Mondays."), document_id="menu", tenant_id="t")
    vsm.add_chunks(chunks("Confidential pay bands: engineers earn 90k to 120k."), document_id="pay", tenant_id="t", acl_tier=2)
    vsm.add_chunks(chunks("Restricted merger plan: acquire Globex in Q3 for 40m."), document_id="merger", tenant_id="t", acl_tier=3)
    return vsm


def _filter(clearance):
    return {"$and": [{"tenant_id": "t"}, {"acl_tier": {"$lte": clearance}}]}


def test_retrieval_never_returns_documents_above_the_users_clearance(vsm):
    tiered_store(vsm)
    for clearance, allowed in ((1, {"menu"}), (2, {"menu", "pay"}), (3, {"menu", "pay", "merger"})):
        found = {h["metadata"]["document_id"] for h in vsm.search("pay merger menu plan", k=6, filter=_filter(clearance))}
        assert found <= allowed and found, (clearance, found)
    # the keyword index applies the same filter as Chroma
    keyword_only = vsm.hybrid_search("Globex merger acquire", k=6, filter=_filter(1))
    assert all(h["metadata"]["document_id"] == "menu" for h in keyword_only)


def test_a_documents_tier_can_be_changed_everywhere(vsm):
    tiered_store(vsm)
    pay = next(h for h, e in vsm.registry.items() if e["document_id"] == "pay")
    assert vsm.set_acl_tier(pay, 1) == 1
    assert vsm.registry[pay]["acl_tier"] == 1
    assert {h["metadata"]["document_id"] for h in vsm.search("pay bands engineers", k=6, filter=_filter(1))} >= {"pay"}
    with pytest.raises(ValueError):
        vsm.set_acl_tier(pay, 4)


def test_documents_indexed_before_tiers_existed_get_tier_1(tmp_path):
    first = VectorStoreManager(persist_directory=str(tmp_path))
    first.add_chunks(chunks("Old document about expense reports due within 30 days."), document_id="old")
    ids = first.vector_store.get(include=[])["ids"]
    first.vector_store._collection.update(ids=ids, metadatas=[{"source": "doc.txt", "tenant_id": DEFAULT_TENANT}])
    for c in first.chunks_cache:
        c["metadata"].pop("acl_tier", None)
    for e in first.registry.values():
        e.pop("acl_tier", None)
    first._save_bm25(); first._save_registry()

    second = VectorStoreManager(persist_directory=str(tmp_path))
    assert second.vector_store.get(include=["metadatas"])["metadatas"][0]["acl_tier"] == 1
    assert all(e["acl_tier"] == 1 for e in second.registry.values())
    assert second.search("expense reports due", k=2, filter={"$and": [{"tenant_id": DEFAULT_TENANT}, {"acl_tier": {"$lte": 1}}]})


def test_the_keyword_filter_supports_the_chroma_operators():
    from app.rag.vectorstore import metadata_matches
    meta = {"tenant_id": "t", "acl_tier": 2}
    assert metadata_matches(meta, {"$and": [{"tenant_id": "t"}, {"acl_tier": {"$lte": 2}}]})
    assert not metadata_matches(meta, {"$and": [{"tenant_id": "t"}, {"acl_tier": {"$lte": 1}}]})
    assert metadata_matches({"tenant_id": "t"}, {"acl_tier": {"$lte": 1}}), "a missing tier counts as 1"
    assert metadata_matches(meta, {"$or": [{"tenant_id": "x"}, {"acl_tier": {"$in": [2, 3]}}]})
    assert not metadata_matches(meta, {"tenant_id": {"$ne": "t"}})


@pytest.fixture()
def tiered_registry(monkeypatch):
    from app import routes
    registry = {
        "h1": {"filename": "menu.txt", "document_id": "menu", "tenant_id": "t", "acl_tier": 1, "ids": []},
        "h2": {"filename": "pay.txt", "document_id": "pay", "tenant_id": "t", "acl_tier": 2, "ids": []},
        "h3": {"filename": "merger.txt", "document_id": "merger", "tenant_id": "t", "acl_tier": 3, "ids": []},
    }
    monkeypatch.setattr(routes.vsm, "registry", registry)
    changed = []
    monkeypatch.setattr(routes.vsm, "set_acl_tier", lambda h, tier: changed.append((h, tier)) or 0)
    return changed


def test_listing_and_preview_hide_documents_above_the_users_clearance(users, tiered_registry):
    users.create("ann", tenant="t")
    users.store.update_user(users.store.get_by_username("ann").id, clearance=2)
    ann = users.headers("ann", tenant="t")
    assert set(client.get("/documents", headers=ann).json()) == {"h1", "h2"}
    assert client.get("/documents/h3/preview", headers=ann).status_code == 404
    assert set(client.get("/documents", headers=users.headers("vera", role="viewer", tenant="t")).json()) == {"h1"}
    assert set(client.get("/documents", headers=users.headers("tara", role="tenant_admin", tenant="t")).json()) == {"h1", "h2", "h3"}


def test_only_an_admin_of_the_tenant_changes_a_documents_tier(users, tiered_registry):
    assert client.patch("/documents/h2", json={"acl_tier": 1}, headers=users.headers("ann", tenant="t")).status_code == 403
    other = users.headers("otto", role="tenant_admin", tenant="other")
    assert client.patch("/documents/h2", json={"acl_tier": 1}, headers=other).status_code == 404
    tara = users.headers("tara", role="tenant_admin", tenant="t")
    assert client.patch("/documents/h2", json={"acl_tier": 9}, headers=tara).status_code == 400
    r = client.patch("/documents/h2", json={"acl_tier": 3}, headers=tara)
    assert r.status_code == 200 and tiered_registry == [("h2", 3)]


def test_only_an_admin_uploads_above_tier_1(users, saved_uploads):
    ann = users.headers("ann", tenant="t")
    assert client.post("/upload", files=FILE, data={"acl_tier": "2"}, headers=ann).status_code == 403
    assert client.post("/upload", files=FILE, data={"acl_tier": "7"}, headers=ann).status_code == 400
    tara = users.headers("tara", role="tenant_admin", tenant="t")
    r = client.post("/upload", files=FILE, data={"acl_tier": "3"}, headers=tara)
    assert r.status_code == 200 and r.json()["acl_tier"] == 3 and saved_uploads[-1][-1] == 3


def test_the_query_carries_the_accounts_clearance(users, monkeypatch):
    from app import routes
    calls = []
    monkeypatch.setattr(routes.engine, "query", lambda **kw: calls.append(kw) or {"answer": "ok", "citations": []})
    users.create("ann")
    users.store.update_user(users.store.get_by_username("ann").id, clearance=2)
    client.post("/query", json={"question": "hi"}, headers=users.headers("ann"))
    client.post("/query", json={"question": "hi"}, headers=users.headers("tara", role="tenant_admin"))
    assert [c["clearance"] for c in calls] == [2, 3]


def test_an_honest_low_clearance_user_gains_no_suspicion_from_restricted_documents(vsm):
    """Restricted chunks are filtered before Layer 2, so Layer 2 never counts an authorization failure for them."""
    from app.rag.trust_gate import Layer2TrustGate
    tiered_store(vsm)
    candidates = vsm.search("pay bands merger plan menu", k=6, filter=_filter(1))
    result = Layer2TrustGate(vsm=vsm).process_candidates(candidates=candidates, tenant_id="t", user_clearance=1)
    assert result.rejected_count == 0
