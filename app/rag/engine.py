import os
import time
from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer
from . import tracing
from .llm_gate import llm_slot
from .vectorstore import VectorStoreManager
from .reranker import DocumentReranker
from .chunker import EMBEDDING_MODEL
from .threat_gate import ThreatGate
from .trust_gate import Layer2TrustGate
from .layer3 import Layer3Gate

PROMPT_PATH = os.path.join("prompts", "enterprise_rag_v1.yaml")
PATTERNS_PATH = os.path.join("config", "injection_patterns.yaml")

# Layer 3 failure reasons mapped to fixed codes for traces (some reasons carry tenant names or scanner detail)
_L3_REASON_CODES = (
    ("Cross-tenant", "tenant_violation"),
    ("Evidence gap", "evidence_gap"),
    ("The model returned no checkable answer", "no_checkable_answer"),
    ("No sentence of the draft", "nothing_verified"),
    ("Only ", "below_min_grounding"),
)


def l3_reason_code(reason: Optional[str], safety_flagged: bool = False) -> Optional[str]:
    if not reason:
        return None
    if safety_flagged:
        return "unsafe_output"
    for prefix, code in _L3_REASON_CODES:
        if reason.startswith(prefix):
            return code
    return "other"


def _uid(chunk: Dict[str, Any]) -> Optional[str]:
    return (chunk.get("metadata") or {}).get("chunk_uid")

CONTEXT_TOKEN_BUDGET = 1800  # ~45s to read on this project's CPU-only Llama 3 (~40 tok/s measured);
                              # matches chunker.TABLE_WHOLE_MAX_TOKENS
MAX_CONTEXT_ITEMS = 10
RETRIEVAL_K = 16         # candidates per retriever (dense and BM25 are fused inside the vector store)
RERANK_TOP_N = 8         # passages kept after cross-encoder reranking
RERANK_FLOOR = -5.0      # ms-marco cross-encoder logit below which a passage is treated as off-topic

class RAGEngine:
    """Orchestrates the pipeline: Layer 1 threat gate, hybrid retrieval, rerank, Layer 2 trust gate, Layer 3 output gate."""

    def __init__(self, vsm: Optional[VectorStoreManager] = None, adapt_downstream: bool = True, feedback: bool = True):
        # adapt_downstream: the user's suspicion also tightens Layers 2 and 3.
        # feedback: Layer 2 and 3 outcomes raise the user's suspicion. Both are switches for ablation studies.
        self.adapt_downstream, self.feedback = adapt_downstream, feedback
        self.vsm = vsm if vsm is not None else VectorStoreManager()
        self.reranker = DocumentReranker()
        self.threat_gate = ThreatGate()
        self.layer2_gate = Layer2TrustGate(vsm=self.vsm)
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3")
        # num_ctx must hold the evidence (~1800 tokens) + prompt + answer; Ollama's default is smaller
        self.llm = ChatOllama(model=self.model_name, temperature=0, num_ctx=4096, num_predict=700)
        self.threshold = RERANK_FLOOR
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.json_llm = ChatOllama(model=self.model_name, temperature=0, num_ctx=4096, num_predict=300, format="json")
        self.layer3_gate = Layer3Gate(llm_invoker=self._llm_invoke, json_invoker=self._json_invoke)
        self.versions = self._component_versions()

    def _component_versions(self) -> Dict[str, Any]:
        """What produced an answer, recorded in every trace so a change in behaviour can be tied to a change here."""
        return {
            "generator": self.model_name,
            "prompt_sha": tracing.file_sha256(PROMPT_PATH),
            "injection_patterns_sha": tracing.file_sha256(PATTERNS_PATH),
            "layer3": getattr(getattr(self, "layer3_gate", None), "model_version", None),
            "verifier": "v3" if os.getenv("ATTS_L3_V3", "1") == "1" else "v2",
            "embedding": EMBEDDING_MODEL,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }

    def _json_invoke(self, prompt: str) -> str:
        """JSON-mode LLM call (valid JSON guaranteed by Ollama); used only for the optional chart."""
        try:
            return self._invoke(self.json_llm, prompt).content
        except Exception as e:
            logger.error(f"Chart LLM invocation failed: {e}")
            return ""

    def _llm_invoke(self, prompt: str) -> str:
        """Plain-text LLM call used by Layer 3; a failure becomes an empty draft, which Layer 3 rejects."""
        try:
            return self._invoke(self.llm, prompt).content
        except Exception as e:
            logger.error(f"Layer 3 LLM invocation failed: {e}")
            return ""


    def count_tokens(self, text: str) -> int:
        """Tokens as the embedding model's tokenizer counts them (used only to size the context budget)."""
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    @staticmethod
    def _element_text(chunk: Dict[str, Any]) -> str:
        """The text to actually send to the LLM: a table's whole content if it has one, else the chunk itself."""
        metadata = chunk["metadata"]
        if metadata.get("element_type") == "table":
            whole = metadata.get("whole_content")
            if whole:
                return whole
        return chunk["content"]

    @staticmethod
    def _dedupe_by_parent(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapses several hits on the same table (summary + rows) into one entry.

        Keeps the highest-ranked occurrence's position, since chunks arrive already reranked.
        Paragraph chunks carry no parent_id, so they fall back to their own chunk_uid and are
        never merged with each other.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for c in chunks:
            metadata = c["metadata"]
            key = metadata.get("parent_id") or metadata.get("chunk_uid") or c["content"]
            if key not in seen:
                seen[key] = c
                order.append(key)
        return [seen[k] for k in order]

    def _assemble_within_budget(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Takes chunks in rank order until CONTEXT_TOKEN_BUDGET or MAX_CONTEXT_ITEMS is reached.

        Always includes at least the top-ranked element, even if it alone exceeds the budget.
        """
        used: List[Dict[str, Any]] = []
        total = 0
        for c in chunks:
            if len(used) >= MAX_CONTEXT_ITEMS:
                break
            tokens = self.count_tokens(self._element_text(c))
            if used and total + tokens > CONTEXT_TOKEN_BUDGET:
                break
            used.append(c)
            total += tokens
        return used or chunks[:1]

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
    def _invoke(self, llm, prompt: str):
        """Runs a single LLM call, retrying a transient Ollama failure once. Calls take turns (see llm_gate).

        Each attempt is recorded in the active trace: time spent waiting for the model slot, wall time, token
        counts and Ollama's own timings, or the exception type if it failed."""
        trace = tracing.current()
        queued = time.perf_counter()
        with llm_slot():
            started = time.perf_counter()
            try:
                response = llm.invoke(prompt)
            except Exception as exc:
                if trace is not None:
                    trace.record_llm(getattr(self, "model_name", "?"), (started - queued) * 1000.0,
                                     (time.perf_counter() - started) * 1000.0, error=exc)
                raise
        if trace is not None:
            trace.record_llm(getattr(self, "model_name", "?"), (started - queued) * 1000.0,
                             (time.perf_counter() - started) * 1000.0, response=response)
        return response


    def _expand_tables(self, evidence: List[Dict[str, Any]], reveal_pii: bool = False) -> List[Dict[str, Any]]:
        """Swaps a table hit's single row/summary view for the whole table, after redaction and a poison scan.

        Layer 2 vetted only the matched view, so the whole table gets the same PII redaction and
        injection scan before it can reach the LLM; if either check objects, the vetted view stays.
        """
        expanded = []
        for item in evidence:
            md = item["metadata"]
            whole = md.get("whole_content") if md.get("element_type") == "table" else None
            if whole:
                redactor = self.layer2_gate.redactor
                redacted, _ = redactor.redact(whole, reveal=redactor.REVEALABLE if reveal_pii else None)
                if self.layer2_gate.post_assembly_scan(redacted):
                    item = {**item, "content": redacted}
            expanded.append(item)
        return expanded

    ADMIN_AUDIT_LOG = os.path.join("logs", "admin_view_audit.jsonl")

    def _audit_admin_view(self, user_id: str, tenant_id: str, session_id: str, question: str,
                          evidence_items: List[Dict[str, Any]], revealed: Dict[str, int]) -> None:
        """Records that unmasked personal data was shown: who, when, which documents and how many items per class.

        The values themselves, and the question text, are never written (only a hash of the question)."""
        import hashlib
        import json
        from datetime import datetime, timezone
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user_id": user_id, "tenant_id": tenant_id, "session_id": session_id,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(), "question_chars": len(question),
            "documents": sorted({(i.get("metadata", {}).get("filename") or i.get("metadata", {}).get("source") or "document")
                                 for i in evidence_items}),
            "revealed_counts": revealed,
        }
        path = os.getenv("ADMIN_AUDIT_LOG", self.ADMIN_AUDIT_LOG)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.error(f"Could not write the admin-view audit record: {e}")
        logger.warning(f"ADMIN VIEW: unmasked personal data shown to {user_id} (tenant {tenant_id}): {revealed}")

    def query(
        self,
        question: str,
        session_id: str = "default_session",
        tenant_id: str = "default_tenant",
        user_id: str = "default_user",
        client_ip: str = "127.0.0.1",
        admin_view: bool = False,
        authenticated: bool = False,
        role: Optional[str] = None,
        clearance: int = 1,
    ) -> Dict[str, Any]:
        """Runs the full pipeline. admin_view=True (the caller must already have verified that the user is an admin)
        leaves personal data (SSN, cards, e-mail) unmasked in the evidence; API keys stay masked and Layer 3 still
        scans for secrets. authenticated=True means tenant_id and user_id come from a verified login token; role is that
        account's role (Layer 1's authority-claim check). clearance (1-3) is the account's clearance: documents of a
        higher tier are never retrieved (and Layer 2 checks it again).

        Every call writes one per-request trace (app/rag/tracing.py); the response carries its `trace_id` and a
        `telemetry` summary (per-stage milliseconds, LLM calls, tokens)."""
        trace = tracing.Trace(
            question, session_id=session_id, tenant_id=tenant_id, user_id=user_id, authenticated=bool(authenticated),
            role=role, clearance=int(clearance), admin_view=bool(admin_view),
            retrieval_mode="hybrid" if getattr(self.vsm, "bm25", None) else "dense",
            versions=getattr(self, "versions", None) or self._component_versions(),
        )
        with tracing.activate(trace):
            try:
                result = self._run_query(question, session_id, tenant_id, user_id, client_ip, admin_view, authenticated,
                                         role, clearance)
            except Exception as exc:
                trace.error("pipeline", exc)
                trace.finish("error")
                try:
                    exc.trace_id = trace.trace_id   # lets the API log name the trace instead of the question
                except Exception:
                    pass
                raise
        result["admin_view"] = bool(admin_view)
        result["trace_id"] = trace.trace_id
        result["telemetry"] = trace.summary()
        trace.finish(trace.outcome or "answered")
        return result

    def _run_query(self, question: str, session_id: str, tenant_id: str, user_id: str, client_ip: str,
                   admin_view: bool, authenticated: bool = False, role: Optional[str] = None,
                   clearance: int = 1) -> Dict[str, Any]:
        trace: List[str] = []   # the reasoning log shown to the user (not the per-request trace)
        active = tracing.current()
        tag = f"[{active.trace_id[:12]}] " if active is not None else ""

        def log(message: str):
            logger.info(tag + message)
            trace.append(message)

        def outcome(name: str):
            if active is not None:
                active.outcome = name

        # The question goes to the user's own reasoning log only; the server log gets its length, not its text
        trace.append(f"User Question: {question}")
        logger.info(f"{tag}Query received ({len(question)} chars)")

        # ---- Layer 1: threat gate -------------------------------------------------------------
        with tracing.span("layer1") as sp:
            threat = self.threat_gate.screen(
                query=question, session_id=session_id, tenant_id=tenant_id, user_id=user_id, client_ip=client_ip,
                authenticated=authenticated, role=role,
            )
            sp.set(allowed=threat.allowed, action=threat.action, final_risk=threat.final_risk, base_risk=threat.base_risk,
                   session_risk=threat.session_risk, detector_scores=threat.detector_scores, reasons=threat.reasons,
                   gate_ms=threat.execution_time_ms, query_rewritten=threat.processed_query != question)
        log(f"Layer 1 Threat Gate: action={threat.action}, risk={threat.final_risk:.3f}, time={threat.execution_time_ms:.1f}ms")
        l1 = {
            "allowed": threat.allowed, "action": threat.action, "final_risk": threat.final_risk,
            "base_risk": threat.base_risk, "session_risk": threat.session_risk,
            "detector_scores": threat.detector_scores, "reasons": threat.reasons,
            "execution_time_ms": threat.execution_time_ms,
            "suspicion": threat.session_risk, "policy": threat.policy.as_dict(),
        }
        store, user_key = self.threat_gate.trust_store, threat.session_key
        policy = threat.policy if self.adapt_downstream else None

        def note_suspicion():
            l1["suspicion_after"] = round(store.suspicion(user_key), 4)  # what the next query will be judged with
        note_suspicion()
        if active is not None:
            active.set(suspicion_after_l1=l1["suspicion_after"])
        if not threat.allowed:
            log(f"Layer 1 Gate BLOCKED query. Reasons: {threat.reasons}")
            outcome("blocked_l1")
            return {
                "answer": "Your query was blocked by the Layer 1 threat gate due to security policy violations.",
                "citations": [], "contexts": [], "reasoning_log": trace, "threat_gate": l1,
            }
        effective_query = threat.processed_query

        # ---- Retrieval: hybrid (dense + BM25, fused) then cross-encoder rerank -----------------
        # Every chunk carries a tenant and a clearance tier (older ones were tagged default_tenant / 1 on start-up).
        # Chunks above the user's clearance are filtered out here, not left for Layer 2: Layer 2's authorization
        # failures raise suspicion, which would punish an honest user for what retrieval happened to find.
        pre_filter = {"$and": [{"tenant_id": tenant_id}, {"acl_tier": {"$lte": int(clearance)}}]}
        with tracing.span("retrieval", k=RETRIEVAL_K) as sp:
            candidates = self.vsm.search(effective_query, k=RETRIEVAL_K, filter=pre_filter)
            sp.set(n_candidates=len(candidates),
                   candidates=[{"rank": r, "chunk_uid": _uid(c), "score": c.get("retrieval_score")}
                               for r, c in enumerate(candidates, 1)])
        log(f"Reranking {len(candidates)} candidates...")
        with tracing.span("rerank", top_n=RERANK_TOP_N, floor=self.threshold) as sp:
            reranked = self.reranker.rerank(effective_query, candidates, top_n=RERANK_TOP_N)
            relevant = [c for c in reranked if c.get("rerank_score", -99.0) >= self.threshold]
            # from_rank: where retrieval had put the chunk, to see how much the cross-encoder reordered
            first_rank = {id(c): r for r, c in enumerate(candidates, 1)}
            sp.set(n_above_floor=len(relevant),
                   ranked=[{"rank": r, "chunk_uid": _uid(c), "score": c.get("rerank_score"), "from_rank": first_rank.get(id(c)),
                            "above_floor": c.get("rerank_score", -99.0) >= self.threshold}
                           for r, c in enumerate(reranked, 1)])
        if not relevant:
            log(f"No passage scored above the relevance floor ({self.threshold}); declining to answer.")
            outcome("no_relevant")
            return {
                "answer": "I could not find anything in the uploaded documents that is relevant to that question.",
                "citations": [], "contexts": [], "reasoning_log": trace, "threat_gate": l1,
            }
        deduped = self._dedupe_by_parent(relevant)

        # ---- Layer 2: trust gate ----------------------------------------------------------------
        with tracing.span("layer2", n_in=len(deduped)) as sp:
            layer2 = self.layer2_gate.process_candidates(candidates=deduped, tenant_id=tenant_id, user_clearance=int(clearance), policy=policy,
                                                         reveal_pii=admin_view)
            sp.set(allowed=layer2.allowed, status=layer2.status, tier_1=layer2.tier_1_count, tier_2=layer2.tier_2_count,
                   withheld=layer2.withheld_count, quarantined=len(layer2.quarantined_items),
                   unauthorized=layer2.rejected_count, fallback_unverified=layer2.fallback_unverified,
                   gate_ms=layer2.execution_time_ms, evidence=[_uid(i) for i in layer2.evidence_package])
        if self.feedback:  # Layer 2 -> suspicion: poisoned or unauthorized candidates were seen for this user
            store.observe_layer2(user_key, quarantined=len(layer2.quarantined_items), auth_failures=layer2.rejected_count)
            note_suspicion()
        log(f"Layer 2 Trust Gate: status={layer2.status}, TIER_1={layer2.tier_1_count}, TIER_2={layer2.tier_2_count}, time={layer2.execution_time_ms:.1f}ms")
        if not layer2.allowed:
            log(f"Layer 2 Gate output INSUFFICIENT_TRUSTED_EVIDENCE. Reasons: {layer2.reasons}")
            outcome("blocked_l2")
            return {
                "answer": "I cannot find any verifiable, trusted information in the uploaded documents to answer that question.",
                "citations": [], "contexts": [], "reasoning_log": trace, "threat_gate": l1,
                "layer2_gate": {"allowed": False, "status": layer2.status, "reasons": layer2.reasons,
                                "execution_time_ms": layer2.execution_time_ms, "withheld_count": layer2.withheld_count},
            }
        l2 = {"allowed": True, "status": layer2.status, "tier_1_count": layer2.tier_1_count,
              "tier_2_count": layer2.tier_2_count, "execution_time_ms": layer2.execution_time_ms,
              "withheld_count": layer2.withheld_count, "fallback_unverified": layer2.fallback_unverified,
              "revealed": layer2.revealed}

        with tracing.span("assembly") as sp:
            evidence_items = self._assemble_within_budget(self._expand_tables(layer2.evidence_package, reveal_pii=admin_view))
            if active is not None:
                sp.set(n_items=len(evidence_items), budget_tokens=CONTEXT_TOKEN_BUDGET,
                       context_tokens=sum(self.count_tokens(self._element_text(c)) for c in evidence_items),
                       evidence=[_uid(i) for i in evidence_items])
        if admin_view and layer2.revealed:
            self._audit_admin_view(user_id, tenant_id, session_id, question, evidence_items, layer2.revealed)
        log(f"Assembled {len(evidence_items)} evidence passage(s) for generation.")

        # ---- Layer 3: grounded answer + verification -------------------------------------------
        # The trace id doubles as Layer 3's query id, so a layer3_audit.jsonl record joins its trace
        with tracing.span("layer3") as sp:
            decision = self.layer3_gate.process(
                question=effective_query,
                layer2_evidence_package=evidence_items,
                query_id=active.trace_id if active is not None else f"Q-{session_id}",
                tenant_id=tenant_id,
                policy=policy,
            )
            sp.set(decision=decision.decision, reason_code=l3_reason_code(decision.failure_reason, decision.safety_flagged),
                   verified=len(decision.verified_claims), failed=len(decision.failed_claims),
                   grounding_ratio=decision.grounding_ratio, contradictions=decision.contradictions,
                   safety_flagged=decision.safety_flagged, chart=decision.chart is not None,
                   gate_ms=decision.telemetry.get("execution_time_ms", 0.0))
        if self.feedback:  # Layer 3 -> suspicion: the answer leaked something or contradicted the evidence
            store.observe_layer3(user_key, safety_hit=decision.safety_flagged, contradictions=decision.contradictions)
            note_suspicion()
        log(f"Layer 3 Output Gate: decision={decision.decision}, verified={len(decision.verified_claims)}, "
            f"unsupported={len(decision.failed_claims)}, time={decision.telemetry.get('execution_time_ms', 0.0):.0f}ms")

        by_uid = {(i.get("metadata", {}).get("chunk_uid") or ""): i for i in evidence_items}
        by_label = {label: by_uid.get(uid) for label, uid in decision.evidence_map.items()}

        def detail(label: str, item: Dict[str, Any]) -> Dict[str, Any]:
            meta = item.get("metadata", {})
            return {
                "id": label,
                "chunk_uid": meta.get("chunk_uid"),
                "filename": meta.get("filename") or meta.get("source", "Document"),
                "page": meta.get("page"),
                "source_tier": meta.get("source_tier", "unknown"),
                "relevance_score": round(float(item.get("relevance_score", 0.0)), 3),
                "snippet": item.get("content", "")[:400],
                "text": item.get("content", "")[:2500],
            }

        l3 = {
            "decision": decision.decision,
            "reason": decision.failure_reason,
            "verified_claims_count": len(decision.verified_claims),
            "failed_claims_count": len(decision.failed_claims),
            "grounding_ratio": decision.grounding_ratio,
            "execution_time_ms": decision.telemetry.get("execution_time_ms", 0.0),
            "telemetry": decision.telemetry,
        }
        contexts = [i["content"] for i in evidence_items]

        if active is not None:
            active.set(suspicion_after=l1["suspicion_after"])
        if decision.decision == "REJECT":
            outcome("rejected_l3")
            return {
                "answer": "I found related passages, but could not produce an answer that I could verify against them, "
                          "so I am not releasing one. The closest sources are listed below.",
                "citations": [], "citation_details": [detail(l, i) for l, i in by_label.items() if i],
                "contexts": contexts, "reasoning_log": trace,
                "threat_gate": l1, "layer2_gate": l2, "layer3_gate": l3,
            }

        used_labels = list(dict.fromkeys(e for c in decision.verified_claims for e in c.evidence_ids))
        outcome("answered")
        if active is not None:
            active.set(cited=[by_label[l].get("metadata", {}).get("chunk_uid") for l in used_labels if by_label.get(l)])
        return {
            "answer": decision.reconstructed_answer,
            "intent": decision.intent,
            "chart": decision.chart.model_dump() if decision.chart else None,
            "citations": used_labels,
            "citation_details": [detail(l, by_label[l]) for l in used_labels if by_label.get(l)],
            "contexts": contexts,
            "reasoning_log": trace,
            "threat_gate": l1,
            "layer2_gate": l2,
            "layer3_gate": {**l3, "verified_claims": [c.model_dump() for c in decision.verified_claims]},
        }
