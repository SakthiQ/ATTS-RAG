import os
import yaml
from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer
from .vectorstore import VectorStoreManager
from .reranker import DocumentReranker
from .chunker import EMBEDDING_MODEL
from .threat_gate import ThreatGate
from .trust_gate import Layer2TrustGate
from .layer3 import Layer3Gate

RERANK_CANDIDATES = 20  # Reranked before deduplication, so several rows of one table can compete
CONTEXT_TOKEN_BUDGET = 1800  # ~45s to read on this project's CPU-only Llama 3 (~40 tok/s measured);
                              # matches chunker.TABLE_WHOLE_MAX_TOKENS
MAX_CONTEXT_ITEMS = 10

class RAGEngine:
    """The Central Brain: Orchestrates query expansion, retrieval, reranking, and synthesis."""

    def __init__(self, vsm: Optional[VectorStoreManager] = None):
        self.vsm = vsm if vsm is not None else VectorStoreManager()
        self.reranker = DocumentReranker()
        self.threat_gate = ThreatGate()
        self.layer2_gate = Layer2TrustGate(vsm=self.vsm)
        self.layer3_gate = Layer3Gate(llm_invoker=self._raw_llm_invoke)
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3")
        self.llm = ChatOllama(model=self.model_name, temperature=0)
        self.json_llm = ChatOllama(model=self.model_name, temperature=0, format="json")
        self.threshold = -5.0  # Rerank score threshold
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.reasoning_log = []

        # Load the enterprise prompt from YAML
        self.prompt_template = self._load_template("prompts/enterprise_rag_v1.yaml")

    def _raw_llm_invoke(self, prompt: str) -> str:
        """Invokes LLM for Layer 3 claim extraction / JSON generation."""
        try:
            return self._invoke(self.json_llm, prompt)
        except Exception as e:
            logger.error(f"Layer 3 LLM invocation failed: {e}")
            return ""


    def _load_template(self, path: str) -> str:
        """Loads prompt template from a YAML file."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if "template" in config:
                return config["template"]
            if "system_instruction" in config:
                return config["system_instruction"]
        return "{context}\n\n{question}"

    def _log(self, message: str):
        """Helper to log both to console and the reasoning trace."""
        logger.info(message)
        self.reasoning_log.append(message)

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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _invoke(self, chain, inputs: Dict[str, Any]):
        """Runs a single LLM call, retrying transient Ollama failures."""
        return chain.invoke(inputs)

    def multi_query_expand(self, question: str) -> List[str]:
        """Returns the original question plus up to 2 variations to improve retrieval recall."""
        self._log("Recall: Generating query variations...")
        prompt = ChatPromptTemplate.from_template(
            "Generate 2 different variations of the following question to retrieve diverse context. "
            "Return a JSON list of strings with the key 'queries'.\nQuestion: {question}"
        )
        chain = prompt | self.json_llm | JsonOutputParser()
        try:
            variations = self._invoke(chain, {"question": question}).get("queries", [])
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            variations = []

        # Always search the user's own question; variations only add to it
        extra = [v for v in variations if isinstance(v, str) and v.strip()][:2]
        queries = [question] + extra
        self._log(f"Search queries: {queries}")
        return queries

    def query(
        self,
        question: str,
        session_id: str = "default_session",
        tenant_id: str = "default_tenant",
        user_id: str = "default_user",
        client_ip: str = "127.0.0.1"
    ) -> Dict[str, Any]:
        """Entry point for the RAG pipeline with Layer 1 Threat Gate & Layer 2 Trust Gate."""
        self.reasoning_log = []
        self._log(f"User Question: {question}")

        # Layer 1: Adaptive Threat Intelligence Gate
        threat_result = self.threat_gate.screen(
            query=question,
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            client_ip=client_ip
        )

        self._log(f"Layer 1 Threat Gate: action={threat_result.action}, risk={threat_result.final_risk:.3f}, time={threat_result.execution_time_ms:.1f}ms")

        if not threat_result.allowed:
            self._log(f"Layer 1 Gate BLOCKED query. Reasons: {threat_result.reasons}")
            return {
                "answer": "Your query was blocked by the Layer 1 Adaptive Threat Intelligence Gate due to security policy violations.",
                "citations": [],
                "contexts": [],
                "reasoning_log": self.reasoning_log,
                "threat_gate": {
                    "allowed": False,
                    "action": threat_result.action,
                    "final_risk": threat_result.final_risk,
                    "base_risk": threat_result.base_risk,
                    "session_risk": threat_result.session_risk,
                    "disagreement": threat_result.disagreement,
                    "detector_scores": threat_result.detector_scores,
                    "reasons": threat_result.reasons,
                    "execution_time_ms": threat_result.execution_time_ms
                }
            }

        effective_query = threat_result.processed_query

        # FAST PATH: Short queries (e.g. 1-3 keywords) skip query expansion
        words = effective_query.strip().split()
        if len(words) <= 3:
            self._log("Fast Path: Short query detected. Skipping query expansion.")
            search_queries = [effective_query]
        else:
            search_queries = self.multi_query_expand(effective_query)

        # Retrieval with Pre-Retrieval Tenant Metadata Filter
        pre_filter = {"tenant_id": tenant_id} if tenant_id != "default_tenant" else None
        all_candidates = []
        for q in search_queries:
            all_candidates.extend(self.vsm.search(q, k=10, filter=pre_filter))

        # Deduplicate candidates by content
        unique_candidates = list({c["content"]: c for c in all_candidates}.values())

        # Rerank a wide pool
        self._log(f"Reranking {len(unique_candidates)} unique candidates...")
        reranked = self.reranker.rerank(effective_query, unique_candidates, top_n=RERANK_CANDIDATES)
        deduped = self._dedupe_by_parent(reranked)

        # Layer 2: Knowledge Trust & Retrieval Gate
        layer2_result = self.layer2_gate.process_candidates(
            candidates=deduped,
            tenant_id=tenant_id,
            user_clearance=1
        )

        self._log(f"Layer 2 Trust Gate: status={layer2_result.status}, TIER_1={layer2_result.tier_1_count}, TIER_2={layer2_result.tier_2_count}, time={layer2_result.execution_time_ms:.1f}ms")

        if not layer2_result.allowed:
            self._log(f"Layer 2 Gate output INSUFFICIENT_TRUSTED_EVIDENCE. Reasons: {layer2_result.reasons}")
            return {
                "answer": "I'm sorry, I cannot find any verifiable, trusted information in the uploaded documents to answer that question.",
                "citations": [],
                "contexts": [],
                "reasoning_log": self.reasoning_log,
                "threat_gate": {
                    "allowed": True,
                    "action": threat_result.action,
                    "final_risk": threat_result.final_risk
                },
                "layer2_gate": {
                    "allowed": False,
                    "status": layer2_result.status,
                    "reasons": layer2_result.reasons,
                    "execution_time_ms": layer2_result.execution_time_ms
                }
            }

        # Convert Layer 2 evidence package back to chunk format for synthesis
        trusted_chunks = [
            {"content": item["content"], "metadata": item["metadata"], "rerank_score": item["relevance_score"]}
            for item in layer2_result.evidence_package
        ]

        # Layer 3: Evidence-to-Answer & Output Verification Gate
        import asyncio
        import concurrent.futures

        def _run_in_thread():
            return asyncio.run(
                self.layer3_gate.process(
                    question=effective_query,
                    layer2_evidence_package=layer2_result.evidence_package,
                    query_id=f"Q-{session_id}",
                    tenant_id=tenant_id
                )
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                layer3_decision = executor.submit(_run_in_thread).result()
        else:
            layer3_decision = _run_in_thread()

        self._log(f"Layer 3 Output Gate: decision={layer3_decision.decision}, verified_claims={len(layer3_decision.verified_claims)}, retries={layer3_decision.retry_count}")

        if layer3_decision.decision == "REJECT":
            return {
                "answer": "I'm sorry, the generated response could not pass Layer 3 output verification and safety checks.",
                "citations": [],
                "contexts": [c["content"] for c in trusted_chunks],
                "reasoning_log": self.reasoning_log,
                "threat_gate": {"allowed": True, "action": threat_result.action, "final_risk": threat_result.final_risk},
                "layer2_gate": {"allowed": True, "status": layer2_result.status},
                "layer3_gate": {
                    "decision": "REJECT",
                    "reason": layer3_decision.failure_reason,
                    "retry_count": layer3_decision.retry_count,
                    "execution_time_ms": layer3_decision.telemetry.get("execution_time_ms", 0.0) if layer3_decision.telemetry else 0.0,
                    "telemetry": layer3_decision.telemetry
                }
            }

        citations = []
        citation_details = []
        for item in layer2_result.evidence_package:
            meta = item.get("metadata", {})
            cid = item.get("id") or (str(meta.get("doc_hash", "")) + ":" + str(meta.get("chunk_id", "")))
            citation_details.append({
                "id": cid,
                "filename": meta.get("filename", "Document"),
                "source_tier": meta.get("source_tier", "unknown"),
                "relevance_score": round(float(item.get("relevance_score", 0.0)), 3),
                "trust_weight": float(item.get("trust_weight", 1.0)),
                "snippet": item.get("content", "")[:400]
            })

        for claim in layer3_decision.verified_claims:
            for eid in claim.evidence_ids:
                if eid not in citations:
                    citations.append(eid)

        return {
            "answer": layer3_decision.reconstructed_answer,
            "citations": citations,
            "citation_details": citation_details,
            "contexts": [c["content"] for c in trusted_chunks],
            "reasoning_log": self.reasoning_log,
            "threat_gate": {
                "allowed": True,
                "action": threat_result.action,
                "final_risk": threat_result.final_risk,
                "reasons": threat_result.reasons,
                "execution_time_ms": threat_result.execution_time_ms
            },
            "layer2_gate": {
                "allowed": True,
                "status": layer2_result.status,
                "tier_1_count": layer2_result.tier_1_count,
                "tier_2_count": layer2_result.tier_2_count,
                "execution_time_ms": layer2_result.execution_time_ms
            },
            "layer3_gate": {
                "decision": "PASS",
                "verified_claims": [c.model_dump() for c in layer3_decision.verified_claims],
                "verified_claims_count": len(layer3_decision.verified_claims),
                "failed_claims_count": len(layer3_decision.failed_claims),
                "retry_count": layer3_decision.retry_count,
                "execution_time_ms": layer3_decision.telemetry.get("execution_time_ms", 0.0) if layer3_decision.telemetry else 0.0,
                "telemetry": layer3_decision.telemetry
            }
        }



    def _generate_final_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes the final response using the Enterprise template."""
        self._log("Synthesizing final answer with Enterprise Template...")

        if not context_chunks:
             return {
                "answer": "I couldn't find any relevant information to answer that question.",
                "citations": [],
                "contexts": [],
                "reasoning_log": self.reasoning_log
            }

        best_score = max([c.get("rerank_score", -10) for c in context_chunks])
        if best_score < self.threshold:
            self._log(f"Threshold: Best rerank score {best_score} below {self.threshold}. Declining to answer.")
            return {
                "answer": "I'm sorry, I cannot find any relevant information in the uploaded documents to answer that question.",
                "citations": [],
                "contexts": [],
                "reasoning_log": self.reasoning_log
            }

        used_chunks = self._assemble_within_budget(context_chunks)
        context_text = "\n\n".join(self._element_text(c) for c in used_chunks)
        self._log(f"Assembled {len(used_chunks)} element(s), {self.count_tokens(context_text)} tokens, into the context.")

        # A table too large for TABLE_WHOLE_MAX_TOKENS was never given a whole_content at ingestion,
        # so _element_text fell back to whatever single row/summary view matched the search. Log
        # that loudly here rather than let it pass as if the model saw the complete table.
        for c in used_chunks:
            md = c["metadata"]
            if md.get("element_type") == "table" and not md.get("whole_content"):
                self._log(
                    f"Degraded table context: the table on page {md.get('page', '?')} "
                    f"({md.get('table_rows', '?')} rows) exceeded the context budget, so only its "
                    f"matched '{md.get('view_type', '?')}' view was used, not the whole table."
                )

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = self._invoke(chain, {"context": context_text, "question": question})
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "Error generating answer. Please try again."

        # Form Citations
        citations = []
        for r in used_chunks:
            citations.append({
                "source": r["metadata"].get("source", "Unknown"),
                "page": r["metadata"].get("page", "N/A"),
                "score": round(r.get("rerank_score", 0), 2),
                "type": r["metadata"].get("element_type", "paragraph"),
            })

        return {
            "answer": answer,
            "citations": citations,
            "contexts": [self._element_text(c) for c in used_chunks],
            "reasoning_log": self.reasoning_log
        }
