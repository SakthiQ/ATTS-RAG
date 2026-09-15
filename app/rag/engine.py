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

RERANK_CANDIDATES = 20  # Reranked before deduplication, so several rows of one table can compete
CONTEXT_TOKEN_BUDGET = 1800  # ~45s to read on this project's CPU-only Llama 3 (~40 tok/s measured);
                              # matches chunker.TABLE_WHOLE_MAX_TOKENS
MAX_CONTEXT_ITEMS = 10

class RAGEngine:
    """The Central Brain: Orchestrates query expansion, retrieval, reranking, and synthesis."""

    def __init__(self, vsm: Optional[VectorStoreManager] = None):
        self.vsm = vsm if vsm is not None else VectorStoreManager()
        self.reranker = DocumentReranker()
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3")
        self.llm = ChatOllama(model=self.model_name, temperature=0)
        self.json_llm = ChatOllama(model=self.model_name, temperature=0, format="json")
        self.threshold = -5.0  # Rerank score threshold
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.reasoning_log = []

        # Load the enterprise prompt from YAML
        self.prompt_template = self._load_template("prompts/enterprise_rag_v1.yaml")

    def _load_template(self, path: str) -> str:
        """Loads prompt template from a YAML file."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config["template"]
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

    def query(self, question: str) -> Dict[str, Any]:
        """Entry point for the RAG pipeline."""
        self.reasoning_log = []
        self._log(f"User Question: {question}")

        # FAST PATH: Short queries (e.g. 1-3 keywords) skip query expansion
        words = question.strip().split()
        if len(words) <= 3:
            self._log("Fast Path: Short query detected. Skipping query expansion.")
            search_queries = [question]
        else:
            search_queries = self.multi_query_expand(question)

        # Retrieval
        all_candidates = []
        for q in search_queries:
            all_candidates.extend(self.vsm.search(q, k=10))

        # Deduplicate candidates by content
        unique_candidates = list({c["content"]: c for c in all_candidates}.values())

        # Rerank a wide pool, then collapse multiple hits on the same table to one entry
        self._log(f"Reranking {len(unique_candidates)} unique candidates...")
        reranked = self.reranker.rerank(question, unique_candidates, top_n=RERANK_CANDIDATES)
        deduped = self._dedupe_by_parent(reranked)
        if len(deduped) != len(reranked):
            self._log(f"Deduplicated {len(reranked)} candidates to {len(deduped)} unique elements (grouped by table).")

        return self._generate_final_answer(question, deduped)

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
