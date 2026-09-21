import json
import re
from typing import List, Dict, Any, Optional, Callable
from app.rag.layer3.contracts import GenerationContract, ClaimObject


class ContractConstrainedGenerator:
    """Generates contract-constrained structured JSON claims from Layer 2 evidence."""

    def __init__(self, llm_invoker: Optional[Callable[[str], str]] = None):
        self.llm_invoker = llm_invoker

    def generate(
        self,
        question: str,
        evidence_package: List[Dict[str, Any]],
        retry_instruction: Optional[str] = None
    ) -> GenerationContract:
        """Invokes LLM with strict system prompt requiring structured claim JSON."""
        # Build prompt evidence list
        evidence_lines = []
        for idx, item in enumerate(evidence_package):
            eid = item.get("metadata", {}).get("chunk_uid") or f"E{idx+1}"
            content = item.get("content", "")
            evidence_lines.append(f"[{eid}]: {content}")

        evidence_str = "\n".join(evidence_lines)

        prompt = (
            f"You are a strict security-verifiable RAG generator.\n"
            f"Question: {question}\n\n"
            f"Permitted Verified Evidence:\n{evidence_str}\n\n"
        )

        if retry_instruction:
            prompt += f"RETRY INSTRUCTION: {retry_instruction}\n\n"

        prompt += (
            f"Respond ONLY with a valid JSON object matching this schema:\n"
            f'{{"claims": [{{"claim_id": "C1", "order": 1, "text": "claim text", "evidence_ids": ["E1"]}}]}}\n'
            f"Rule: Do not include raw free-text prose outside the JSON object.\n"
        )

        if self.llm_invoker:
            raw_output = self.llm_invoker(prompt)
            return self._parse_output(raw_output, evidence_package)
        else:
            # Fallback heuristic generation for testing/mock environment
            return self._synthetic_fallback(question, evidence_package)

    def _parse_output(self, raw_output: str, evidence_package: List[Dict[str, Any]]) -> GenerationContract:
        """Extracts and parses JSON object from LLM response."""
        try:
            # Look for JSON block in markdown backticks or raw output
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return GenerationContract.model_validate(data)
        except Exception:
            pass

        return GenerationContract(claims=[])

    def _synthetic_fallback(self, question: str, evidence_package: List[Dict[str, Any]]) -> GenerationContract:
        """Generates synthetic claims based on evidence package when no LLM is attached."""
        claims = []
        for idx, item in enumerate(evidence_package):
            eid = item.get("metadata", {}).get("chunk_uid") or f"E{idx+1}"
            text = item.get("content", "")
            if text:
                claims.append(
                    ClaimObject(
                        claim_id=f"C{idx+1}",
                        order=idx + 1,
                        text=text.strip(),
                        evidence_ids=[eid]
                    )
                )
        return GenerationContract(claims=claims)
