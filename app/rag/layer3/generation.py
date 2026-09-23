import json
import re
from typing import List, Dict, Any, Optional, Callable
from app.rag.layer3.contracts import GenerationContract, ClaimObject


import os
import yaml

class ContractConstrainedGenerator:
    """Generates contract-constrained structured JSON claims from Layer 2 evidence."""

    def __init__(self, llm_invoker: Optional[Callable[[str], str]] = None, prompt_path: str = "prompts/enterprise_rag_v1.yaml"):
        self.llm_invoker = llm_invoker
        self.prompt_config = self._load_prompt_config(prompt_path)

    def _load_prompt_config(self, path: str) -> Dict[str, str]:
        """Loads prompt system instruction and schema rules from YAML config."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    return {
                        "system_instruction": cfg.get("system_instruction", "You are an Enterprise RAG Generator operating under strict zero-trust security controls."),
                        "json_schema_rule": cfg.get("json_schema_rule", 'Respond ONLY with a valid JSON object matching this schema:\n{"claims": [{"claim_id": "C1", "order": 1, "text": "claim text", "evidence_ids": ["E1"]}]}')
                    }
            except Exception:
                pass
        return {
            "system_instruction": "You are a strict security-verifiable RAG generator.",
            "json_schema_rule": 'Respond ONLY with a valid JSON object matching this schema:\n{"claims": [{"claim_id": "C1", "order": 1, "text": "claim text", "evidence_ids": ["E1"]}]}'
        }

    def generate(
        self,
        question: str,
        evidence_package: List[Dict[str, Any]],
        retry_instruction: Optional[str] = None
    ) -> GenerationContract:
        """Invokes LLM with system prompt from enterprise_rag_v1.yaml requiring structured claim JSON."""
        # Build prompt evidence list
        evidence_lines = []
        for idx, item in enumerate(evidence_package):
            eid = item.get("metadata", {}).get("chunk_uid") or f"E{idx+1}"
            content = item.get("content", "")
            evidence_lines.append(f"[{eid}]: {content}")

        evidence_str = "\n".join(evidence_lines)

        sys_inst = self.prompt_config["system_instruction"]
        schema_rule = self.prompt_config["json_schema_rule"]

        prompt = (
            f"{sys_inst}\n\n"
            f"Question: {question}\n\n"
            f"Permitted Verified Evidence:\n{evidence_str}\n\n"
        )

        if retry_instruction:
            prompt += f"RETRY INSTRUCTION: {retry_instruction}\n\n"

        prompt += f"{schema_rule}\n"

        if self.llm_invoker:
            raw_output = self.llm_invoker(prompt)
            return self._parse_output(raw_output, evidence_package)
        else:
            # Fallback heuristic generation for testing/mock environment
            return self._synthetic_fallback(question, evidence_package)

    def _parse_output(self, raw_output: Any, evidence_package: List[Dict[str, Any]]) -> GenerationContract:
        """Extracts and parses JSON object from LLM response, with fallback sentence extraction."""
        if not raw_output:
            return self._synthetic_fallback("", evidence_package)

        # Handle AIMessage or LangChain response objects safely
        if hasattr(raw_output, "content"):
            raw_output_str = str(raw_output.content)
        else:
            raw_output_str = str(raw_output or "")

        if not raw_output_str.strip():
            return self._synthetic_fallback("", evidence_package)

        # 1. Try stripping codeblock fences first
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_output_str.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

        # 2. Try direct JSON parsing
        try:
            data = json.loads(cleaned)
            contract = GenerationContract.model_validate(data)
            if contract.claims:
                return contract
        except Exception:
            pass

        # 3. Try regex matching first JSON object block
        json_match = re.search(r"(\{[\s\S]*\})", raw_output_str)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                contract = GenerationContract.model_validate(data)
                if contract.claims:
                    return contract
            except Exception:
                pass

        # 4. Fallback: LLM generated prose text instead of JSON; parse sentences into atomic claims
        all_eids = [
            item.get("metadata", {}).get("chunk_uid") or f"E{idx+1}"
            for idx, item in enumerate(evidence_package)
        ]
        
        # Split prose into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_output_str) if s.strip() and len(s.strip()) > 10]
        claims = []
        for idx, s in enumerate(sentences[:5]):
            claims.append(
                ClaimObject(
                    claim_id=f"C{idx+1}",
                    order=idx + 1,
                    text=s,
                    evidence_ids=all_eids
                )
            )

        if claims:
            return GenerationContract(claims=claims)

        return self._synthetic_fallback("", evidence_package)

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
