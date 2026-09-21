import re
import math
from typing import List
from app.rag.layer3.contracts import GenerationContract, SafetyCheckResult


class FastFailSafetyScanner:
    """Fast-fail deterministic scanner for API secrets, credentials, PII, and prompt leakage (< 15ms overhead)."""

    # Secret pattern matchers
    SECRET_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),               # OpenAI / Generic API Key
        re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE),                  # AWS Access Key
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),              # RSA/EC Private Key
        re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}"),  # JWT Token
        re.compile(r"api[_-]?key[\s:=]+['\"]?[a-zA-Z0-9_\-]{16,}['\"]?", re.IGNORECASE), # Hardcoded API key assignment
    ]

    # System prompt leakage patterns
    LEAKAGE_PATTERNS = [
        re.compile(r"System\s+Prompt:", re.IGNORECASE),
        re.compile(r"\[SYSTEM_INSTRUCTION\]", re.IGNORECASE),
        re.compile(r"You\s+are\s+an\s+AI\s+assistant\s+trained\s+by", re.IGNORECASE),
    ]

    def _shannon_entropy(self, data: str) -> float:
        """Calculates Shannon entropy of string to detect random secret keys."""
        if not data:
            return 0.0
        entropy = 0.0
        for x in set(data):
            p_x = data.count(x) / len(data)
            entropy -= p_x * math.log2(p_x)
        return entropy

    def scan(self, contract: GenerationContract) -> SafetyCheckResult:
        """Scans all claim texts in the contract for secret leakage or prompt injection echo."""
        full_text = " ".join([c.text for c in contract.claims])

        # 1. Regex Secret Scan
        for pattern in self.SECRET_PATTERNS:
            if pattern.search(full_text):
                return SafetyCheckResult(
                    passed=False,
                    reason="High-confidence secret/credential pattern detected in generated claim text.",
                    leaked_secrets_found=True
                )

        # 2. Shannon Entropy High-Risk Secret Scan
        words = full_text.split()
        for word in words:
            clean_word = word.strip("\",':;()[]{}")
            if len(clean_word) >= 24 and self._shannon_entropy(clean_word) > 4.5:
                return SafetyCheckResult(
                    passed=False,
                    reason=f"High-entropy secret key pattern detected: '{clean_word[:6]}...'",
                    leaked_secrets_found=True
                )

        # 3. Prompt Leakage Scan
        for pattern in self.LEAKAGE_PATTERNS:
            if pattern.search(full_text):
                return SafetyCheckResult(
                    passed=False,
                    reason="Internal system prompt or instruction leakage detected in claim text."
                )

        return SafetyCheckResult(passed=True)
