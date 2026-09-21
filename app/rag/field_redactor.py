import re
from typing import Dict, List, Tuple
from loguru import logger


class FieldRedactor:
    """Inline PII/PHI Field Redaction Hook.
    
    Redacts sensitive pattern occurrences (SSNs, Credit Cards, API Keys, Email addresses)
    from evidence text before context assembly to enforce least privilege at the field level.
    """

    def __init__(self):
        # Compiled regular expressions for common sensitive data formats
        self.patterns: List[Tuple[str, re.Pattern, str]] = [
            ("SSN", re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"), "[REDACTED_SSN]"),
            ("CREDIT_CARD", re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"), "[REDACTED_CARD]"),
            ("API_KEY", re.compile(r"\b(?:api_key_[0-9a-zA-Z]{24,32}|SECRET_KEY_[0-9A-Z]{16}|token_[0-9a-zA-Z]{36})\b"), "[REDACTED_API_KEY]"),
            ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        ]

    def redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redacts sensitive data patterns in text and returns the redacted text and counts."""
        redacted_text = text
        counts: Dict[str, int] = {}

        for name, pattern, replacement in self.patterns:
            matches = pattern.findall(redacted_text)
            if matches:
                counts[name] = len(matches)
                redacted_text = pattern.sub(replacement, redacted_text)

        if counts:
            logger.debug(f"FieldRedactor redacted sensitive entries: {counts}")

        return redacted_text, counts
