import re
from typing import List
from app.rag.layer3.contracts import GenerationContract, RelevanceResult


class RelevanceEvaluator:
    """Evaluates semantic relevance between the user query and reconstructed claims."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def evaluate(self, question: str, contract: GenerationContract) -> RelevanceResult:
        """Evaluates whether claims directly address the user query."""
        if not contract.claims:
            return RelevanceResult(passed=False, score=0.0, reason="No claims to evaluate.")

        question_terms = set(re.findall(r"\w+", question.lower()))
        # Remove basic stopwords
        stopwords = {"what", "is", "the", "are", "how", "do", "does", "a", "an", "in", "on", "for", "to", "of", "and", "or"}
        query_keywords = question_terms - stopwords

        if not query_keywords:
            return RelevanceResult(passed=True, score=1.0)

        combined_text = " ".join([c.text for c in contract.claims]).lower()

        # Calculate keyword overlap score
        matches = sum(1 for kw in query_keywords if kw in combined_text)
        score = matches / max(len(query_keywords), 1)

        # Baseline threshold check
        passed = score >= self.threshold or matches >= 1

        return RelevanceResult(
            passed=passed,
            score=round(score, 2),
            reason=None if passed else f"Relevance score {score:.2f} below threshold {self.threshold}"
        )

