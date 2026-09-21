from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ClaimObject(BaseModel):
    """Atomic factual claim unit extracted during generation."""
    claim_id: str
    order: int
    text: str
    evidence_ids: List[str] = Field(default_factory=list)


class GenerationContract(BaseModel):
    """Structured generation output from the LLM."""
    claims: List[ClaimObject] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    """Verification result for an individual claim."""
    claim_id: str
    text: str
    cited_evidence_ids: List[str]
    status: Literal["ENTAILMENT", "CONTRADICTION", "INSUFFICIENT", "UNKNOWN"]
    confidence: float = 1.0
    reason: Optional[str] = None


class SafetyCheckResult(BaseModel):
    """Fast-fail safety and schema validation result."""
    passed: bool
    reason: Optional[str] = None
    leaked_secrets_found: bool = False
    invalid_evidence_ids: List[str] = Field(default_factory=list)


class RelevanceResult(BaseModel):
    """Relevance evaluation result between query and answer content."""
    passed: bool
    score: float
    reason: Optional[str] = None


class Layer3GateDecision(BaseModel):
    """Final release decision and audit record from Layer 3."""
    decision: Literal["PASS", "REJECT"]
    reconstructed_answer: Optional[str] = None
    verified_claims: List[ClaimObject] = Field(default_factory=list)
    failed_claims: List[ClaimObject] = Field(default_factory=list)
    failure_reason: Optional[str] = None
    retry_count: int = 0
    telemetry: Dict[str, Any] = Field(default_factory=dict)
