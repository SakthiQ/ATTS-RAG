from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ClaimObject(BaseModel):
    """One checkable sentence of the draft answer, with the evidence it cites."""
    claim_id: str
    order: int
    text: str
    evidence_ids: List[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    """Verification result for an individual claim."""
    claim_id: str
    text: str
    cited_evidence_ids: List[str]
    status: Literal["ENTAILMENT", "CONTRADICTION", "INSUFFICIENT", "UNKNOWN"]
    confidence: float = 1.0
    reason: Optional[str] = None
    supporting_evidence_id: Optional[str] = None


class SafetyCheckResult(BaseModel):
    """Result of the deterministic secret / prompt-leak scan."""
    passed: bool
    reason: Optional[str] = None
    leaked_secrets_found: bool = False


class ChartPoint(BaseModel):
    label: str
    value: float


class ChartSpec(BaseModel):
    """A chart whose every label and value was found verbatim in the cited evidence."""
    type: Literal["bar", "line", "pie"] = "bar"
    title: str
    x_label: str = ""
    y_label: str = ""
    unit: str = ""
    data: List[ChartPoint]
    evidence_id: Optional[str] = None


class Layer3GateDecision(BaseModel):
    """Final release decision and audit record from Layer 3."""
    decision: Literal["PASS", "REJECT"]
    reconstructed_answer: Optional[str] = None
    intent: Optional[str] = None
    chart: Optional[ChartSpec] = None
    verified_claims: List[ClaimObject] = Field(default_factory=list)
    failed_claims: List[ClaimObject] = Field(default_factory=list)
    grounding_ratio: float = 0.0
    evidence_map: Dict[str, str] = Field(default_factory=dict)  # E1 -> chunk_uid
    failure_reason: Optional[str] = None
    retry_count: int = 0
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    # Evidence for the adaptive layers: what this answer revealed about the user's session
    safety_flagged: bool = False    # the draft leaked a secret or the system prompt
    contradictions: int = 0         # claims the evidence contradicts

