# contracts/evaluation_contracts.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class RubricCategory(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_FIDELITY = "source_fidelity"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    FINANCIAL_REASONING = "financial_reasoning"
    MATERIALITY_RELEVANCE = "materiality_relevance"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    TEMPORAL_VALIDITY = "temporal_validity"
    RISK_AWARENESS = "risk_awareness"
    CLARITY_INTERPRETABILITY = "clarity_interpretability"
    UNCERTAINTY_HANDLING = "uncertainty_handling"
    ACTIONABILITY = "actionability"


class ScoringScale(str, Enum):
    ZERO_TO_TWO = "0-2"
    PASS_FAIL = "pass/fail"
    PERCENTAGE = "percentage"
    CUSTOM = "custom"


class EvaluationRequest(BaseModel):
    """Contract for evaluation requests"""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    agent_id: str = Field(..., description="ID of the agent being evaluated")
    analysis_content: str = Field(..., description="The financial analysis text to evaluate")
    source_documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source SEC documents used in analysis"
    )
    rubrics_to_evaluate: List[RubricCategory] = Field(
        default_factory=lambda: list(RubricCategory),
        description="Which rubrics to evaluate"
    )
    scoring_scale: ScoringScale = Field(default=ScoringScale.ZERO_TO_TWO)
    minimum_threshold: Optional[float] = Field(default=1.0, ge=0, le=2)
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('source_documents')
    @classmethod
    def validate_source_documents(cls, v):
        # Accept any dict structure for flexibility in tests
        return v


class RubricScore(BaseModel):
    """Individual rubric score"""
    rubric: RubricCategory
    score: float = Field(..., ge=0, le=2)
    passed: bool = Field(...)
    feedback: str = Field(...)
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)


class EvaluationResult(BaseModel):
    """Complete evaluation result"""
    evaluation_id: str = Field(...)
    request_id: str = Field(...)
    agent_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    overall_score: float = Field(..., ge=0, le=2)
    passed: bool = Field(...)
    rubric_scores: Dict[RubricCategory, RubricScore] = Field(...)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class A2AMessage(BaseModel):
    """A2A protocol message contract"""
    message_id: str = Field(...)
    sender_id: str = Field(...)
    receiver_id: str = Field(...)
    message_type: str = Field(...)  # Allow any message type
    content: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)