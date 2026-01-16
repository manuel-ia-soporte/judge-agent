# contracts/judge_contracts.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict


class JudgeCapabilities(BaseModel):
    """Judge agent capabilities contract"""
    agent_id: str = Field(...)
    supported_rubrics: List[str] = Field(...)
    max_concurrent_evaluations: int = Field(default=5)
    evaluation_timeout: int = Field(default=30, description="Seconds")
    requires_grounding: bool = Field(default=True)
    compliance_level: str = Field(default="strict")


class JudgeMetrics(BaseModel):
    """Performance metrics for judge agent"""
    evaluations_completed: int = Field(default=0)
    average_score: float = Field(default=0.0)
    false_positives: int = Field(default=0)
    false_negatives: int = Field(default=0)
    avg_processing_time: float = Field(default=0.0)
    uptime: float = Field(default=100.0)


class JudgeConfiguration(BaseModel):
    """Judge agent configuration"""
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "factual_accuracy": 0.25,
            "source_fidelity": 0.20,
            "regulatory_compliance": 0.15,
            "financial_reasoning": 0.10,
            "materiality_relevance": 0.10,
            "completeness": 0.05,
            "risk_awareness": 0.05,
            "uncertainty_handling": 0.05,
            "clarity_interpretability": 0.025,
            "consistency": 0.025,
            "temporal_validity": 0.025
        }
    )
    pass_threshold: float = Field(default=1.5)
    strict_mode: bool = Field(default=False)
    log_detailed: bool = Field(default=True)
    auto_calibrate: bool = Field(default=False)

    @field_validator('pass_threshold')
    @classmethod
    def validate_pass_threshold(cls, value: float) -> float:
        if not 0.5 <= value <= 2.0:
            raise ValueError("pass_threshold must be between 0.5 and 2.0")
        return value

    @model_validator(mode='after')
    def validate_scoring_weights(self) -> 'JudgeConfiguration':
        weights = self.scoring_weights or {}
        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"scoring weight '{name}' must be non-negative")
        return self