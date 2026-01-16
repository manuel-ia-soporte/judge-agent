# domain/models/evaluation.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class RubricCategory(Enum):
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


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RubricScore:
    score: int
    rationale: str


@dataclass
class RubricEvaluation:
    """Evaluation of a single rubric."""
    rubric_name: str = ""
    score: float = 0.0
    is_passed: bool = True
    feedback: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    # Legacy fields for compatibility
    rubric: RubricCategory = None
    max_score: float = 2.0
    rationale: str = ""
    confidence: float = 1.0

    @property
    def passed(self) -> bool:
        return self.is_passed


@dataclass
class Evaluation:
    """Complete evaluation of an analysis."""
    evaluation_id: str
    analysis_id: str
    agent_id: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    rubric_evaluations: Dict[str, RubricEvaluation] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = False
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    _threshold: float = 1.5

    @property
    def is_passed(self) -> bool:
        return self.passed

    def add_rubric_evaluation(self, rubric_eval: RubricEvaluation) -> None:
        name = rubric_eval.rubric_name
        self.rubric_evaluations[name] = rubric_eval
        self.calculate_overall_score()

    def calculate_overall_score(self) -> float:
        if not self.rubric_evaluations:
            return 0.0
        evals = self.rubric_evaluations.values() if isinstance(self.rubric_evaluations, dict) else self.rubric_evaluations
        total = sum(r.score for r in evals)
        count = len(list(evals))
        self.overall_score = total / count if count > 0 else 0.0
        self.passed = self.overall_score >= self._threshold
        return self.overall_score

    def complete_evaluation(self) -> None:
        """Mark the evaluation as completed."""
        self.status = EvaluationStatus.COMPLETED
        self.completed_at = datetime.now()

    def get_failed_rubrics(self) -> Dict[str, RubricEvaluation]:
        """Get all rubrics that failed."""
        return {
            name: rubric for name, rubric in self.rubric_evaluations.items()
            if not rubric.is_passed
        }
