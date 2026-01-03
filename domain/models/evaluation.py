# domain/models/evaluation.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RubricWeight:
    """Domain value object for rubric weights"""

    def __init__(self, rubric: str, weight: float):
        if not 0 <= weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {weight}")
        self.rubric = rubric
        self.weight = weight


@dataclass
class RubricEvaluation:
    """Domain entity for individual rubric evaluation"""
    rubric_name: str
    score: float
    is_passed: bool
    feedback: str
    evidence: List[str]
    confidence_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def adjust_score(self, adjustment: float) -> None:
        """Adjust score with bounds checking"""
        new_score = self.score + adjustment
        self.score = max(0, min(2, new_score))
        self._update_pass_status()

    def _update_pass_status(self) -> None:
        """Update pass status based on score"""
        self.is_passed = self.score >= 1.0  # Default threshold


@dataclass
class Evaluation:
    """Aggregate root for evaluation domain"""
    evaluation_id: str
    analysis_id: str
    agent_id: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    rubric_evaluations: Dict[str, RubricEvaluation] = field(default_factory=dict)
    overall_score: float = 0.0
    is_passed: bool = False
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def add_rubric_evaluation(self, rubric_eval: RubricEvaluation) -> None:
        """Add a rubric evaluation to the aggregate"""
        self.rubric_evaluations[rubric_eval.rubric_name] = rubric_eval
        self._recalculate_overall_score()

    def _recalculate_overall_score(self) -> None:
        """Recalculate overall weighted score"""
        if not self.rubric_evaluations:
            self.overall_score = 0.0
            return

        # Default equal weighting if not specified
        total_score = sum(eval.score for eval in self.rubric_evaluations.values())
        self.overall_score = total_score / len(self.rubric_evaluations)
        self.is_passed = self.overall_score >= 1.5  # Configurable threshold

    def complete_evaluation(self) -> None:
        """Mark evaluation as completed"""
        self.status = EvaluationStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def get_failed_rubrics(self) -> List[str]:
        """Get list of failed rubrics"""
        return [name for name, eval in self.rubric_evaluations.items()
                if not eval.is_passed]