# domain/services/evaluation_service.py
from typing import Dict, List, Any
from domain.models.evaluation import RubricCategory, RubricScore, Evaluation, EvaluationStatus, RubricEvaluation


class EvaluationService:
    def aggregate(self, scores: Dict[RubricCategory, RubricScore]) -> float:
        if not scores:
            return 0.0
        return sum(s.score for s in scores.values()) / len(scores)


class EvaluationOrchestrator:
    """Orchestrates the evaluation process."""

    def __init__(self, evaluation_service: EvaluationService = None):
        self._service = evaluation_service or EvaluationService()

    def create_evaluation(self, analysis_id: str, agent_id: str) -> Evaluation:
        """Create a new evaluation."""
        import uuid
        return Evaluation(
            evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
            analysis_id=analysis_id,
            agent_id=agent_id,
            status=EvaluationStatus.PENDING,
        )

    def evaluate(self, evaluation: Evaluation, scores: Dict[RubricCategory, RubricScore]) -> Evaluation:
        """Evaluate an analysis using the provided scores."""
        evaluation.status = EvaluationStatus.IN_PROGRESS

        for category, score in scores.items():
            rubric_eval = RubricEvaluation(
                rubric=category,
                score=float(score.score),
                rationale=score.rationale,
            )
            evaluation.add_rubric_evaluation(rubric_eval)

        evaluation.calculate_overall_score()
        evaluation.passed = evaluation.overall_score >= 50.0
        evaluation.status = EvaluationStatus.COMPLETED

        return evaluation
