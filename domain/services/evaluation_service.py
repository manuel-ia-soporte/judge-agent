# domain/services/evaluation_service.py
from typing import Dict, List, Any, Tuple
from domain.models.evaluation import RubricCategory, RubricScore, Evaluation, EvaluationStatus, RubricEvaluation
from domain.models.finance import FinancialAnalysis


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

    @staticmethod
    def calculate_weighted_score(
        rubric_scores: Dict[str, RubricEvaluation],
        weights: Dict[str, float],
    ) -> float:
        total_weight = 0.0
        weighted = 0.0
        for name, evaluation in rubric_scores.items():
            weight = weights.get(name, 1.0)
            weighted += evaluation.score * weight
            total_weight += weight
        return weighted / total_weight if total_weight else 0.0

    @staticmethod
    def determine_passed_status(
        rubric_scores: Dict[str, RubricEvaluation],
        overall_score: float,
        required_rubrics: List[str],
        min_required_score: float = 1.0,
    ) -> Tuple[bool, List[str]]:
        failed: List[str] = []
        for rubric in required_rubrics:
            result = rubric_scores.get(rubric)
            if not result or result.score < min_required_score:
                failed.append(rubric)
        if overall_score < min_required_score:
            failed.append("overall_score")
        return (len(failed) == 0, failed)

    @staticmethod
    def generate_recommendations(
        rubric_scores: Dict[str, RubricEvaluation],
        analysis: FinancialAnalysis,
    ) -> List[str]:
        recommendations: List[str] = []
        for name, evaluation in rubric_scores.items():
            if not evaluation.is_passed:
                if name == "factual_accuracy":
                    recommendations.append("Verify numerical values against authoritative filings")
                recommendations.append(
                    f"Improve {name.replace('_', ' ')}: {evaluation.feedback or 'add supporting evidence'}"
                )
        if not analysis.source_documents:
            recommendations.append("Provide source documents for verification")
        if not recommendations:
            recommendations.append("Maintain analysis quality; no critical issues detected")
        return recommendations

    @staticmethod
    def calculate_confidence_score(rubric_scores: Dict[str, RubricEvaluation]) -> float:
        if not rubric_scores:
            return 0.0
        return sum(r.confidence_score for r in rubric_scores.values()) / len(rubric_scores)

    @staticmethod
    def validate_evaluation_consistency(
        rubric_scores: Dict[str, RubricEvaluation]
    ) -> Tuple[bool, List[str]]:
        warnings: List[str] = []
        factual = rubric_scores.get("factual_accuracy")
        source = rubric_scores.get("source_fidelity")
        if factual and source and factual.is_passed and not source.is_passed:
            warnings.append("High factual accuracy with low source fidelity")
        return (len(warnings) == 0, warnings)
