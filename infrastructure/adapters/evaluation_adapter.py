# infrastructure/adapters/evaluation_adapter.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from domain.models.evaluation import Evaluation, RubricEvaluation
from contracts.evaluation_contracts import EvaluationRequest, EvaluationResult, RubricScore
from domain.services.evaluation_service import EvaluationOrchestrator


class EvaluationAdapter:
    """Adapter between domain models and external contracts"""

    @staticmethod
    def request_to_domain(request: EvaluationRequest) -> Evaluation:
        """Convert evaluation request to domain entity"""
        return Evaluation(
            evaluation_id=f"eval_{request.analysis_id}_{datetime.utcnow().timestamp()}",
            analysis_id=request.analysis_id,
            agent_id=request.agent_id,
            metadata={
                "request": request.dict(exclude={"analysis_content", "source_documents"}),
                "received_at": datetime.utcnow().isoformat()
            }
        )

    @staticmethod
    def domain_to_result(evaluation: Evaluation) -> EvaluationResult:
        """Convert domain entity to evaluation result"""

        # Convert rubric evaluations
        rubric_scores = {}
        for rubric_name, rubric_eval in evaluation.rubric_evaluations.items():
            rubric_scores[rubric_name] = RubricScore(
                rubric=rubric_eval.rubric_name,
                score=rubric_eval.score,
                passed=rubric_eval.is_passed,
                feedback=rubric_eval.feedback,
                evidence=rubric_eval.evidence,
                confidence=rubric_eval.confidence_score
            )

        # Generate recommendations if not already present
        if not evaluation.recommendations and evaluation.rubric_evaluations:
            from domain.models.finance import FinancialAnalysis
            dummy_analysis = FinancialAnalysis(
                analysis_id=evaluation.analysis_id,
                agent_id=evaluation.agent_id,
                company_ticker="",
                analysis_date=datetime.utcnow(),
                content="",
                metrics_used=[],
                source_documents=[],
                conclusions=[],
                risks_identified=[]
            )
            recommendations = EvaluationOrchestrator.generate_recommendations(
                evaluation.rubric_evaluations, dummy_analysis
            )
            evaluation.recommendations = recommendations

        # Check consistency
        is_consistent, warnings = EvaluationOrchestrator.validate_evaluation_consistency(
            evaluation.rubric_evaluations
        )
        if warnings and not evaluation.warnings:
            evaluation.warnings = warnings

        return EvaluationResult(
            evaluation_id=evaluation.evaluation_id,
            request_id=evaluation.analysis_id,
            agent_id=evaluation.agent_id,
            timestamp=evaluation.created_at,
            overall_score=evaluation.overall_score,
            passed=evaluation.is_passed,
            rubric_scores=rubric_scores,
            recommendations=evaluation.recommendations,
            warnings=evaluation.warnings,
            metadata=evaluation.metadata
        )

    @staticmethod
    def rubric_to_domain(rubric_name: str, score_data: Dict[str, Any]) -> RubricEvaluation:
        """Convert rubric score data to domain entity"""
        return RubricEvaluation(
            rubric_name=rubric_name,
            score=score_data.get("score", 0.0),
            is_passed=score_data.get("passed", False),
            feedback=score_data.get("feedback", ""),
            evidence=score_data.get("evidence", []),
            confidence_score=score_data.get("confidence", 1.0)
        )

    @staticmethod
    def merge_evaluations(
            primary: Evaluation,
            secondary: Evaluation,
            weights: Dict[str, float] = None
    ) -> Evaluation:
        """Merge two evaluations"""
        if weights is None:
            weights = {"primary": 0.7, "secondary": 0.3}

        merged = Evaluation(
            evaluation_id=f"merged_{primary.evaluation_id}",
            analysis_id=primary.analysis_id,
            agent_id=primary.agent_id
        )

        # Merge rubric evaluations
        all_rubrics = set(primary.rubric_evaluations.keys()) | set(secondary.rubric_evaluations.keys())

        for rubric in all_rubrics:
            if rubric in primary.rubric_evaluations and rubric in secondary.rubric_evaluations:
                # Weighted average
                primary_score = primary.rubric_evaluations[rubric]
                secondary_score = secondary.rubric_evaluations[rubric]

                weighted_score = (
                        primary_score.score * weights["primary"] +
                        secondary_score.score * weights["secondary"]
                )

                merged_eval = RubricEvaluation(
                    rubric_name=rubric,
                    score=weighted_score,
                    is_passed=weighted_score >= 1.0,
                    feedback=f"Merged: {primary_score.feedback} | {secondary_score.feedback}",
                    evidence=primary_score.evidence + secondary_score.evidence,
                    confidence_score=(primary_score.confidence_score + secondary_score.confidence_score) / 2
                )

                merged.add_rubric_evaluation(merged_eval)
            elif rubric in primary.rubric_evaluations:
                merged.add_rubric_evaluation(primary.rubric_evaluations[rubric])
            else:
                merged.add_rubric_evaluation(secondary.rubric_evaluations[rubric])

        # Merge recommendations and warnings
        merged.recommendations = list(set(primary.recommendations + secondary.recommendations))
        merged.warnings = list(set(primary.warnings + secondary.warnings))

        merged.complete_evaluation()
        return merged

    @staticmethod
    def normalize_scores(
            evaluation: Evaluation,
            target_min: float = 0.0,
            target_max: float = 2.0
    ) -> Evaluation:
        """Normalize scores to target range"""
        normalized = Evaluation(
            evaluation_id=f"normalized_{evaluation.evaluation_id}",
            analysis_id=evaluation.analysis_id,
            agent_id=evaluation.agent_id
        )

        if not evaluation.rubric_evaluations:
            return normalized

        # Find current min and max
        scores = [eval.score for eval in evaluation.rubric_evaluations.values()]
        current_min = min(scores)
        current_max = max(scores)

        # Normalize each rubric
        for rubric_name, rubric_eval in evaluation.rubric_evaluations.items():
            if current_max > current_min:  # Avoid division by zero
                normalized_score = (
                                           (rubric_eval.score - current_min) / (current_max - current_min)
                                   ) * (target_max - target_min) + target_min
            else:
                normalized_score = (target_min + target_max) / 2

            normalized_eval = RubricEvaluation(
                rubric_name=rubric_name,
                score=normalized_score,
                is_passed=normalized_score >= 1.0,
                feedback=f"Normalized: {rubric_eval.feedback}",
                evidence=rubric_eval.evidence,
                confidence_score=rubric_eval.confidence_score
            )

            normalized.add_rubric_evaluation(normalized_eval)

        normalized.complete_evaluation()
        return normalized

    @staticmethod
    def create_summary_report(evaluation: Evaluation) -> Dict[str, Any]:
        """Create summary report from evaluation"""
        if not evaluation.rubric_evaluations:
            return {"error": "No rubric evaluations"}

        # Calculate statistics
        scores = [eval.score for eval in evaluation.rubric_evaluations.values()]
        passed_count = sum(1 for eval in evaluation.rubric_evaluations.values() if eval.is_passed)

        # Group by score ranges
        score_ranges = {
            "excellent": sum(1 for s in scores if s >= 1.8),
            "good": sum(1 for s in scores if 1.5 <= s < 1.8),
            "fair": sum(1 for s in scores if 1.0 <= s < 1.5),
            "poor": sum(1 for s in scores if s < 1.0)
        }

        # Identify strengths and weaknesses
        strengths = [
            rubric for rubric, eval in evaluation.rubric_evaluations.items()
            if eval.score >= 1.8
        ]

        weaknesses = [
            rubric for rubric, eval in evaluation.rubric_evaluations.items()
            if eval.score < 1.0
        ]

        return {
            "summary": {
                "overall_score": evaluation.overall_score,
                "passed": evaluation.is_passed,
                "rubrics_evaluated": len(evaluation.rubric_evaluations),
                "rubrics_passed": passed_count,
                "pass_rate": (passed_count / len(evaluation.rubric_evaluations)) * 100,
                "score_distribution": score_ranges
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "key_recommendations": evaluation.recommendations[:3],  # Top 3
            "critical_warnings": [w for w in evaluation.warnings if "critical" in w.lower()],
            "timestamp": evaluation.created_at.isoformat(),
            "evaluation_duration": (
                (evaluation.completed_at - evaluation.created_at).total_seconds()
                if evaluation.completed_at else None
            )
        }