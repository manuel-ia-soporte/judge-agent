"""Adapters between domain evaluation objects and API contracts."""

import uuid
from typing import Dict, Any, Iterable

from domain.models.evaluation import Evaluation, RubricEvaluation, RubricCategory
from contracts.evaluation_contracts import (
    EvaluationRequest,
    EvaluationResult,
    RubricScore,
)


class EvaluationAdapter:
    """Utility helpers to translate evaluation objects between layers."""

    @staticmethod
    def request_to_domain(request: EvaluationRequest) -> Evaluation:
        evaluation = Evaluation(
            evaluation_id=f"eval_{uuid.uuid4().hex[:8]}",
            analysis_id=request.analysis_id,
            agent_id=request.agent_id,
        )
        evaluation.metadata["request"] = request.model_dump()
        return evaluation

    @staticmethod
    def domain_to_result(evaluation: Evaluation) -> EvaluationResult:
        rubric_scores: Dict[str, RubricScore] = {}
        for rubric_name, rubric_eval in evaluation.rubric_evaluations.items():
            category = EvaluationAdapter._to_rubric_category(rubric_name)
            rubric_scores[rubric_name] = RubricScore(
                rubric=category,
                score=float(rubric_eval.score),
                passed=rubric_eval.is_passed,
                feedback=rubric_eval.feedback,
                evidence=list(rubric_eval.evidence),
                confidence=rubric_eval.confidence_score,
            )

        return EvaluationResult(
            evaluation_id=evaluation.evaluation_id,
            request_id=evaluation.metadata.get("request", {}).get("analysis_id", evaluation.analysis_id),
            agent_id=evaluation.agent_id,
            overall_score=float(evaluation.overall_score),
            passed=evaluation.passed,
            rubric_scores=rubric_scores,
            recommendations=evaluation.recommendations,
            warnings=evaluation.warnings,
            metadata=evaluation.metadata,
        )

    @staticmethod
    def rubric_to_domain(rubric_name: str, score_data: Dict[str, Any]) -> RubricEvaluation:
        return RubricEvaluation(
            rubric_name=rubric_name,
            score=float(score_data.get("score", 0.0)),
            is_passed=bool(score_data.get("passed", False)),
            feedback=score_data.get("feedback", ""),
            evidence=list(score_data.get("evidence", [])),
            confidence_score=float(score_data.get("confidence", 1.0)),
        )

    @staticmethod
    def merge_evaluations(
        primary: Evaluation,
        secondary: Evaluation,
        weights: Dict[str, float],
    ) -> Evaluation:
        merged = Evaluation(
            evaluation_id=f"merged_{uuid.uuid4().hex[:6]}",
            analysis_id=primary.analysis_id,
            agent_id=primary.agent_id,
        )

        merged.metadata["sources"] = {
            "primary": primary.evaluation_id,
            "secondary": secondary.evaluation_id,
        }

        total_weight = max(1e-6, weights.get("primary", 1.0) + weights.get("secondary", 1.0))
        combined: Dict[str, Iterable[RubricEvaluation]] = {}

        for name, evaluation in primary.rubric_evaluations.items():
            combined.setdefault(name, []).append((evaluation, weights.get("primary", 1.0)))

        for name, evaluation in secondary.rubric_evaluations.items():
            combined.setdefault(name, []).append((evaluation, weights.get("secondary", 1.0)))

        for name, evaluations in combined.items():
            weighted_score = sum(ev.score * w for ev, w in evaluations) / total_weight
            passed = all(ev.is_passed for ev, _ in evaluations)
            feedback = " | ".join(ev.feedback for ev, _ in evaluations if ev.feedback)
            evidence: list = []
            for ev, _ in evaluations:
                evidence.extend(ev.evidence)
            merged.add_rubric_evaluation(
                RubricEvaluation(
                    rubric_name=name,
                    score=weighted_score,
                    is_passed=passed,
                    feedback=feedback,
                    evidence=evidence,
                    confidence_score=sum(ev.confidence_score for ev, _ in evaluations) / len(evaluations),
                )
            )

        merged.calculate_overall_score()
        merged.complete_evaluation()
        return merged

    @staticmethod
    def normalize_scores(evaluation: Evaluation, new_min: float, new_max: float) -> Evaluation:
        if new_min >= new_max:
            raise ValueError("new_max must be greater than new_min")

        normalized = Evaluation(
            evaluation_id=f"normalized_{evaluation.evaluation_id}",
            analysis_id=evaluation.analysis_id,
            agent_id=evaluation.agent_id,
        )

        old_min, old_max = 0.0, 2.0
        span = old_max - old_min or 1.0
        target_span = new_max - new_min

        for rubric_name, rubric_eval in evaluation.rubric_evaluations.items():
            scaled_score = ((rubric_eval.score - old_min) / span) * target_span + new_min
            normalized.add_rubric_evaluation(
                RubricEvaluation(
                    rubric_name=rubric_name,
                    score=scaled_score,
                    is_passed=rubric_eval.is_passed,
                    feedback=rubric_eval.feedback,
                    evidence=list(rubric_eval.evidence),
                    confidence_score=rubric_eval.confidence_score,
                )
            )

        normalized.recommendations = list(evaluation.recommendations)
        normalized.warnings = list(evaluation.warnings)
        normalized.calculate_overall_score()
        return normalized

    @staticmethod
    def create_summary_report(evaluation: Evaluation) -> Dict[str, Any]:
        strengths = [
            name
            for name, rubric in evaluation.rubric_evaluations.items()
            if rubric.score >= 1.5
        ]
        weaknesses = [
            name
            for name, rubric in evaluation.rubric_evaluations.items()
            if rubric.score < 1.0
        ]

        return {
            "summary": {
                "overall_score": evaluation.overall_score,
                "passed": evaluation.passed,
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "key_recommendations": evaluation.recommendations or [
                f"Improve {name.replace('_', ' ')}"
                for name in weaknesses
            ],
        }

    @staticmethod
    def _to_rubric_category(rubric_name: str) -> RubricCategory:
        if rubric_name in {member.value for member in RubricCategory}:
            return RubricCategory(rubric_name)
        try:
            return RubricCategory[rubric_name.upper()]
        except KeyError:
            # Fallback to factual accuracy to avoid crashes while preserving data
            return RubricCategory.FACTUAL_ACCURACY
