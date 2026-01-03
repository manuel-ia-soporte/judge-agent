# domain/services/evaluation_service.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from domain.models.evaluation import Evaluation, RubricEvaluation
from domain.models.finance import FinancialAnalysis
import statistics


@dataclass
class EvaluationOrchestrator:
    """Domain service for orchestrating evaluations"""

    @staticmethod
    def calculate_weighted_score(
            rubric_scores: Dict[str, RubricEvaluation],
            weights: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score"""
        total_weight = 0.0
        weighted_sum = 0.0

        for rubric_name, rubric_eval in rubric_scores.items():
            weight = weights.get(rubric_name, 1.0)
            weighted_sum += rubric_eval.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def determine_passed_status(
            rubric_scores: Dict[str, RubricEvaluation],
            overall_score: float,
            required_rubrics: List[str],
            min_required_score: float = 1.5
    ) -> Tuple[bool, List[str]]:
        """Determine if evaluation passed with feedback"""
        failed_rubrics = []

        # Check required rubrics
        for rubric in required_rubrics:
            if rubric in rubric_scores and not rubric_scores[rubric].is_passed:
                failed_rubrics.append(rubric)

        # Check overall score
        if overall_score < min_required_score:
            failed_rubrics.append("overall_score")

        return len(failed_rubrics) == 0, failed_rubrics

    @staticmethod
    def generate_recommendations(
            rubric_scores: Dict[str, RubricEvaluation],
            analysis: FinancialAnalysis
    ) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []

        # Check factual accuracy
        if "factual_accuracy" in rubric_scores:
            score = rubric_scores["factual_accuracy"].score
            if score < 1.0:
                recommendations.append(
                    "Verify numerical values against source documents"
                )

        # Check source fidelity
        if "source_fidelity" in rubric_scores:
            score = rubric_scores["source_fidelity"].score
            if score < 1.0:
                recommendations.append(
                    "Include specific citations for key facts and figures"
                )

        # Check risk awareness
        if "risk_awareness" in rubric_scores:
            score = rubric_scores["risk_awareness"].score
            if score < 1.0:
                recommendations.append(
                    "Include discussion of material risks from Item 1A"
                )

        # Check completeness
        if "completeness" in rubric_scores:
            score = rubric_scores["completeness"].score
            if score < 1.0:
                recommendations.append(
                    "Address all aspects of the prompt or question"
                )

        # General recommendations
        if len(analysis.risks_identified) < 3:
            recommendations.append(
                "Identify at least 3 key risks from the filings"
            )

        if not analysis.assumptions:
            recommendations.append(
                "Explicitly state assumptions made in the analysis"
            )

        return list(set(recommendations))  # Remove duplicates

    @staticmethod
    def calculate_confidence_score(
            rubric_scores: Dict[str, RubricEvaluation]
    ) -> float:
        """Calculate overall confidence score"""
        if not rubric_scores:
            return 0.0

        # Use average of rubric confidence scores
        confidences = [eval.confidence_score for eval in rubric_scores.values()]
        return statistics.mean(confidences) if confidences else 0.0

    @staticmethod
    def validate_evaluation_consistency(
            rubric_scores: Dict[str, RubricEvaluation]
    ) -> Tuple[bool, List[str]]:
        """Validate consistency between rubric scores"""
        warnings = []

        # Check for conflicting scores
        high_accuracy = rubric_scores.get("factual_accuracy", None)
        low_source_fidelity = rubric_scores.get("source_fidelity", None)

        if (high_accuracy and high_accuracy.score > 1.5 and
                low_source_fidelity and low_source_fidelity.score < 1.0):
            warnings.append(
                "High factual accuracy with low source fidelity - verify citations"
            )

        # Check for completeness vs materiality alignment
        completeness = rubric_scores.get("completeness", None)
        materiality = rubric_scores.get("materiality_relevance", None)

        if (completeness and completeness.score > 1.5 and
                materiality and materiality.score < 1.0):
            warnings.append(
                "Complete but not material - focus on key information"
            )

        return len(warnings) == 0, warnings