# application/use_cases/score_rubrics.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from domain.models.evaluation import RubricEvaluation, RubricCategory
from domain.models.finance import FinancialAnalysis
from domain.services.rubrics_service import RubricsService, RubricEvaluator


@dataclass(frozen=True)
class ScoreRubricsCommand:
    analysis_content: str
    rubrics: List[Union[RubricCategory, str]]
    source_documents: List[Any]
    expected_values: Optional[Dict[str, Any]] = None


class ScoreRubricsUseCase:
    """Scores rubric categories for a given analysis content."""

    def __init__(
        self,
        sec_data_provider: Any = None,
        rubrics_service: Optional[RubricsService] = None,
        rubric_evaluator: Optional[RubricEvaluator] = None,
    ) -> None:
        self.sec_data_provider = sec_data_provider
        self._rubrics_service = rubrics_service
        self._rubric_evaluator = rubric_evaluator or RubricEvaluator()

    async def execute(
        self,
        analysis_content: str,
        rubrics: List[Union[RubricCategory, str]],
        source_documents: List[Any],
        expected_values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, RubricEvaluation]:
        analysis = self._build_analysis(analysis_content, source_documents)
        rubric_list = rubrics or [cat.value for cat in RubricCategory]

        results: Dict[str, RubricEvaluation] = {}
        for rubric in rubric_list:
            rubric_name = rubric.value if isinstance(rubric, RubricCategory) else str(rubric)
            evaluation = self._score_rubric(
                rubric_name,
                analysis,
                source_documents,
                expected_values or {},
            )
            results[rubric_name] = evaluation
        return results

    async def calculate_composite_score(
        self,
        rubric_scores: Dict[str, RubricEvaluation],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        if not rubric_scores:
            return {
                "composite_score": 0.0,
                "passed": False,
                "passed_rubrics": [],
                "failed_rubrics": [],
            }

        weighted_total = 0.0
        total_weight = 0.0
        passed_rubrics: List[str] = []
        failed_rubrics: List[str] = []

        for name, evaluation in rubric_scores.items():
            weight = weights.get(name, 1.0)
            weighted_total += evaluation.score * weight
            total_weight += weight
            (passed_rubrics if evaluation.is_passed else failed_rubrics).append(name)

        composite_score = weighted_total / total_weight if total_weight else 0.0
        passed = composite_score >= 1.5 and not failed_rubrics

        return {
            "composite_score": composite_score,
            "passed": passed,
            "passed_rubrics": passed_rubrics,
            "failed_rubrics": failed_rubrics,
        }

    def _build_analysis(
        self, analysis_content: str, source_documents: List[Any]
    ) -> FinancialAnalysis:
        return FinancialAnalysis(
            analysis_id="temp_analysis",
            agent_id="judge_agent",
            company_ticker="",
            analysis_date=datetime.utcnow(),
            content=analysis_content,
            metrics_used=self._extract_metrics(analysis_content),
            source_documents=source_documents,
            conclusions=[],
            risks_identified=[],
            assumptions=[],
        )

    def _extract_metrics(self, content: str) -> List[str]:
        keywords = [
            "revenue",
            "net income",
            "gross margin",
            "operating margin",
            "eps",
            "debt",
        ]
        lowered = content.lower()
        return [kw for kw in keywords if kw in lowered]

    def _score_rubric(
        self,
        rubric_name: str,
        analysis: FinancialAnalysis,
        source_documents: List[Any],
        expected_values: Dict[str, Any],
    ) -> RubricEvaluation:
        name = rubric_name.lower()

        if name == RubricCategory.FACTUAL_ACCURACY.value:
            return self._rubric_evaluator.evaluate_factual_accuracy(
                analysis, expected_values
            )
        if name == RubricCategory.SOURCE_FIDELITY.value:
            return self._rubric_evaluator.evaluate_source_fidelity(
                analysis, source_documents
            )
        if name == RubricCategory.REGULATORY_COMPLIANCE.value:
            return self._rubric_evaluator.evaluate_regulatory_compliance(analysis)
        if name == RubricCategory.FINANCIAL_REASONING.value:
            return self._rubric_evaluator.evaluate_financial_reasoning(analysis)
        if name == RubricCategory.MATERIALITY_RELEVANCE.value:
            return self._rubric_evaluator.evaluate_materiality_relevance(analysis)
        if name == RubricCategory.COMPLETENESS.value:
            return self._rubric_evaluator.evaluate_completeness(analysis)
        if name == RubricCategory.CONSISTENCY.value:
            return self._rubric_evaluator.evaluate_consistency(analysis)
        if name == RubricCategory.TEMPORAL_VALIDITY.value:
            return self._rubric_evaluator.evaluate_temporal_validity(analysis)
        if name == RubricCategory.RISK_AWARENESS.value:
            return self._rubric_evaluator.evaluate_risk_awareness(analysis)
        if name == RubricCategory.CLARITY_INTERPRETABILITY.value:
            return self._rubric_evaluator.evaluate_clarity_interpretability(analysis)
        if name == RubricCategory.UNCERTAINTY_HANDLING.value:
            return self._rubric_evaluator.evaluate_uncertainty_handling(analysis)
        if name == RubricCategory.ACTIONABILITY.value:
            return self._rubric_evaluator.evaluate_actionability(analysis)

        return RubricEvaluation(
            rubric_name=name,
            score=1.0,
            is_passed=True,
            feedback="Default evaluation",
            evidence=[],
        )
