# application/use_cases/evaluate_analysis.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import asyncio
import re

from domain.models.evaluation import RubricCategory, RubricScore, Evaluation, RubricEvaluation
from application.use_cases._shared import EvaluationContext
from application.use_cases.score_rubrics import ScoreRubricsUseCase
from domain.services.evaluation_service import EvaluationOrchestrator
from infrastructure.adapters.evaluation_adapter import EvaluationAdapter
from contracts.evaluation_contracts import EvaluationRequest, EvaluationResult


@dataclass(frozen=True)
class EvaluateAnalysisCommand:
    context: EvaluationContext


class EvaluateAnalysisUseCase:
    def __init__(
        self,
        mcp_client: Any = None,
        a2a_client: Any = None,
        sec_data_provider: Any = None,
        score_rubrics_use_case: Optional[ScoreRubricsUseCase] = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.a2a_client = a2a_client
        self.sec_data_provider = sec_data_provider
        self._score_rubrics = score_rubrics_use_case or ScoreRubricsUseCase(
            sec_data_provider=sec_data_provider
        )
        self._orchestrator = EvaluationOrchestrator()

    async def execute(self, request: EvaluationRequest) -> EvaluationResult:
        source_documents = await self._fetch_source_documents(request.source_documents)
        parsed_content = self._parse_analysis_content(request.analysis_content)
        expected_values = self._extract_expected_values(parsed_content, request.context)

        rubric_scores = await self._score_rubrics.execute(
            request.analysis_content,
            request.rubrics_to_evaluate,
            source_documents,
            expected_values,
        )

        evaluation = EvaluationAdapter.request_to_domain(request)
        for rubric_eval in rubric_scores.values():
            evaluation.add_rubric_evaluation(rubric_eval)

        evaluation.recommendations = self._build_recommendations(rubric_scores)
        evaluation.warnings = self._build_warnings(request, rubric_scores)
        evaluation.calculate_overall_score()
        evaluation.complete_evaluation()

        return self._to_result_contract(evaluation)

    def score_signals(
        self, command: EvaluateAnalysisCommand
    ) -> Dict[RubricCategory, RubricScore]:
        results: Dict[RubricCategory, RubricScore] = {}
        for name, value in command.context.signals.items():
            category = (
                RubricCategory(name)
                if name in {member.value for member in RubricCategory}
                else RubricCategory.FACTUAL_ACCURACY
            )
        normalized = max(0.0, min(2.0, float(value)))
        results[category] = RubricScore(
            score=int(normalized * 50),
            rationale=f"Derived from signal '{name}'",
        )
        return results

    async def _fetch_source_documents(
        self, source_refs: List[Dict[str, Any]]
    ) -> List[Any]:
        if not source_refs or not self.sec_data_provider:
            return source_refs or []

        documents: List[Any] = []
        fetcher = getattr(self.sec_data_provider, "fetch_document", None)
        if not fetcher:
            return source_refs

        for ref in source_refs:
            result = fetcher(ref)  # type: ignore[arg-type]
            if asyncio.iscoroutine(result):
                doc = await result
            else:
                doc = result
            documents.append(doc)
        return documents

    def _parse_analysis_content(self, content: str) -> Dict[str, Any]:
        metrics: List[Dict[str, Union[str, float]]] = []
        for match in re.finditer(r"(revenue|net income|eps|growth)[^$]*\$?([\d,\.]+)", content, re.IGNORECASE):
            metric_name = match.group(1).strip().lower()
            raw_value = match.group(2).replace(",", "") if match.group(2) else "0"
            try:
                value = float(raw_value)
            except ValueError:
                continue
            metrics.append({"metric": metric_name, "value": value})

        sentences = [s.strip() for s in re.split(r"[.!?]", content) if s.strip()]
        return {"metrics": metrics, "sentences": sentences}

    def _extract_expected_values(
        self, parsed_content: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        expected: Dict[str, Any] = {}
        for metric in parsed_content.get("metrics", []):
            name = str(metric.get("metric"))
            if name in {"revenue", "net income", "growth"}:
                expected[name] = metric.get("value")
        expected.update(context.get("expected_values", {}))
        return expected

    def _build_recommendations(
        self, rubric_scores: Dict[str, RubricEvaluation]
    ) -> List[str]:
        recommendations: List[str] = []
        for name, evaluation in rubric_scores.items():
            if not evaluation.is_passed:
                recommendations.append(
                    f"Improve {name.replace('_', ' ')}: {evaluation.feedback or 'add detail'}"
                )
        return recommendations

    def _build_warnings(
        self,
        request: EvaluationRequest,
        rubric_scores: Dict[str, RubricEvaluation],
    ) -> List[str]:
        warnings: List[str] = []
        if not request.source_documents:
            warnings.append("No source documents provided for verification")
        if rubric_scores.get("temporal_validity") and not rubric_scores["temporal_validity"].is_passed:
            warnings.append("Analysis may reference outdated information")
        return warnings

    def _to_result_contract(self, evaluation: Evaluation) -> EvaluationResult:
        return EvaluationAdapter.domain_to_result(evaluation)
