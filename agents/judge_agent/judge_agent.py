# agents/judge_agent/judge_agent.py
from typing import Any, Dict, Optional, Union

from domain.models.evaluation import RubricCategory, RubricScore
from agents.judge_agent.rubrics_evaluator import RubricEvaluator
from application.use_cases.evaluate_analysis import (
    EvaluateAnalysisCommand,
    EvaluateAnalysisUseCase,
)
from application.use_cases._shared import EvaluationContext, EvaluationAssumptions
from contracts.evaluation_contracts import EvaluationRequest


class JudgeAgent:
    """Judge agent for evaluating financial analyses.

    Supports both the legacy synchronous `judge()` method and the new async
    `evaluate()` method for benchmark integration.
    """

    def __init__(
        self,
        evaluate_use_case: Optional[EvaluateAnalysisUseCase] = None,
        *,
        agent_id: str = "judge_agent",
    ) -> None:
        self._evaluate = evaluate_use_case
        self._evaluator = RubricEvaluator()
        self.is_active: bool = True
        self.agent_id = agent_id

    def judge(
        self, signals: Dict[str, float]
    ) -> Dict[RubricCategory, RubricScore]:
        """Legacy synchronous evaluation method using signals."""
        if self._evaluate is None:
            raise RuntimeError("EvaluateAnalysisUseCase not configured")

        context = EvaluationContext(
            signals=signals,
            assumptions=EvaluationAssumptions(),
        )
        command = EvaluateAnalysisCommand(context=context)
        raw_scores = self._evaluate.execute(command)
        return self._evaluator.evaluate(raw_scores)

    async def evaluate(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Async evaluation for benchmark integration.

        Evaluates the analysis content against all rubrics and returns
        a dictionary with rubric_scores, recommendations, and warnings.

        Args:
            request: EvaluationRequest with analysis content and context.

        Returns:
            Dictionary with rubric_scores, recommendations, warnings.
        """
        # Extract signals from the analysis content
        # This is a simplified implementation that uses basic heuristics
        signals = self._extract_signals_from_analysis(request)

        # Use the rubrics evaluator to score
        if self._evaluate is not None:
            context = EvaluationContext(
                signals=signals,
                assumptions=EvaluationAssumptions(
                    confidence_threshold=0.7,
                    allow_estimates=True,
                    require_sources=bool(request.source_documents),
                ),
            )
            command = EvaluateAnalysisCommand(context=context)
            raw_scores = self._evaluate.execute(command)
            evaluated = self._evaluator.evaluate(raw_scores)
        else:
            # Fallback: evaluate directly without use case
            evaluated = self._evaluator.evaluate(signals)

        # Convert to output format
        rubric_scores: Dict[str, Dict[str, Any]] = {}
        for category, score_obj in evaluated.items():
            cat_name = category.value if hasattr(category, "value") else str(category)
            if isinstance(score_obj, RubricScore):
                rubric_scores[cat_name] = {
                    "score": score_obj.score,
                    "rationale": score_obj.rationale,
                    "confidence": 1.0,
                }
            else:
                rubric_scores[cat_name] = {
                    "score": int(score_obj) if isinstance(score_obj, (int, float)) else 1,
                    "rationale": "Evaluated by judge agent",
                    "confidence": 1.0,
                }

        recommendations: list[str] = []
        warnings: list[str] = []

        # Generate recommendations based on low scores
        for cat_name, score_data in rubric_scores.items():
            if score_data["score"] < 1.0:
                recommendations.append(f"Improve {cat_name.replace('_', ' ')}")

        # Add warnings for missing sources
        if not request.source_documents:
            warnings.append("No source documents provided for verification")

        return {
            "rubric_scores": rubric_scores,
            "recommendations": recommendations,
            "warnings": warnings,
        }

    def _extract_signals_from_analysis(
        self, request: EvaluationRequest
    ) -> Dict[str, float]:
        """Extract evaluation signals from analysis content.

        This is a simplified heuristic-based extraction. In production,
        this would use more sophisticated NLP or an LLM.
        """
        content = request.analysis_content.lower()
        signals: Dict[str, float] = {}

        # Factual accuracy: check for specific numbers and citations
        has_numbers = any(c.isdigit() for c in content)
        has_citations = any(
            marker in content
            for marker in ["source:", "according to", "as reported", "sec filing"]
        )
        signals["factual_accuracy"] = 1.5 if (has_numbers and has_citations) else 1.0

        # Source fidelity: check if sources are referenced
        has_source_docs = bool(request.source_documents)
        signals["source_fidelity"] = 1.5 if has_source_docs else 0.5

        # Completeness: check length and structure
        word_count = len(content.split())
        signals["completeness"] = min(2.0, word_count / 200.0)

        # Risk awareness: check for risk-related keywords
        risk_keywords = [
            "risk", "uncertainty", "volatility", "exposure", "threat",
            "challenge", "concern", "warning"
        ]
        risk_mentions = sum(1 for kw in risk_keywords if kw in content)
        signals["risk_awareness"] = min(2.0, 0.5 + risk_mentions * 0.3)

        # Clarity: simple heuristic based on sentence length
        sentences = content.count(".") + content.count("!") + content.count("?")
        avg_sentence_len = (word_count / sentences) if sentences > 0 else 50
        signals["clarity"] = 2.0 if avg_sentence_len < 25 else (1.0 if avg_sentence_len < 40 else 0.5)

        return signals

    async def evaluate_analysis(
        self, request: EvaluationRequest
    ) -> Dict[str, Any]:
        """Alias for evaluate() for backward compatibility."""
        return await self.evaluate(request)
