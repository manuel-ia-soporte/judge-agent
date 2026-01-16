# agents/judge_agent/judge_agent.py
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from domain.models.evaluation import RubricCategory, RubricScore
from agents.judge_agent.rubrics_evaluator import RubricEvaluator
from application.use_cases.evaluate_analysis import (
    EvaluateAnalysisCommand,
    EvaluateAnalysisUseCase,
)
from application.use_cases._shared import EvaluationContext, EvaluationAssumptions
from contracts.evaluation_contracts import EvaluationRequest


@dataclass
class JudgeCapabilities:
    """Capabilities of the judge agent."""
    supported_rubrics: List[str] = field(default_factory=lambda: [
        "factual_accuracy", "source_fidelity", "regulatory_compliance",
        "financial_reasoning", "materiality_relevance", "completeness",
        "consistency", "temporal_validity", "risk_awareness",
        "clarity_interpretability", "uncertainty_handling", "actionability"
    ])
    max_concurrent_evaluations: int = 10
    supports_batch: bool = True
    supports_async: bool = True


@dataclass
class JudgeMetrics:
    """Metrics for the judge agent."""
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    average_score: float = 0.0
    total_processing_time: float = 0.0
    _scores: List[float] = field(default_factory=list)

    def record_evaluation(self, score: float, processing_time: float = 0.0) -> None:
        """Record a completed evaluation."""
        self.evaluations_completed += 1
        self._scores.append(score)
        self.average_score = sum(self._scores) / len(self._scores)
        self.total_processing_time += processing_time

    def record_failure(self) -> None:
        """Record a failed evaluation."""
        self.evaluations_failed += 1


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
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._evaluate = evaluate_use_case
        self._evaluator = RubricEvaluator()
        self.is_active: bool = True
        self.agent_id = agent_id
        self.configuration = configuration or {}
        self._status = "active"
        self._queue_size = 0
        self.capabilities = JudgeCapabilities()
        self.metrics = JudgeMetrics()

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
        """Async evaluation for benchmark integration."""
        import uuid
        import time

        start_time = time.time()

        # Extract signals from the analysis content
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
        total_score = 0.0
        count = 0

        for category, score_obj in evaluated.items():
            cat_name = category.value if hasattr(category, "value") else str(category)
            if isinstance(score_obj, RubricScore):
                score_val = float(score_obj.score)
                rubric_scores[cat_name] = {
                    "score": score_val,
                    "rationale": score_obj.rationale,
                    "confidence": 1.0,
                }
            else:
                score_val = float(score_obj) if isinstance(score_obj, (int, float)) else 1.0
                rubric_scores[cat_name] = {
                    "score": score_val,
                    "rationale": "Evaluated by judge agent",
                    "confidence": 1.0,
                }
            total_score += score_val
            count += 1

        overall_score = total_score / count if count > 0 else 0.0

        recommendations: List[str] = []
        warnings: List[str] = []

        # Generate recommendations based on low scores
        for cat_name, score_data in rubric_scores.items():
            if score_data["score"] < 1.0:
                recommendations.append(f"Improve {cat_name.replace('_', ' ')}")

        # Add warnings for missing sources
        if not request.source_documents:
            warnings.append("No source documents provided for verification")

        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.record_evaluation(overall_score, processing_time)

        return {
            "evaluation_id": f"eval_{uuid.uuid4().hex[:8]}",
            "agent_id": request.agent_id,
            "analysis_id": request.analysis_id,
            "overall_score": overall_score,
            "passed": overall_score >= 1.0,
            "rubric_scores": rubric_scores,
            "recommendations": recommendations,
            "warnings": warnings,
        }

    async def batch_evaluate(self, requests: List[EvaluationRequest]) -> List[Dict[str, Any]]:
        """Evaluate multiple analyses in batch."""
        results = []
        for request in requests:
            result = await self.evaluate(request)
            results.append(result)
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "status": self._status,
            "is_active": self.is_active,
            "metrics": {
                "evaluations_completed": self.metrics.evaluations_completed,
                "evaluations_failed": self.metrics.evaluations_failed,
                "average_score": self.metrics.average_score,
            },
            "queue_size": self._queue_size,
            "capabilities": {
                "supported_rubrics": self.capabilities.supported_rubrics,
                "supports_batch": self.capabilities.supports_batch,
            },
        }

    async def stop(self) -> None:
        """Stop the agent."""
        self.is_active = False
        self._status = "stopped"

    def list_capabilities(self) -> Dict[str, Any]:
        """List agent capabilities."""
        return {
            "evaluate": {
                "description": "Evaluate financial analyses",
                "rubrics": self.capabilities.supported_rubrics,
            },
            "batch_evaluate": {
                "description": "Batch evaluate multiple analyses",
                "max_batch_size": self.capabilities.max_concurrent_evaluations,
            },
        }

    def _extract_signals_from_analysis(
        self, request: EvaluationRequest
    ) -> Dict[str, float]:
        """Extract evaluation signals from analysis content."""
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
