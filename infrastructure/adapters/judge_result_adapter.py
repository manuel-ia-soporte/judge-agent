"""Adapter to convert JudgeAgent results to contract format."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from contracts.evaluation_contracts import (
    EvaluationRequest,
    EvaluationResult,
    RubricCategory,
    RubricScore as ContractRubricScore,
)


def to_contract_evaluation_result(
    *,
    request: EvaluationRequest,
    judge_result: Dict[str, Any],
    started_at: datetime,
    processing_time_ms: int,
    judge_agent_id: str,
) -> EvaluationResult:
    """Convert JudgeAgent raw result to EvaluationResult contract.

    Args:
        request: The original evaluation request.
        judge_result: Raw result from JudgeAgent.evaluate().
        started_at: When evaluation started.
        processing_time_ms: Processing time in milliseconds.
        judge_agent_id: ID of the judge agent.

    Returns:
        EvaluationResult contract object.
    """
    rubric_scores: Dict[RubricCategory, ContractRubricScore] = {}
    total_score = 0.0
    count = 0

    # Extract rubric scores from judge_result
    raw_scores = judge_result.get("rubric_scores", {})
    for cat_name, score_data in raw_scores.items():
        try:
            category = RubricCategory(cat_name)
        except ValueError:
            # Skip unknown categories
            continue

        if isinstance(score_data, dict):
            score = float(score_data.get("score", 0))
            feedback = str(score_data.get("rationale", ""))
            evidence = score_data.get("evidence", [])
            confidence = float(score_data.get("confidence", 1.0))
        else:
            score = float(score_data)
            feedback = ""
            evidence = []
            confidence = 1.0

        rubric_scores[category] = ContractRubricScore(
            rubric=category,
            score=min(2.0, max(0.0, score)),  # Clamp to 0-2
            passed=score >= (request.minimum_threshold or 1.0),
            feedback=feedback,
            evidence=evidence if isinstance(evidence, list) else [],
            confidence=confidence,
        )

        total_score += score
        count += 1

    overall_score = (total_score / count) if count > 0 else 0.0
    threshold = request.minimum_threshold or 1.0
    passed = overall_score >= threshold

    recommendations = judge_result.get("recommendations", [])
    warnings = judge_result.get("warnings", [])

    return EvaluationResult(
        evaluation_id=uuid4().hex,
        request_id=request.analysis_id,
        agent_id=request.agent_id,
        timestamp=started_at,
        overall_score=round(overall_score, 3),
        passed=passed,
        rubric_scores=rubric_scores,
        recommendations=recommendations if isinstance(recommendations, list) else [],
        warnings=warnings if isinstance(warnings, list) else [],
        metadata={
            "judge_agent_id": judge_agent_id,
            "processing_time_ms": processing_time_ms,
        },
    )
