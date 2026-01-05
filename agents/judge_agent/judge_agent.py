# agents/judge_agent/judge_agent.py
from typing import Dict

from domain.models.evaluation import RubricCategory, RubricScore
from application.use_cases.evaluate_analysis import (
    EvaluateAnalysisCommand,
    EvaluateAnalysisUseCase,
)
from application.use_cases._shared import EvaluationContext, EvaluationAssumptions


class JudgeAgent:
    def __init__(self, evaluate_use_case: EvaluateAnalysisUseCase) -> None:
        self._evaluate = evaluate_use_case
        self.is_active: bool = True

    def judge(
        self, signals: Dict[str, float]
    ) -> Dict[RubricCategory, RubricScore]:
        context = EvaluationContext(
            signals=signals,
            assumptions=EvaluationAssumptions(),
        )
        command = EvaluateAnalysisCommand(context=context)
        return self._evaluate.execute(command)
