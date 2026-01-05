# application/use_cases/evaluate_analysis.py
from dataclasses import dataclass
from typing import Dict

from domain.models.evaluation import RubricCategory, RubricScore
from application.use_cases._shared import EvaluationContext
from application.use_cases.score_rubrics import (
    ScoreRubricsCommand,
    ScoreRubricsUseCase,
)


@dataclass(frozen=True)
class EvaluateAnalysisCommand:
    context: EvaluationContext


class EvaluateAnalysisUseCase:
    def __init__(
        self,
        score_rubrics_use_case: ScoreRubricsUseCase,
    ) -> None:
        self._score_rubrics = score_rubrics_use_case

    def execute(
        self, command: EvaluateAnalysisCommand
    ) -> Dict[RubricCategory, RubricScore]:
        score_command = ScoreRubricsCommand(
            context=command.context
        )
        return self._score_rubrics.execute(score_command)
