# application/use_cases/score_rubrics.py
from dataclasses import dataclass
from typing import Dict

from domain.models.evaluation import RubricCategory, RubricScore
from domain.services.rubrics_service import RubricsService
from application.use_cases._shared import EvaluationContext


@dataclass(frozen=True)
class ScoreRubricsCommand:
    context: EvaluationContext


class ScoreRubricsUseCase:
    def __init__(self, rubrics_service: RubricsService) -> None:
        self._rubrics_service = rubrics_service

    def execute(
        self, command: ScoreRubricsCommand
    ) -> Dict[RubricCategory, RubricScore]:
        return self._rubrics_service.evaluate(
            command.context.signals
        )
