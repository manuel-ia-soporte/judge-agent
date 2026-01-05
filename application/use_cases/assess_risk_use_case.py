# application/use_cases/assess_risk_use_case.py
from dataclasses import dataclass
from typing import Dict, Any

from domain.services.operational_analysis_service import OperationalAnalysisService


@dataclass(frozen=True)
class AssessRiskCommand:
    financial_metrics: Dict[str, float]


class AssessRiskUseCase:
    def __init__(
        self,
        operational_service: OperationalAnalysisService,
    ) -> None:
        self._operational_service = operational_service

    def execute(self, command: AssessRiskCommand) -> Dict[str, Any]:
        return self._operational_service.evaluate(
            command.financial_metrics
        )
