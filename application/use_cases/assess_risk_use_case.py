# application/use_cases/assess_risk_use_case.py

from application.ports.analysis_ports import OperationalAnalysisPort
from application.dtos.risk_dtos import RiskAssessmentDTO


class AssessRiskUseCase:
    """
    Coordinates operational risk evaluation.
    """

    def __init__(self, operational_analysis: OperationalAnalysisPort):
        self._operational = operational_analysis

    async def execute(self, company_cik: str) -> RiskAssessmentDTO:
        operational_risk = await self._operational.evaluate(company_cik)

        return RiskAssessmentDTO(
            company_cik=company_cik,
            operational_risk_assessment=operational_risk,
            risk_level="medium",
            confidence_level=0.8,
        )
