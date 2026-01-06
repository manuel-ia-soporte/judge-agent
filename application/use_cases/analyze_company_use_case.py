# application/use_cases/analyze_company_use_case.py

from application.ports.analysis_ports import (
    FinancialAnalysisPort,
    OperationalAnalysisPort,
    StrategicAnalysisPort,
)
from application.commands.analyze_company_command import AnalyzeCompanyCommand
from application.dtos.analysis_dtos import AnalysisResultDTO


class AnalyzeCompanyUseCase:
    """
    Orchestrates company analysis.
    This use case contains NO business logic – only coordination.
    """

    def __init__(
        self,
        financial_analysis: FinancialAnalysisPort,
        operational_analysis: OperationalAnalysisPort,
        strategic_analysis: StrategicAnalysisPort,
    ):
        self._financial = financial_analysis
        self._operational = operational_analysis
        self._strategic = strategic_analysis

    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        errors = command.validate()
        if errors:
            raise ValueError(f"Invalid AnalyzeCompanyCommand: {errors}")

        financial_result = await self._financial.analyze(command.company_cik)
        operational_result = await self._operational.evaluate(command.company_cik)
        strategic_result = await self._strategic.assess(command.company_cik)

        return AnalysisResultDTO(
            analysis_id=command.request_id or "generated",
            company_cik=command.company_cik,
            analysis_type=command.analysis_type.value,
            financial_assessment=financial_result,
            operational_assessment=operational_result,
            strategic_assessment=strategic_result,
            confidence_score=0.85,
        )
