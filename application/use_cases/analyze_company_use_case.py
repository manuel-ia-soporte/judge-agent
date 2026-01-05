# application/use_cases/analyze_company_use_case.py
from dataclasses import dataclass
from typing import Dict, Any, List
from uuid import UUID, uuid4

from domain.models.entities import SECDocument
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService


@dataclass(frozen=True)
class AnalyzeCompanyCommand:
    company_cik: str
    sec_documents: List[SECDocument]


class AnalyzeCompanyUseCase:
    def __init__(
        self,
        financial_service: FinancialAnalysisService,
        operational_service: OperationalAnalysisService,
        strategic_service: StrategicAnalysisService,
    ) -> None:
        self._financial_service = financial_service
        self._operational_service = operational_service
        self._strategic_service = strategic_service

    def execute(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        analysis_id: UUID = uuid4()

        financial_metrics = self._financial_service.analyze(
            command.sec_documents
        )
        operational_risks = self._operational_service.evaluate(
            financial_metrics
        )
        strategic_outlook = self._strategic_service.assess(
            financial_metrics
        )

        return {
            "analysis_id": analysis_id,
            "company_cik": command.company_cik,
            "financial_metrics": financial_metrics,
            "operational_risks": operational_risks,
            "strategic_outlook": strategic_outlook,
        }
