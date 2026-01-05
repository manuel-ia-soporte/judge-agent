# agents/finance_agent/analyzers/financial_analyzer.py
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.models.entities import SECDocument


class FinancialAnalyzer:
    def __init__(self, service: FinancialAnalysisService) -> None:
        self._service = service

    def analyze(self, documents: list[SECDocument]) -> dict:
        metrics = self._service.extract_metrics(documents)
        ratios = self._service.calculate_ratios(metrics)

        return {
            "financial_metrics": metrics,
            "financial_ratios": ratios,
        }
