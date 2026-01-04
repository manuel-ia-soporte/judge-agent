# agents/finance_agent/analyzers/financial_analyzer.py
from typing import Dict, Any
from dataclasses import dataclass
from ....application.commands.analyze_company_command import AnalyzeCompanyCommand
from .analyzer_interface import Analyzer
from ....domain.services.financial_analysis_service import FinancialAnalysisService
from ....domain.repositories.sec_document_repository import SECDocumentRepository


@dataclass
class FinancialAnalyzer(Analyzer):
    """Financial analyzer implementation"""

    sec_repository: SECDocumentRepository
    financial_service: FinancialAnalysisService

    async def analyze(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        """Perform financial analysis"""

        # Fetch documents
        documents = await self.sec_repository.find_by_cik(
            cik=command.company_cik,
            filing_types=["10-K", "10-Q"],
            start_date=command.start_date,
            end_date=command.end_date
        )

        if not documents:
            return {"error": "No documents found"}

        # Extract metrics
        metrics = self.financial_service.extract_metrics(documents)

        # Calculate ratios
        ratios = self.financial_service.calculate_ratios(metrics)

        # Analyze trends
        trends = {}
        for metric_name in ["Revenue", "NetIncomeLoss", "Assets"]:
            trend = self.financial_service.analyze_trends(metrics, metric_name)
            if trend:
                trends[metric_name] = {
                    "trend": trend.trend,
                    "slope": trend.slope,
                    "volatility": trend.volatility
                }

        return {
            "analysis_type": "financial",
            "company_cik": command.company_cik,
            "metrics": [m.__dict__ for m in metrics],
            "ratios": [r.__dict__ for r in ratios],
            "trends": trends,
            "documents_analyzed": len(documents)
        }

    def can_handle(self, analysis_type: str) -> bool:
        """Check if it can handle financial analysis"""
        return analysis_type in ["financial", "comprehensive"]