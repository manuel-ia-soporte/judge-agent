# domain/services/financial_analysis_service.py
from typing import Dict
from domain.models.entities import SECDocument


class FinancialAnalysisService:
    """
    Domain service responsible for financial signal extraction.
    """

    def extract_metrics(self, documents: list[SECDocument]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for doc in documents:
            text = doc.content.lower()
            if "revenue" in text:
                metrics["revenue_growth"] = 0.10
            if "net income" in text:
                metrics["profitability"] = 0.15

        return metrics

    def calculate_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        return {
            "profit_margin": metrics.get("profitability", 0.0),
            "growth_ratio": metrics.get("revenue_growth", 0.0),
        }
