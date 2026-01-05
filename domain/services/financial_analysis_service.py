# domain/services/financial_analysis_service.py
from typing import Dict, Any, List

from domain.models.entities import SECDocument


class FinancialAnalysisService:
    def analyze(self, documents: List[SECDocument]) -> Dict[str, Any]:
        statements = self._parse_financial_statements(documents)
        return {
            "revenue": statements.get("revenue"),
            "net_income": statements.get("net_income"),
            "assets": statements.get("assets"),
        }

    @staticmethod
    def _parse_financial_statements(
        documents: List[SECDocument]
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for doc in documents:
            text = doc.content.lower()
            if "revenue" in text:
                metrics["revenue"] = metrics.get("revenue", 0.0) + 1.0
            if "net income" in text:
                metrics["net_income"] = metrics.get("net_income", 0.0) + 1.0
            if "assets" in text:
                metrics["assets"] = metrics.get("assets", 0.0) + 1.0

        return metrics
