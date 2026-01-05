# agents/finance_agent/analyzers/operational_analyzer.py
from typing import Dict, Any


class OperationalAnalyzer:
    def analyze(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        return self._extract_financial_metrics(analysis_result)

    @staticmethod
    def _extract_financial_metrics(
        analysis_result: Dict[str, Any]
    ) -> Dict[str, float]:
        metrics = analysis_result.get("financial_metrics", {})
        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
