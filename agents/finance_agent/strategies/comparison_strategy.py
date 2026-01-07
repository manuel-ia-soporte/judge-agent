# agents/finance_agent/strategies/comparison_strategy.py

from typing import Dict, Any, List


class ComparisonStrategy:
    """
    Strategy for comparing multiple analysis outputs.
    Used for peer benchmarking or multi-agent evaluation.
    """

    def compare(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not analyses:
            return {"companies_compared": 0}

        return {
            "companies_compared": len(analyses),
            "average_risk_score": sum(
                a["risk"]["risk_score"] for a in analyses
            ) / len(analyses),
        }
