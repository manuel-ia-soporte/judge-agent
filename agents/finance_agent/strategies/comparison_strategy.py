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

        # Extract risk scores from risk_assessment
        risk_scores = []
        for a in analyses:
            risk_assessment = a.get("risk_assessment", {})
            if isinstance(risk_assessment, dict):
                risk_scores.append(risk_assessment.get("risk_score", 0.0))
            else:
                risk_scores.append(0.0)

        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

        # Extract metrics for comparison
        metrics_comparison = {}
        for a in analyses:
            cik = a.get("company_cik", "unknown")
            metrics = a.get("metrics", [])
            metrics_comparison[cik] = {
                m.get("name"): m.get("value") for m in metrics if isinstance(m, dict)
            }

        return {
            "companies_compared": len(analyses),
            "average_risk_score": avg_risk,
            "risk_scores": {a.get("company_cik", f"company_{i}"): risk_scores[i] for i, a in enumerate(analyses)},
            "metrics_by_company": metrics_comparison,
        }
