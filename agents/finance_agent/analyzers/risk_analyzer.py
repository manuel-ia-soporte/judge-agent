# agents/finance_agent/analyzers/risk_analyzer.py

from typing import Dict, Any


class RiskAnalyzer:
    """
    Domain analyzer responsible ONLY for computing risk
    from already-extracted financial signals.
    """

    def analyze(self, financial_signals: Dict[str, float]) -> Dict[str, Any]:
        if not financial_signals:
            return {"risk_score": 0.0, "risk_level": "unknown"}

        score = sum(financial_signals.values()) / len(financial_signals)

        return {
            "risk_score": round(score, 2),
            "risk_level": "high" if score >= 0.7 else "medium",
        }
