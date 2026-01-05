# agents/finance_agent/analyzers/risk_analyzer.py
from typing import Dict


class RiskAnalyzer:
    def analyze(self, operational_risks: Dict[str, str]) -> int:
        severity_weights = {
            "low": 10,
            "medium": 50,
            "high": 90,
        }

        scores = [
            severity_weights.get(level, 50)
            for level in operational_risks.values()
        ]
        return int(sum(scores) / max(1, len(scores)))
