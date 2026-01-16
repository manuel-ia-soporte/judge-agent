# agents/finance_agent/analyzers/operational_analyzer.py
from typing import Dict, Any, List
from domain.models.entities import SECDocument


class OperationalAnalyzer:
    def analyze(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Analyze operational aspects from SEC documents."""
        efficiency_score = 0.75
        highlights = []

        for doc in documents:
            content = doc.content.lower()
            if "operating" in content:
                efficiency_score += 0.05
            if "efficiency" in content:
                highlights.append("Operational efficiency mentioned")
            if "supply chain" in content:
                highlights.append("Supply chain operations discussed")

        return {
            "efficiency_score": min(efficiency_score, 1.0),
            "operational_status": "healthy" if efficiency_score > 0.7 else "needs_attention",
            "highlights": highlights if highlights else ["Standard operations"],
            "documents_reviewed": len(documents),
        }
