# domain/services/operational_analysis_service.py
from domain.models.entities import SECDocument


class OperationalAnalysisService:
    """
    Domain service responsible for operational efficiency.
    """

    def extract_operational_metrics(self, documents: list[SECDocument]) -> dict[str, float]:
        efficiency = 0.0
        for doc in documents:
            if "cost reduction" in doc.content.lower():
                efficiency += 0.1
        return {"efficiency_score": min(efficiency, 1.0)}

    def analyze_operational_efficiency(self, metrics: dict[str, float]) -> float:
        return metrics.get("efficiency_score", 0.0)

    def calculate_working_capital_metrics(self, documents: list[SECDocument]) -> dict[str, float]:
        return {"working_capital_ratio": 1.3 if documents else 0.0}
