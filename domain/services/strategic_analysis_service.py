# domain/services/strategic_analysis_service.py
from domain.models.entities import SECDocument


class StrategicAnalysisService:
    """
    Domain service responsible for strategic assessment
    based on SEC document evidence.
    """

    def analyze_strategic_position(self, documents: list[SECDocument]) -> dict[str, str]:
        if not documents:
            return {"position": "unknown"}

        for doc in documents:
            if "market leader" in doc.content.lower():
                return {"position": "leader"}

        return {"position": "challenger"}

    def assess_competitive_advantage(self, documents: list[SECDocument]) -> dict[str, str]:
        for doc in documents:
            if "patent" in doc.content.lower():
                return {"advantage": "strong"}

        return {"advantage": "moderate"}

    def analyze_growth_strategies(self, documents: list[SECDocument]) -> dict[str, str]:
        for doc in documents:
            if "acquisition" in doc.content.lower():
                return {"growth_strategy": "inorganic"}

        return {"growth_strategy": "organic"}

    def identify_strategic_risks(self, documents: list[SECDocument]) -> list[str]:
        risks: list[str] = []

        for doc in documents:
            if "competition" in doc.content.lower():
                risks.append("high_competition")

        return risks

    def assess_innovation_capability(self, documents: list[SECDocument]) -> dict[str, str]:
        for doc in documents:
            if "research and development" in doc.content.lower():
                return {"innovation": "high"}

        return {"innovation": "average"}

    def analyze_market_position(self, documents: list[SECDocument]) -> dict[str, str]:
        for doc in documents:
            if "top five" in doc.content.lower():
                return {"market_position": "top_5"}

        return {"market_position": "unranked"}
