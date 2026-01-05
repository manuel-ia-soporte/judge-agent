from domain.services.strategic_analysis_service import StrategicAnalysisService
from domain.models.entities import SECDocument


class StrategicAnalyzer:
    def __init__(self, service: StrategicAnalysisService) -> None:
        self._service = service

    def analyze(self, documents: list[SECDocument]) -> dict:
        return {
            "strategic_position": self._service.analyze_strategic_position(documents),
            "competitive_advantage": self._service.assess_competitive_advantage(documents),
            "growth_strategy": self._service.analyze_growth_strategies(documents),
            "strategic_risks": self._service.identify_strategic_risks(documents),
            "innovation": self._service.assess_innovation_capability(documents),
            "market_position": self._service.analyze_market_position(documents),
        }
