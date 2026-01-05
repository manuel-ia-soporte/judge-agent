# domain/services/strategic_analysis_service.py
from typing import Dict, Any

from domain.services.operational_analysis_service import SeverityLevel


class StrategicAnalysisService:
    def assess(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        strategy = {}

        growth = indicators.get("growth", 0.0)
        if growth > 0.2:
            strategy["growth_outlook"] = SeverityLevel.LOW.value
        elif growth > 0:
            strategy["growth_outlook"] = SeverityLevel.MEDIUM.value
        else:
            strategy["growth_outlook"] = SeverityLevel.HIGH.value

        return strategy
