# domain/services/operational_analysis_service.py
from enum import Enum
from typing import Dict, Any


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OperationalAnalysisService:
    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        risks = {}

        revenue = metrics.get("revenue", 0.0)
        if revenue < 1:
            risks["revenue"] = SeverityLevel.HIGH.value
        else:
            risks["revenue"] = SeverityLevel.LOW.value

        return risks
