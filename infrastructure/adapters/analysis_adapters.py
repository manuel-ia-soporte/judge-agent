# infrastructure/adapters/analysis_adapters.py
from typing import Dict, Any
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService

class FinancialAnalysisAdapter:
    def __init__(self, service: FinancialAnalysisService) -> None:
        self._service = service

    async def analyze(self, company_cik: str) -> Dict[str, Any]:
        # In real app: fetch docs → call service
        return {"financial_metrics": {}, "financial_ratios": {}}

class OperationalAnalysisAdapter:
    def __init__(self, service: OperationalAnalysisService) -> None:
        self._service = service

    async def evaluate(self, company_cik: str) -> Dict[str, Any]:
        return {"efficiency": 0.8}

class StrategicAnalysisAdapter:
    def __init__(self, service: StrategicAnalysisService) -> None:
        self._service = service

    async def assess(self, company_cik: str) -> Dict[str, Any]:
        return {"position": "challenger"}