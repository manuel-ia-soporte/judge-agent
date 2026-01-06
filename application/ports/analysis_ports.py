# application/ports/analysis_ports.py
from typing import Protocol, Dict, Any


class FinancialAnalysisPort(Protocol):
    async def analyze(self, company_cik: str) -> Dict[str, Any]:
        ...


class OperationalAnalysisPort(Protocol):
    async def evaluate(self, company_cik: str) -> Dict[str, Any]:
        ...


class StrategicAnalysisPort(Protocol):
    async def assess(self, company_cik: str) -> Dict[str, Any]:
        ...
