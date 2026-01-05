# agents/finance_agent/core/finance_agent.py
from enum import Enum
from typing import Dict, Any

from agents.finance_agent.strategies.analysis_strategy import (
    AnalysisStrategy,
)
from application.use_cases.analyze_company_use_case import (
    AnalyzeCompanyCommand,
)


class AnalysisType(str, Enum):
    FULL = "full"


class FinanceAgent:
    def __init__(self, strategy: AnalysisStrategy) -> None:
        self._strategy = strategy
        self.is_active: bool = True

    def analyze(
        self, command: AnalyzeCompanyCommand
    ) -> Dict[str, Any]:
        return self._strategy.execute(command)
