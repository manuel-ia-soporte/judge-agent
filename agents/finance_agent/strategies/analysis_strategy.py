# agents/finance_agent/strategies/analysis_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any

from application.use_cases.analyze_company_use_case import (
    AnalyzeCompanyCommand,
    AnalyzeCompanyUseCase,
)


class AnalysisStrategy(ABC):
    @abstractmethod
    def execute(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        ...


class FullAnalysisStrategy(AnalysisStrategy):
    def __init__(self, use_case: AnalyzeCompanyUseCase) -> None:
        self._use_case = use_case

    def execute(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        return self._use_case.execute(command)
