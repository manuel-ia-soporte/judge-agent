# agents/finance_agent/strategies/analysis_strategy.py

from abc import ABC, abstractmethod
from application.use_cases.analyze_company_use_case import (
    AnalyzeCompanyCommand,
    AnalyzeCompanyUseCase,
)
from application.dtos.analysis_dtos import AnalysisResultDTO


class AnalysisStrategy(ABC):
    """
    Strategy = HOW the analysis is executed.
    """

    @abstractmethod
    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        ...


class FullAnalysisStrategy(AnalysisStrategy):
    """
    Default strategy: full use-case execution.
    """

    def __init__(self, use_case: AnalyzeCompanyUseCase) -> None:
        self._use_case = use_case

    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        return await self._use_case.execute(command)
