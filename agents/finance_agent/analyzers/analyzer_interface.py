# agents/finance_agent/analyzers/analyzer_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from application.commands.analyze_company_command import AnalyzeCompanyCommand


# Strategy Pattern Interface
class Analyzer(ABC):
    """Analyzer interface (Strategy Pattern)"""

    @abstractmethod
    async def analyze(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        """Analyze company based on command"""
        pass

    @abstractmethod
    def can_handle(self, analysis_type: str) -> bool:
        """Check if analyzer can handle the analysis type"""
        pass