# agents/finance_agent/strategies/analysis_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from agents.finance_agent.analyzers.financial_analyzer import FinancialAnalyzer
from domain.models.entities import SECDocument
from ..analyzers.risk_analyzer import RiskAnalyzer
from ..analyzers.operational_analyzer import OperationalAnalyzer
from ..analyzers.strategic_analyzer import StrategicAnalyzer


# Strategy Pattern
class AnalysisStrategy(ABC):
    """Strategy interface for analysis"""

    @abstractmethod
    async def execute(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Execute analysis strategy"""
        pass

    @abstractmethod
    def get_required_documents(self) -> List[str]:
        """Get required document types"""
        pass


class ComprehensiveStrategy(AnalysisStrategy):
    """Comprehensive analysis strategy"""

    async def execute(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Execute comprehensive analysis"""
        # Use multiple analyzers (Composite Pattern)
        financial_analyzer = FinancialAnalyzer()
        risk_analyzer = RiskAnalyzer()
        operational_analyzer = OperationalAnalyzer()
        strategic_analyzer = StrategicAnalyzer()

        results = {
            "financial": await financial_analyzer.analyze(documents),
            "risk": await risk_analyzer.analyze(documents),
            "operational": await operational_analyzer.analyze(documents),
            "strategic": await strategic_analyzer.analyze(documents)
        }

        # Generate integrated conclusions
        results["conclusions"] = self._integrate_conclusions(results)

        return results

    def get_required_documents(self) -> List[str]:
        return ["10-K", "10-Q"]

    @staticmethod
    def _integrate_conclusions(results: Dict[str, Any]) -> List[str]:
        """Integrate conclusions from all analyses"""
        conclusions = []

        financial = results.get("financial", {})
        risk = results.get("risk", {})

        # Integrated financial-risk conclusions
        if (financial.get("current_ratio", 0) < 1 and
                risk.get("risk_level") == "high"):
            conclusions.append("High risk compounded by liquidity concerns")

        return conclusions


class FinancialStrategy(AnalysisStrategy):
    """Financial analysis strategy"""

    async def execute(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Execute financial analysis"""
        # Use financial analyzer
        pass

    def get_required_documents(self) -> List[str]:
        return ["10-K", "10-Q"]


class RiskStrategy(AnalysisStrategy):
    """Risk analysis strategy"""

    async def execute(self, documents: List[SECDocument]) -> Dict[str, Any]:
        """Execute risk analysis"""
        # Use risk analyzer
        pass

    def get_required_documents(self) -> List[str]:
        return ["10-K"]