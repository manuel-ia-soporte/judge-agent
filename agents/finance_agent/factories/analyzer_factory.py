# agents/finance_agent/factories/analyzer_factory.py
from typing import Dict
from ..analyzers.analyzer_interface import Analyzer
from ..analyzers.financial_analyzer import FinancialAnalyzer
from ..analyzers.risk_analyzer import RiskAnalyzer
from ..analyzers.operational_analyzer import OperationalAnalyzer
from ..analyzers.strategic_analyzer import StrategicAnalyzer
from .quick_analyzer import QuickAnalyzer


# Factory Pattern
class AnalyzerFactory:
    """Factory for creating analyzers"""

    def __init__(self):
        self._analyzers: Dict[str, Analyzer] = {}

    def register_analyzer(self, analyzer_type: str, analyzer: Analyzer):
        """Register analyzer type"""
        self._analyzers[analyzer_type] = analyzer

    def create_analyzer(self, analysis_type: str) -> Analyzer:
        """Create analyzer based on type"""
        analyzers = {
            "financial": FinancialAnalyzer,
            "risk": RiskAnalyzer,
            "operational": OperationalAnalyzer,
            "strategic": StrategicAnalyzer,
            "quick": QuickAnalyzer,
            "comprehensive": ComprehensiveAnalyzer  # Composite pattern
        }

        analyzer_class = analyzers.get(analysis_type)
        if not analyzer_class:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # This would use dependency injection in real implementation
        return analyzer_class()