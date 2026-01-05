# agents/finance_agent/factories/analyzer_factory.py
from agents.finance_agent.analyzers.operational_analyzer import (
    OperationalAnalyzer,
)
from agents.finance_agent.analyzers.risk_analyzer import RiskAnalyzer


class AnalyzerFactory:
    @staticmethod
    def create_operational() -> OperationalAnalyzer:
        return OperationalAnalyzer()

    @staticmethod
    def create_risk() -> RiskAnalyzer:
        return RiskAnalyzer()
