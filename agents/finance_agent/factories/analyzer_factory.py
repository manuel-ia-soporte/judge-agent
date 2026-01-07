# agents/finance_agent/factories/analyzer_factory.py

from agents.finance_agent.analyzers.risk_analyzer import RiskAnalyzer


class AnalyzerFactory:
    """
    Factory for analyzer creation.
    Only exposes analyzers actually used by agents.
    """

    @staticmethod
    def create_risk() -> RiskAnalyzer:
        return RiskAnalyzer()
