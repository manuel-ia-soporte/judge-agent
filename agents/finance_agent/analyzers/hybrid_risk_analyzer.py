# agents/finance_agent/factories/hybrid_risk_analyzer.py
from typing import Dict, Any
from agents.finance_agent.analyzers.risk_analyzer import RiskAnalyzer
from agents.finance_agent.analyzers.llm_risk_analyzer import LLMRiskAnalyzer


class HybridRiskAnalyzer:
    """
    Uses LLM when available, falls back to deterministic logic.
    """

    def __init__(self):
        self._llm = LLMRiskAnalyzer()
        self._fallback = RiskAnalyzer()

    async def analyze(self, financials: Dict[str, float]) -> Dict[str, Any]:
        try:
            return await self._llm.analyze(financials)
        except Exception as e:
            print("Exception while analyzing financials:", e)
            return self._fallback.analyze(financials)
