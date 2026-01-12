# agents/finance_agent/analyzers/hybrid_risk_analyzer.py
from typing import Dict, Any
from application.dtos.risk_dtos import FinancialMetricsDTO, RiskAnalysisResultDTO
from agents.finance_agent.analyzers.llm_risk_analyzer import LLMRiskAnalyzer
from agents.finance_agent.analyzers.risk_analyzer import RiskAnalyzer

class HybridRiskAnalyzer:
    """
    Uses LLM when available, falls back to deterministic logic.
    Accepts dict-like financial signals and converts to DTO.
    """

    def __init__(self):
        self._llm = LLMRiskAnalyzer()
        self._fallback = RiskAnalyzer()

    async def analyze(self, financials: Dict[str, float]) -> Dict[str, Any]:
        # Convert dict to FinancialMetricsDTO
        dto = FinancialMetricsDTO(
            debt=financials.get("debt", 0.0),
            equity=financials.get("equity", 1.0),
            revenue=financials.get("revenue", 1.0),
            net_income=financials.get("net_income", 0.0),
            ebitda=financials.get("ebitda", 0.0),
            cash=financials.get("cash", 0.0)
        )
        try:
            result: RiskAnalysisResultDTO = await self._llm.analyze(dto)
            return result.model_dump()  # Pydantic .dict()
        except Exception as e:
            print("Exception while analyzing financials:", e)
            fallback_result = self._fallback.analyze(financials)
            return fallback_result