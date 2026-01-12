# agents/finance_agent/analyzers/llm_risk_analyzer.py
### SHOULD IMPLEMENT
# LLMRiskAnalyzer
# currently just placeholder

from typing import Dict
from dataclasses import dataclass
import time

from application.dtos import FinancialMetricsDTO, RiskAnalysisResultDTO, RiskLevelDTO
from agents.shared.value_objects.agent_metrics import AgentMetrics


@dataclass
class RiskThresholds:
    """Centralized risk thresholds configuration"""
    DEBT_TO_EQUITY_HIGH: float = 2.0
    PROFIT_MARGIN_LOW: float = 0.05
    INTEREST_COVERAGE_WEAK: float = 3.0
    CASH_RATIO_INSUFFICIENT: float = 0.2
    INTEREST_RATE: float = 0.05


class LLMRiskAnalyzer:
    """
    Simulated LLM-backed risk analyzer.
    Uses financial signals to estimate risk score, rationale.
    Implements fallback logic if LLM unavailable.

    Now tracks AgentMetrics per operation (Issue #10).
    """

    def __init__(self, thresholds: RiskThresholds = None):
        self.thresholds = thresholds or RiskThresholds()
        self.metrics = AgentMetrics(processing_time_ms=0, tokens_used=0, warnings_count=0)

    async def analyze(self, financials: FinancialMetricsDTO) -> RiskAnalysisResultDTO:
        start = time.time()
        self._start_operation()

        try:
            # Calculate derived metrics
            metrics = self._calculate_metrics(financials)

            # Assess risks
            risk_score, risk_factors = self._assess_risks(metrics)

            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)

            result = RiskAnalysisResultDTO(
                risk_score=round(risk_score, 2),
                risk_level=risk_level,
                explanation=self._format_explanation(risk_factors),
                key_metrics=self._format_metrics(metrics),
                risk_factors=risk_factors
            )

            duration_ms = int((time.time() - start) * 1000)
            self._record_success(duration_ms, tokens=120)  # estimated token usage

            return result

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            self._record_failure(duration_ms, str(e))
            raise

    def _start_operation(self):
        """Mark the start of a new risk analysis operation."""
        # In this simple version, we don't store start time internally,
        # because duration is computed externally in `analyze()`.
        pass

    def _record_success(self, processing_time_ms: int, tokens: int = 0):
        """Record successful operation metrics."""
        self.metrics = AgentMetrics(
            processing_time_ms=self.metrics.processing_time_ms + processing_time_ms,
            tokens_used=self.metrics.tokens_used + tokens,
            warnings_count=self.metrics.warnings_count
        )

    def _record_failure(self, processing_time_ms: int, error: str):
        """Record failed operation metrics."""
        self.metrics = AgentMetrics(
            processing_time_ms=self.metrics.processing_time_ms + processing_time_ms,
            tokens_used=self.metrics.tokens_used,
            warnings_count=self.metrics.warnings_count + 1
        )

    def get_metrics(self) -> AgentMetrics:
        """Return current agent metrics (for observability)."""
        return self.metrics

    def _calculate_metrics(self, financials: FinancialMetricsDTO) -> Dict[str, float]:
        """Calculate all derived financial metrics"""
        equity = max(financials.equity, 0.01)
        revenue = max(financials.revenue, 0.01)  # Avoid division by zero

        return {
            "debt_to_equity": financials.debt / equity,
            "profit_margin": financials.net_income / revenue,
            "interest_coverage": (
                    financials.ebitda /
                    max(financials.debt * self.thresholds.INTEREST_RATE, 0.01)
            ),
            "cash_ratio": financials.cash / max(financials.debt, 0.01)
        }

    def _assess_risks(self, metrics: Dict[str, float]) -> tuple[float, list]:
        """Calculate risk score and identify risk factors"""
        risk_score = 0.0
        risk_factors = []

        risk_rules = [
            {
                "condition": metrics["debt_to_equity"] > self.thresholds.DEBT_TO_EQUITY_HIGH,
                "weight": 0.3,
                "message": f"High leverage (D/E={metrics['debt_to_equity']:.2f})"
            },
            {
                "condition": metrics["profit_margin"] < self.thresholds.PROFIT_MARGIN_LOW,
                "weight": 0.25,
                "message": f"Low profitability (margin={metrics['profit_margin']:.1%})"
            },
            {
                "condition": metrics["interest_coverage"] < self.thresholds.INTEREST_COVERAGE_WEAK,
                "weight": 0.2,
                "message": f"Weak interest coverage ({metrics['interest_coverage']:.1f}x)"
            },
            {
                "condition": metrics["cash_ratio"] < self.thresholds.CASH_RATIO_INSUFFICIENT,
                "weight": 0.15,
                "message": "Insufficient liquidity buffer"
            }
        ]

        for rule in risk_rules:
            if rule["condition"]:
                risk_score += rule["weight"]
                risk_factors.append(rule["message"])

        return max(0.0, min(1.0, risk_score)), risk_factors

    @staticmethod
    def _determine_risk_level(risk_score: float) -> RiskLevelDTO:
        if risk_score >= 0.7:
            return RiskLevelDTO.HIGH
        elif risk_score >= 0.4:
            return RiskLevelDTO.MEDIUM
        return RiskLevelDTO.LOW

    @staticmethod
    def _format_explanation(risk_factors: list) -> str:
        return "; ".join(risk_factors) if risk_factors else "No significant risks identified"

    @staticmethod
    def _format_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        rounding = {
            "debt_to_equity": 2,
            "profit_margin": 4,
            "interest_coverage": 2,
            "cash_ratio": 2
        }
        return {key: round(value, rounding[key]) for key, value in metrics.items()}