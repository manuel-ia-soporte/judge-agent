"""
Contracts package - Defines interfaces and protocols for the system.
"""
from .evaluation_contracts import RubricCategory, ScoringScale, EvaluationRequest, RubricScore, EvaluationResult, A2AMessage
from .finance_contracts import FilingStatus, FinancialStatementType, SECFilingRequest, FinancialMetricData, CompanyFinancials, MarketDataRequest, RiskAssessment
from .judge_contracts import JudgeCapabilities, JudgeMetrics, JudgeConfiguration

__all__ = [
    "RubricCategory",
    "ScoringScale",
    "EvaluationRequest",
    "RubricScore",
    "EvaluationResult",
    "A2AMessage",
    "FilingStatus",
    "FinancialStatementType",
    "SECFilingRequest",
    "FinancialMetricData",
    "CompanyFinancials",
    "MarketDataRequest",
    "RiskAssessment",
    "JudgeCapabilities",
    "JudgeMetrics",
    "JudgeConfiguration",
]
