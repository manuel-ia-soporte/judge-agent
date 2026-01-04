# domain/models/enums.py
"""Domain enums for the financial analysis system"""

from enum import Enum
from typing import List


class AnalysisType(Enum):
    """Types of analysis that can be performed"""
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    RISK = "risk"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    QUICK = "quick"
    COMPARATIVE = "comparative"
    TREND = "trend"

    @classmethod
    def values(cls) -> List[str]:
        """Get all enum values"""
        return [item.value for item in cls]


class AgentRole(Enum):
    """Roles that agents can play in the system"""
    FINANCE_ANALYST = "finance_analyst"
    RISK_ANALYST = "risk_analyst"
    JUDGE = "judge"
    DATA_EXTRACTOR = "data_extractor"
    COMPLIANCE_CHECKER = "compliance_checker"
    FORECASTER = "forecaster"
    VALIDATOR = "validator"


class FinancialStatementType(Enum):
    """Types of financial statements"""
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    CHANGES_IN_EQUITY = "changes_in_equity"
    COMPREHENSIVE_INCOME = "comprehensive_income"


class RiskCategory(Enum):
    """Categories of risk factors"""
    MARKET = "market"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    STRATEGIC = "strategic"
    REPUTATIONAL = "reputational"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    CYBERSECURITY = "cybersecurity"


class SeverityLevel(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSIGNIFICANT = "insignificant"

    @classmethod
    def from_score(cls, score: float) -> "SeverityLevel":
        """Convert numeric score to severity level"""
        if score >= 0.8:
            return cls.CRITICAL
        elif score >= 0.6:
            return cls.HIGH
        elif score >= 0.4:
            return cls.MEDIUM
        elif score >= 0.2:
            return cls.LOW
        else:
            return cls.INSIGNIFICANT


class TrendDirection(Enum):
    """Trend directions for financial metrics"""
    STRONGLY_INCREASING = "strongly_increasing"
    MODERATELY_INCREASING = "moderately_increasing"
    STABLE = "stable"
    MODERATELY_DECREASING = "moderately_decreasing"
    STRONGLY_DECREASING = "strongly_decreasing"


class MetricConfidence(Enum):
    """Confidence levels for financial metrics"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"
    CALCULATED = "calculated"


class FilingStatus(Enum):
    """Status of SEC filings"""
    FILED = "filed"
    AMENDED = "amended"
    WITHDRAWN = "withdrawn"
    DELAYED = "delayed"
    PENDING = "pending"


class AnalysisStatus(Enum):
    """Status of analysis processes"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    PUBLISHED = "published"