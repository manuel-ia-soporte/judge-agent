# domain/models/value_objects.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from decimal import Decimal


# Value Objects (Immutable, No Identity)

@dataclass(frozen=True)
class FinancialMetric:
    """Value Object for financial metric"""
    name: str
    value: Decimal
    unit: str
    period: datetime
    source_document_id: str
    footnote: Optional[str] = None
    is_estimated: bool = False
    confidence: float = field(default=1.0, metadata={"min": 0.0, "max": 1.0})

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if self.value < 0:
            raise ValueError("Financial metric value cannot be negative")


@dataclass(frozen=True)
class FinancialRatio:
    """Value Object for financial ratio"""
    name: str
    value: float
    category: str  # "liquidity", "solvency", "profitability", "efficiency", "market"
    calculation_method: str
    benchmark: Optional[float] = None
    industry_average: Optional[float] = None

    def is_favorable(self) -> bool:
        """Determine if the ratio is favorable based on category"""
        favorable_thresholds = {
            "current_ratio": 1.5,
            "quick_ratio": 1.0,
            "debt_to_equity": 2.0,
            "profit_margin": 0.1,
            "roe": 0.15,
            "roa": 0.05
        }
        threshold = favorable_thresholds.get(self.name)
        if threshold:
            return self.value >= threshold
        return True


@dataclass(frozen=True)
class RiskFactor:
    """Value Object for the risk factor"""
    description: str
    category: str  # "market", "financial", "operational", "regulatory", "strategic"
    severity: str  # "high", "medium", "low"
    probability: float = field(default=0.5, metadata={"min": 0.0, "max": 1.0})
    impact: str = "unknown"
    mitigation: Optional[str] = None

    def risk_score(self) -> float:
        """Calculate risk score based on severity and probability"""
        severity_weights = {"high": 1.0, "medium": 0.5, "low": 0.2}
        return severity_weights.get(self.severity, 0.5) * self.probability


@dataclass(frozen=True)
class TrendAnalysis:
    """Value Object for trend analysis"""
    metric_name: str
    values: List[Decimal]
    periods: List[datetime]
    trend: str  # "increasing", "decreasing", "stable"
    slope: float
    r_squared: float
    volatility: float

    def is_significant(self, threshold: float = 0.7) -> bool:
        """Determine if the trend is statistically significant"""
        return abs(self.slope) > 0.1 and self.r_squared > threshold