# contracts/api/responses/analysis_responses.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from decimal import Decimal
from enum import Enum


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class FinancialMetricResponse(BaseModel):
    """Response contract for financial metric"""

    name: str
    value: float
    unit: Optional[str] = None
    period: Optional[datetime] = None
    source_document_id: Optional[str] = None
    footnote: Optional[str] = None
    is_estimated: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FinancialRatioResponse(BaseModel):
    """Response contract for financial ratio"""

    name: str
    value: float
    category: Optional[str] = None
    calculation_method: Optional[str] = None
    benchmark: Optional[float] = None
    industry_average: Optional[float] = None
    is_favorable: Optional[bool] = None


class RiskFactorResponse(BaseModel):
    """Response contract for the risk factor"""

    description: str
    category: Optional[str] = None
    severity: Optional[str] = None
    probability: float = Field(default=0.5, ge=0.0, le=1.0)
    impact: Optional[str] = None
    mitigation: Optional[str] = None
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TrendAnalysisResponse(BaseModel):
    """Response contract for trend analysis"""

    metric_name: str
    trend: TrendDirection
    slope: float
    r_squared: float
    volatility: float
    is_significant: bool
    period_start: datetime
    period_end: datetime


class AnalysisResultResponse(BaseModel):
    """Response contract for analysis result"""

    analysis_id: str
    company_cik: str
    analysis_type: str
    analysis_date: datetime = Field(default_factory=datetime.now)

    # Analysis results
    metrics: List[FinancialMetricResponse] = Field(default_factory=list)
    ratios: List[FinancialRatioResponse] = Field(default_factory=list)
    risk_factors: List[RiskFactorResponse] = Field(default_factory=list)
    trends: List[TrendAnalysisResponse] = Field(default_factory=list)

    # Assessments
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    liquidity_assessment: Optional[str] = None
    profitability_assessment: Optional[str] = None

    # Business insights
    conclusions: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # Metadata
    documents_analyzed: int = 0
    processing_time_ms: Optional[int] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "analysis_123",
                "company_cik": "0000320193",
                "analysis_type": "comprehensive",
                "risk_level": "medium",
                "risk_score": 0.45,
                "conclusions": ["Strong liquidity position", "Moderate risk profile"]
            }
        }