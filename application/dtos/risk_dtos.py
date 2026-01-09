# application/dtos/risk_dtos.py
"""Data Transfer Objects for risk assessment"""

from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel,Field
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskSeverity(str, Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSIGNIFICANT = "insignificant"


class RiskCategory(str, Enum):
    """Risk categories"""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    REGULATORY = "regulatory"
    REPUTATIONAL = "reputational"
    MARKET = "market"
    CYBERSECURITY = "cybersecurity"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"


@dataclass
class RiskFactorDTO:
    """DTO for the risk factor"""

    description: str
    category: RiskCategory
    severity: RiskSeverity
    probability: float
    impact: str
    mitigation: Optional[str] = None
    monitoring_indicators: List[str] = field(default_factory=list)
    last_assessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "probability": self.probability,
            "impact": self.impact,
            "mitigation": self.mitigation,
            "monitoring_indicators": self.monitoring_indicators,
            "last_assessed": self.last_assessed.isoformat() if self.last_assessed else None
        }


@dataclass
class RiskAssessmentDTO:
    """DTO for comprehensive risk assessment"""

    company_cik: str
    assessment_date: datetime = field(default_factory=datetime.now)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    risk_categories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    risk_level: str = ""
    risk_score: float = 0.0
    composite_risk_score: float = 0.0

    # Detailed assessments
    financial_risk_assessment: Optional[Dict[str, Any]] = None
    operational_risk_assessment: Optional[Dict[str, Any]] = None
    strategic_risk_assessment: Optional[Dict[str, Any]] = None
    regulatory_risk_assessment: Optional[Dict[str, Any]] = None

    # Mitigation and monitoring
    mitigations: List[str] = field(default_factory=list)
    key_risk_indicators: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    monitoring_indicators: List[str] = field(default_factory=list)

    # Metadata
    next_review_date: Optional[datetime] = None
    documents_analyzed: int = 0
    assessment_methodology: str = "SEC filing analysis"
    confidence_level: float = 0.85
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "assessment_date": self.assessment_date.isoformat(),
            "risk_factors": self.risk_factors,
            "risk_categories": self.risk_categories,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "composite_risk_score": self.composite_risk_score,
            "financial_risk_assessment": self.financial_risk_assessment,
            "operational_risk_assessment": self.operational_risk_assessment,
            "strategic_risk_assessment": self.strategic_risk_assessment,
            "regulatory_risk_assessment": self.regulatory_risk_assessment,
            "mitigations": self.mitigations,
            "key_risk_indicators": self.key_risk_indicators,
            "recommendations": self.recommendations,
            "monitoring_indicators": self.monitoring_indicators,
            "next_review_date": self.next_review_date.isoformat() if self.next_review_date else None,
            "documents_analyzed": self.documents_analyzed,
            "assessment_methodology": self.assessment_methodology,
            "confidence_level": self.confidence_level,
            "version": self.version
        }


@dataclass
class RiskTrendDTO:
    """DTO for risk trend analysis"""

    company_cik: str
    risk_category: Optional[str]
    periods: List[datetime]
    risk_scores: List[float]
    trend: str
    trend_strength: float
    volatility: float
    peak_risk_period: Optional[datetime] = None
    peak_risk_score: Optional[float] = None
    comparison_companies: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "risk_category": self.risk_category,
            "periods": [p.isoformat() for p in self.periods],
            "risk_scores": self.risk_scores,
            "trend": self.trend,
            "trend_strength": self.trend_strength,
            "volatility": self.volatility,
            "peak_risk_period": self.peak_risk_period.isoformat() if self.peak_risk_period else None,
            "peak_risk_score": self.peak_risk_score,
            "comparison_companies": self.comparison_companies
        }


@dataclass
class RiskExposureDTO:
    """DTO for risk exposure analysis"""

    company_cik: str
    scenarios: List[Dict[str, Any]]
    expected_loss: float
    worst_case_loss: float
    value_at_risk: float
    confidence_level: float
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    recovery_analysis: Optional[Dict[str, Any]] = None
    capital_adequacy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "scenarios": self.scenarios,
            "expected_loss": self.expected_loss,
            "worst_case_loss": self.worst_case_loss,
            "value_at_risk": self.value_at_risk,
            "confidence_level": self.confidence_level,
            "sensitivity_analysis": self.sensitivity_analysis,
            "recovery_analysis": self.recovery_analysis,
            "capital_adequacy": self.capital_adequacy
        }


@dataclass
class RiskProfileComparisonDTO:
    """DTO for risk profile comparison"""

    company_ciks: List[str]
    comparison_date: datetime = field(default_factory=datetime.now)
    risk_scores: Dict[str, float] = field(default_factory=dict)  # cik -> score
    risk_levels: Dict[str, str] = field(default_factory=dict)  # cik -> level
    category_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)  # category -> cik -> score
    heatmap_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rankings: Dict[str, List[str]] = field(default_factory=dict)  # category -> ranked ciks
    key_differences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_ciks": self.company_ciks,
            "comparison_date": self.comparison_date.isoformat(),
            "risk_scores": self.risk_scores,
            "risk_levels": self.risk_levels,
            "category_scores": self.category_scores,
            "heatmap_data": self.heatmap_data,
            "rankings": self.rankings,
            "key_differences": self.key_differences
        }


class RiskLevelDTO(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class FinancialMetricsDTO(BaseModel):
    """DTO for financial metrics input"""
    debt: float = 0.0
    equity: float = 1.0
    revenue: float = 1.0
    net_income: float = 0.0
    ebitda: float = 0.0
    cash: float = 0.0

class RiskAnalysisResultDTO(BaseModel):
    """DTO for risk analysis output"""
    risk_score: float = Field(ge=0.0, le=1.0, description="Risk score between 0 and 1")
    risk_level: RiskLevelDTO
    explanation: str
    key_metrics: Dict[str, float]
    risk_factors: Optional[List[str]] = None