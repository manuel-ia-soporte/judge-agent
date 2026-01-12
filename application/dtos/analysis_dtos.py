# application/dtos/analysis_dtos.py
"""Data Transfer Objects for analysis results"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum


class AnalysisStatus(str, Enum):
    """Status of analysis"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


@dataclass
class FinancialMetricDTO:
    """DTO for financial metric"""

    name: str
    value: Decimal
    unit: str
    period: datetime
    source_document_id: str
    footnote: Optional[str] = None
    is_estimated: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "period": self.period.isoformat(),
            "source_document_id": self.source_document_id,
            "footnote": self.footnote,
            "is_estimated": self.is_estimated,
            "confidence": self.confidence
        }


@dataclass
class FinancialRatioDTO:
    """DTO for financial ratio"""

    name: str
    value: float
    category: str
    calculation_method: str
    benchmark: Optional[float] = None
    industry_average: Optional[float] = None
    interpretation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "category": self.category,
            "calculation_method": self.calculation_method,
            "benchmark": self.benchmark,
            "industry_average": self.industry_average,
            "interpretation": self.interpretation
        }


@dataclass
class TrendAnalysisDTO:
    """DTO for trend analysis"""

    metric_name: str
    periods: List[datetime]
    values: List[Decimal]
    trend: str
    slope: float
    r_squared: float
    volatility: float
    confidence_interval: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "periods": [p.isoformat() for p in self.periods],
            "values": [float(v) for v in self.values],
            "trend": self.trend,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "volatility": self.volatility,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class AnalysisResultDTO:
    """DTO for the analysis result"""

    analysis_id: str
    company_cik: str
    analysis_type: str
    analysis_date: datetime = field(default_factory=datetime.now)
    status: str = AnalysisStatus.COMPLETED

    # Analysis results
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    ratios: List[Dict[str, Any]] = field(default_factory=list)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)

    # Assessments
    financial_assessment: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    operational_assessment: Optional[Dict[str, Any]] = None
    strategic_assessment: Optional[Dict[str, Any]] = None

    # Business insights
    key_findings: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    documents_analyzed: int = 0
    processing_time_ms: Optional[int] = None
    confidence_score: float = 0.0
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "analysis_id": self.analysis_id,
            "company_cik": self.company_cik,
            "analysis_type": self.analysis_type,
            "analysis_date": self.analysis_date.isoformat(),
            "status": self.status.value,
            "metrics": self.metrics,
            "ratios": self.ratios,
            "trends": self.trends,
            "risk_factors": self.risk_factors,
            "financial_assessment": self.financial_assessment,
            "risk_assessment": self.risk_assessment,
            "operational_assessment": self.operational_assessment,
            "strategic_assessment": self.strategic_assessment,
            "key_findings": self.key_findings,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "documents_analyzed": self.documents_analyzed,
            "processing_time_ms": self.processing_time_ms,
            "confidence_score": self.confidence_score,
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class ComparisonResultDTO:
    """DTO for the company comparison result"""

    company_ciks: List[str]
    comparison_type: str
    metrics: List[str]
    benchmark_company: Optional[str] = None
    comparisons: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    rankings: Dict[str, List[str]] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_ciks": self.company_ciks,
            "comparison_type": self.comparison_type,
            "metrics": self.metrics,
            "benchmark_company": self.benchmark_company,
            "comparisons": self.comparisons,
            "insights": self.insights,
            "rankings": self.rankings,
            "generated_at": self.generated_at.isoformat()
        }


@dataclass
class QuickAnalysisDTO:
    """DTO for the quick analysis result"""

    company_cik: str
    analysis_date: datetime = field(default_factory=datetime.now)
    key_metrics: List[Dict[str, Any]] = field(default_factory=list)
    top_risks: List[Dict[str, Any]] = field(default_factory=list)
    operational_highlights: List[str] = field(default_factory=list)
    strategic_position: Optional[str] = None
    overall_assessment: str = ""
    recommendations: List[str] = field(default_factory=list)
    processing_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "analysis_date": self.analysis_date.isoformat(),
            "key_metrics": self.key_metrics,
            "top_risks": self.top_risks,
            "operational_highlights": self.operational_highlights,
            "strategic_position": self.strategic_position,
            "overall_assessment": self.overall_assessment,
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms
        }