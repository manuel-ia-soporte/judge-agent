# application/dtos/financial_dtos.py
"""Data Transfer Objects for financial data"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum


class FinancialStatementType(str, Enum):
    """Types of financial statements"""
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW_STATEMENT = "cash_flow_statement"
    CHANGES_IN_EQUITY = "changes_in_equity"
    COMPREHENSIVE_INCOME = "comprehensive_income"


class MetricConfidence(str, Enum):
    """Confidence levels for financial metrics"""
    HIGH = "high"  # Directly from audited statements
    MEDIUM = "medium"  # Calculated from audited data
    LOW = "low"  # Estimated or derived
    ESTIMATED = "estimated" # Management estimates

@dataclass
class FinancialStatementDTO:
    """DTO for financial statement"""

    statement_type: FinancialStatementType
    period: datetime
    line_items: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # item_name -> {value, unit, footnote}
    footnotes: List[str] = field(default_factory=list)
    is_consolidated: bool = True
    currency: str = "USD"
    source_document_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "statement_type": self.statement_type.value,
            "period": self.period.isoformat(),
            "line_items": self.line_items,
            "footnotes": self.footnotes,
            "is_consolidated": self.is_consolidated,
            "currency": self.currency,
            "source_document_id": self.source_document_id
        }


@dataclass
class FinancialMetricSeriesDTO:
    """DTO for time series of financial metrics"""

    metric_name: str
    periods: List[datetime]
    values: List[Decimal]
    units: List[str]
    confidence_scores: List[float]
    trends: List[str] = field(default_factory=list)
    growth_rates: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "periods": [p.isoformat() for p in self.periods],
            "values": [float(v) for v in self.values],
            "units": self.units,
            "confidence_scores": self.confidence_scores,
            "trends": self.trends,
            "growth_rates": self.growth_rates
        }


@dataclass
class RatioAnalysisDTO:
    """DTO for ratio analysis"""

    company_cik: str
    analysis_date: datetime = field(default_factory=datetime.now)
    liquidity_ratios: Dict[str, float] = field(default_factory=dict)
    solvency_ratios: Dict[str, float] = field(default_factory=dict)
    profitability_ratios: Dict[str, float] = field(default_factory=dict)
    efficiency_ratios: Dict[str, float] = field(default_factory=dict)
    market_ratios: Dict[str, float] = field(default_factory=dict)

    # Benchmarks and comparisons
    industry_averages: Dict[str, float] = field(default_factory=dict)
    historical_comparison: Dict[str, List[float]] = field(default_factory=dict)
    peer_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "analysis_date": self.analysis_date.isoformat(),
            "liquidity_ratios": self.liquidity_ratios,
            "solvency_ratios": self.solvency_ratios,
            "profitability_ratios": self.profitability_ratios,
            "efficiency_ratios": self.efficiency_ratios,
            "market_ratios": self.market_ratios,
            "industry_averages": self.industry_averages,
            "historical_comparison": self.historical_comparison,
            "peer_comparison": self.peer_comparison,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations
        }


@dataclass
class TrendFinancialAnalysisResultDTO:
    """DTO for trend analysis results"""

    metric_name: str
    analysis_period: Dict[str, datetime]  # start and end
    trend_direction: str
    trend_strength: float
    slope: float
    r_squared: float
    volatility: float
    seasonal_pattern: Optional[str] = None
    forecast: Optional[Dict[str, Any]] = None
    breakpoints: List[datetime] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "analysis_period": {
                "start": self.analysis_period["start"].isoformat(),
                "end": self.analysis_period["end"].isoformat()
            },
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "volatility": self.volatility,
            "seasonal_pattern": self.seasonal_pattern,
            "forecast": self.forecast,
            "breakpoints": [bp.isoformat() for bp in self.breakpoints]
        }


@dataclass
class FinancialForecastDTO:
    """DTO for financial forecasts"""

    company_cik: str
    forecast_date: datetime = field(default_factory=datetime.now)
    forecast_periods: List[datetime] = field(default_factory=list)
    metric_forecasts: Dict[str, List[float]] = field(default_factory=dict)  # metric -> forecast values
    confidence_intervals: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    scenario_analysis: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "forecast_date": self.forecast_date.isoformat(),
            "forecast_periods": [p.isoformat() for p in self.forecast_periods],
            "metric_forecasts": self.metric_forecasts,
            "confidence_intervals": self.confidence_intervals,
            "assumptions": self.assumptions,
            "scenario_analysis": self.scenario_analysis,
            "sensitivity_analysis": self.sensitivity_analysis
        }


@dataclass
class ValuationDTO:
    """DTO for company valuation"""

    company_cik: str
    valuation_date: datetime = field(default_factory=datetime.now)
    valuation_methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # method -> details
    fair_value_range: Dict[str, float] = field(default_factory=dict)  # low, base, high
    key_assumptions: Dict[str, Any] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, Any] = field(default_factory=dict)
    comparable_companies: List[Dict[str, Any]] = field(default_factory=list)
    discount_rate: float = 0.0
    terminal_growth_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "company_cik": self.company_cik,
            "valuation_date": self.valuation_date.isoformat(),
            "valuation_methods": self.valuation_methods,
            "fair_value_range": self.fair_value_range,
            "key_assumptions": self.key_assumptions,
            "sensitivity_analysis": self.sensitivity_analysis,
            "comparable_companies": self.comparable_companies,
            "discount_rate": self.discount_rate,
            "terminal_growth_rate": self.terminal_growth_rate
        }