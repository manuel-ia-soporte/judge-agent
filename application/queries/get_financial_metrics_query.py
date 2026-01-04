# application/queries/get_financial_metrics_query.py
"""Queries for financial metrics"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List, Optional, Dict, Any
from enum import Enum


class MetricGranularity(str, Enum):
    """Granularity levels for metric retrieval"""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    ALL = "all"


class MetricFormat(str, Enum):
    """Format options for metric data"""
    RAW = "raw"
    NORMALIZED = "normalized"
    PERCENTAGE = "percentage"
    INDEXED = "indexed"  # Indexed to a base period


@dataclass
class GetFinancialMetricsQuery:
    """Query to get financial metrics for a company"""

    company_cik: str
    metric_names: List[str]
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    granularity: MetricGranularity = MetricGranularity.ANNUAL
    format: MetricFormat = MetricFormat.RAW
    include_historical: bool = False
    include_metadata: bool = True
    include_calculations: bool = False
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate metric names
        if not self.metric_names:
            errors.append("At least one metric name is required")

        # Validate dates
        if self.period_start and self.period_end:
            if self.period_start > self.period_end:
                errors.append("Start date must be before end date")

        # Validate granularity for historical data
        if self.include_historical and self.granularity == MetricGranularity.MONTHLY:
            errors.append("Monthly granularity not available for historical data")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert the query to dictionary"""
        return {
            "company_cik": self.company_cik,
            "metric_names": self.metric_names,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "granularity": self.granularity.value,
            "format": self.format.value,
            "include_historical": self.include_historical,
            "include_metadata": self.include_metadata,
            "include_calculations": self.include_calculations,
            "query_id": self.query_id
        }


@dataclass
class GetMetricTrendsQuery:
    """Query to get metric trends over time"""

    company_cik: str
    metric_name: str
    lookback_periods: int = 8  # Number of periods to look back
    trend_type: str = "linear"  # linear, exponential, moving_average
    confidence_level: float = 0.95
    include_forecast: bool = False
    forecast_periods: int = 4
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate metric name
        if not self.metric_name:
            errors.append("Metric name is required")

        # Validate lookback periods
        if self.lookback_periods < 2 or self.lookback_periods > 20:
            errors.append("Lookback periods must be between 2 and 20")

        # Validate trend type
        valid_trends = ["linear", "exponential", "moving_average", "seasonal"]
        if self.trend_type not in valid_trends:
            errors.append(f"Trend type must be one of: {', '.join(valid_trends)}")

        # Validate confidence level
        if self.confidence_level < 0.5 or self.confidence_level > 0.99:
            errors.append("Confidence level must be between 0.5 and 0.99")

        # Validate forecast periods
        if self.include_forecast and (self.forecast_periods < 1 or self.forecast_periods > 12):
            errors.append("Forecast periods must be between 1 and 12")

        return errors


@dataclass
class CompareMetricsQuery:
    """Query to compare metrics across companies"""

    company_ciks: List[str]
    metric_name: str
    comparison_period: datetime  # Specific period to compare
    normalize: bool = True
    include_benchmark: bool = False
    benchmark_value: Optional[float] = None
    include_rankings: bool = True
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate company CIKs
        if len(self.company_ciks) < 2:
            errors.append("At least two companies required for comparison")

        for cik in self.company_ciks:
            if not cik or not cik.isdigit():
                errors.append(f"Invalid CIK: {cik}")

        # Validate metric name
        if not self.metric_name:
            errors.append("Metric name is required")

        # Validate comparison period
        if self.comparison_period > datetime.now(UTC):
            errors.append("Comparison period cannot be in the future")

        return errors


@dataclass
class CalculateDerivedMetricsQuery:
    """Query to calculate derived metrics"""

    company_cik: str
    base_metrics: Dict[str, float]  # metric_name -> value
    calculation_formula: str  # e.g., "current_ratio = current_assets / current_liabilities"
    assumptions: Optional[Dict[str, Any]] = None
    include_sensitivity: bool = False
    sensitivity_range: float = 0.1  # +/- 10%
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate base metrics
        if not self.base_metrics:
            errors.append("Base metrics are required")

        # Validate calculation formula
        if not self.calculation_formula:
            errors.append("Calculation formula is required")

        # Check that formula uses provided metrics
        import re
        metric_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
        formula_metrics = set(re.findall(metric_pattern, self.calculation_formula))
        provided_metrics = set(self.base_metrics.keys())

        # Remove operators and numbers
        operators = {'+', '-', '*', '/', '=', '(', ')', '^'}
        formula_metrics = formula_metrics - operators

        # Check for numeric strings
        formula_metrics = {m for m in formula_metrics if not m.replace('.', '', 1).isdigit()}

        missing_metrics = formula_metrics - provided_metrics
        if missing_metrics:
            errors.append(f"Missing metrics in formula: {', '.join(missing_metrics)}")

        # Validate sensitivity range
        if self.sensitivity_range <= 0 or self.sensitivity_range > 0.5:
            errors.append("Sensitivity range must be between 0 and 0.5")

        return errors