# application/queries/get_risk_assessment_query.py
"""Queries for risk assessment"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskAssessmentScope(str, Enum):
    """Scope of risk assessment"""
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


class RiskTimeframe(str, Enum):
    """Timeframe for risk assessment"""
    CURRENT = "current"
    NEAR_TERM = "near_term"  # 0-12 months
    MEDIUM_TERM = "medium_term"  # 1-3 years
    LONG_TERM = "long_term"  # 3+ years


@dataclass
class GetRiskAssessmentQuery:
    """Query to get risk assessment for a company"""

    company_cik: str
    assessment_scope: RiskAssessmentScope = RiskAssessmentScope.COMPREHENSIVE
    timeframe: RiskTimeframe = RiskTimeframe.CURRENT
    include_mitigations: bool = True
    include_monitoring: bool = True
    risk_categories: Optional[List[str]] = None
    severity_threshold: Optional[str] = None  # Only risks above this severity
    probability_threshold: float = 0.3  # Minimum probability
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate risk categories for custom scope
        if self.assessment_scope == RiskAssessmentScope.CUSTOM and not self.risk_categories:
            errors.append("Risk categories required for custom scope")

        # Validate severity threshold
        valid_severities = ["critical", "high", "medium", "low", "insignificant"]
        if self.severity_threshold and self.severity_threshold not in valid_severities:
            errors.append(f"Severity threshold must be one of: {', '.join(valid_severities)}")

        # Validate probability threshold
        if self.probability_threshold < 0 or self.probability_threshold > 1:
            errors.append("Probability threshold must be between 0 and 1")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert the query to dictionary"""
        return {
            "company_cik": self.company_cik,
            "assessment_scope": self.assessment_scope.value,
            "timeframe": self.timeframe.value,
            "include_mitigations": self.include_mitigations,
            "include_monitoring": self.include_monitoring,
            "risk_categories": self.risk_categories,
            "severity_threshold": self.severity_threshold,
            "probability_threshold": self.probability_threshold,
            "query_id": self.query_id
        }


@dataclass
class GetRiskTrendsQuery:
    """Query to get risk trends over time"""

    company_cik: str
    risk_category: Optional[str] = None  # Specific category or None for all
    lookback_periods: int = 4  # Number of periods to look back
    trend_metric: str = "count"  # count, severity_score, probability_score
    include_comparison: bool = False
    comparison_companies: Optional[List[str]] = None
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate lookback periods
        if self.lookback_periods < 1 or self.lookback_periods > 10:
            errors.append("Lookback periods must be between 1 and 10")

        # Validate trend metric
        valid_metrics = ["count", "severity_score", "probability_score", "composite_score"]
        if self.trend_metric not in valid_metrics:
            errors.append(f"Trend metric must be one of: {', '.join(valid_metrics)}")

        # Validate comparison companies
        if self.include_comparison:
            if not self.comparison_companies:
                errors.append("Comparison companies required when include_comparison is True")
            else:
                for cik in self.comparison_companies:
                    if not cik or not cik.isdigit():
                        errors.append(f"Invalid comparison company CIK: {cik}")

        return errors


@dataclass
class GetRiskExposureQuery:
    """Query to get risk exposure analysis"""

    company_cik: str
    risk_scenarios: List[Dict[str, Any]]  # List of scenario definitions
    include_sensitivity: bool = True
    confidence_level: float = 0.90
    include_recovery: bool = False
    recovery_periods: int = 4
    query_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate risk scenarios
        if not self.risk_scenarios:
            errors.append("At least one risk scenario is required")

        for scenario in self.risk_scenarios:
            if "name" not in scenario:
                errors.append("Risk scenario must have a name")
            if "probability" not in scenario:
                errors.append("Risk scenario must have a probability")
            elif not (0 <= scenario["probability"] <= 1):
                errors.append("Risk scenario probability must be between 0 and 1")
            if "impact" not in scenario:
                errors.append("Risk scenario must have an impact value")

        # Validate confidence level
        if self.confidence_level < 0.5 or self.confidence_level > 0.99:
            errors.append("Confidence level must be between 0.5 and 0.99")

        # Validate recovery periods
        if self.include_recovery and (self.recovery_periods < 1 or self.recovery_periods > 12):
            errors.append("Recovery periods must be between 1 and 12")

        return errors


@dataclass
class CompareRiskProfilesQuery:
    """Query to compare risk profiles across companies"""

    company_ciks: List[str]
    comparison_categories: List[str] = None  # Specific categories or all
    normalize_scores: bool = True
    include_heatmap: bool = True
    risk_weighting: Optional[Dict[str, float]] = None  # Custom weighting
    query_id: Optional[str] = None

    def __post_init__(self):
        if self.comparison_categories is None:
            self.comparison_categories = ["financial", "operational", "strategic", "regulatory"]

    def validate(self) -> List[str]:
        """Validate query parameters"""
        errors = []

        # Validate company CIKs
        if len(self.company_ciks) < 2:
            errors.append("At least two companies required for comparison")

        for cik in self.company_ciks:
            if not cik or not cik.isdigit():
                errors.append(f"Invalid CIK: {cik}")

        # Validate comparison categories
        valid_categories = ["financial", "operational", "strategic", "regulatory",
                            "reputational", "market", "cybersecurity"]
        for category in self.comparison_categories:
            if category not in valid_categories:
                errors.append(f"Invalid risk category: {category}")

        # Validate risk weighting
        if self.risk_weighting:
            total_weight = sum(self.risk_weighting.values())
            if abs(total_weight - 1.0) > 0.001:  # Allow small floating point error
                errors.append("Risk weights must sum to 1.0")
            for category, weight in self.risk_weighting.items():
                if category not in valid_categories:
                    errors.append(f"Invalid category in risk weighting: {category}")
                if weight < 0 or weight > 1:
                    errors.append(f"Risk weight for {category} must be between 0 and 1")

        return errors