# application/commands/compare_companies_command.py
"""Commands for company comparison"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum


class ComparisonType(str, Enum):
    """Types of comparison analysis"""
    FINANCIAL = "financial"
    RISK = "risk"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"


@dataclass
class CompareCompaniesCommand:
    """Command to compare multiple companies"""

    company_ciks: List[str]
    comparison_type: ComparisonType = ComparisonType.COMPREHENSIVE
    metrics: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filing_types: Optional[List[str]] = None
    benchmark_company: Optional[str] = None
    include_visualizations: bool = True
    depth_level: str = "standard"
    request_id: Optional[str] = None
    agent_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate command parameters"""
        errors = []

        # Validate company CIKs
        if len(self.company_ciks) < 2:
            errors.append("At least two companies required for comparison")
        if len(self.company_ciks) > 10:
            errors.append("Maximum 10 companies allowed for comparison")

        for cik in self.company_ciks:
            if not cik or not cik.isdigit():
                errors.append(f"Invalid CIK: {cik}")

        # Check for duplicates
        if len(self.company_ciks) != len(set(self.company_ciks)):
            errors.append("Duplicate CIKs are not allowed")

        # Validate benchmark company
        if self.benchmark_company and self.benchmark_company not in self.company_ciks:
            errors.append("Benchmark company must be in the comparison list")

        # Validate dates
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                errors.append("Start date must be before end date")

        # Validate filing types
        if self.filing_types:
            valid_types = ["10-K", "10-Q", "8-K", "20-F", "6-K"]
            for ft in self.filing_types:
                if ft not in valid_types:
                    errors.append(f"Invalid filing type: {ft}")

        # Validate metrics for custom comparison
        if self.comparison_type == ComparisonType.CUSTOM and not self.metrics:
            errors.append("Metrics list required for custom comparison")

        # Validate depth level
        valid_depths = ["standard", "detailed", "executive"]
        if self.depth_level not in valid_depths:
            errors.append(f"Depth level must be one of: {', '.join(valid_depths)}")

        return errors

    def to_dict(self) -> dict:
        """Convert command to dictionary"""
        return {
            "company_ciks": self.company_ciks,
            "comparison_type": self.comparison_type.value,
            "metrics": self.metrics,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "filing_types": self.filing_types,
            "benchmark_company": self.benchmark_company,
            "include_visualizations": self.include_visualizations,
            "depth_level": self.depth_level,
            "request_id": self.request_id,
            "agent_id": self.agent_id
        }


@dataclass
class IndustryComparisonCommand:
    """Command to compare companies within an industry"""

    industry_sic: str
    company_count: int = 5
    comparison_type: ComparisonType = ComparisonType.FINANCIAL
    exclude_company: Optional[str] = None  # CIK to exclude
    request_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate command parameters"""
        errors = []

        # Validate SIC code
        if not self.industry_sic or len(self.industry_sic) != 4 or not self.industry_sic.isdigit():
            errors.append("Industry SIC must be a 4-digit code")

        # Validate company count
        if self.company_count < 2 or self.company_count > 20:
            errors.append("Company count must be between 2 and 20")

        return errors


@dataclass
class PeerComparisonCommand:
    """Command to compare a company with its peers"""

    company_cik: str
    peer_count: int = 4
    comparison_focus: List[str] = None  # e.g., ["size", "profitability", "growth"]
    include_self: bool = True
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.comparison_focus is None:
            self.comparison_focus = ["profitability", "risk"]

    def validate(self) -> List[str]:
        """Validate command parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate peer count
        if self.peer_count < 1 or self.peer_count > 10:
            errors.append("Peer count must be between 1 and 10")

        # Validate comparison focus
        valid_focus = ["size", "profitability", "growth", "risk", "efficiency", "valuation"]
        for focus in self.comparison_focus:
            if focus not in valid_focus:
                errors.append(f"Invalid comparison focus: {focus}")

        return errors