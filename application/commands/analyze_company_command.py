# application/commands/analyze_company_command.py
"""Commands for company analysis"""

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List, Optional
from enum import Enum


class AnalysisType(str, Enum):
    """Types of analysis that can be requested"""
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    RISK = "risk"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    QUICK = "quick"
    COMPARATIVE = "comparative"


@dataclass
class AnalyzeCompanyCommand:
    """Command to analyze a company"""

    company_cik: str
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filing_types: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    include_trends: bool = True
    include_comparisons: bool = False
    benchmark_company: Optional[str] = None
    depth_level: str = "standard"  # standard, detailed, executive
    request_id: Optional[str] = None
    agent_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate command parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate dates
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                errors.append("Start date must be before end date")
            if self.start_date > datetime.now(UTC):
                errors.append("Start date cannot be in the future")

        # Validate filing types
        if self.filing_types:
            valid_types = ["10-K", "10-Q", "8-K", "20-F", "6-K", "S-1"]
            for ft in self.filing_types:
                if ft not in valid_types:
                    errors.append(f"Invalid filing type: {ft}")

        # Validate depth level
        valid_depths = ["standard", "detailed", "executive"]
        if self.depth_level not in valid_depths:
            errors.append(f"Depth level must be one of: {', '.join(valid_depths)}")

        return errors

    def to_dict(self) -> dict:
        """Convert command to dictionary"""
        return {
            "company_cik": self.company_cik,
            "analysis_type": self.analysis_type.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "filing_types": self.filing_types,
            "metrics": self.metrics,
            "include_trends": self.include_trends,
            "include_comparisons": self.include_comparisons,
            "benchmark_company": self.benchmark_company,
            "depth_level": self.depth_level,
            "request_id": self.request_id,
            "agent_id": self.agent_id
        }


@dataclass
class QuickAnalysisCommand:
    """Command for quick company analysis"""

    company_cik: str
    focus_areas: List[str]  # e.g., ["financial", "risk", "operations"]
    max_documents: int = 2
    include_key_metrics: bool = True
    include_risks: bool = True
    request_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate command parameters"""
        errors = []

        # Validate CIK
        if not self.company_cik or not self.company_cik.isdigit():
            errors.append("Company CIK must be numeric")

        # Validate focus areas
        valid_areas = ["financial", "risk", "operations", "strategy", "compliance"]
        for area in self.focus_areas:
            if area not in valid_areas:
                errors.append(f"Invalid focus area: {area}")

        # Validate max documents
        if self.max_documents < 1 or self.max_documents > 5:
            errors.append("Max documents must be between 1 and 5")

        return errors


@dataclass
class UpdateAnalysisCommand:
    """Command to update an existing analysis"""

    analysis_id: str
    new_metrics: Optional[List[dict]] = None
    new_conclusions: Optional[List[str]] = None
    new_risks: Optional[List[dict]] = None
    update_reason: Optional[str] = None
    agent_id: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate command parameters"""
        errors = []

        if not self.analysis_id:
            errors.append("Analysis ID is required")

        if not any([self.new_metrics, self.new_conclusions, self.new_risks]):
            errors.append("At least one update field must be provided")

        return errors