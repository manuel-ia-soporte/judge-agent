# domain/models/entities.py
"""Domain entities with business logic and invariants."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
from .enums import AnalysisType


@dataclass
class Entity:
    """Base class for all domain entities"""
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class SECDocument(Entity):
    """SEC Document Entity with business invariants"""
    company_cik: str
    company_name: str
    filing_type: str  # 10-K, 10-Q, 8-K, etc.
    filing_date: datetime
    period_end: datetime
    document_url: str
    content: Dict[str, Any]  # Parsed content structure
    raw_text: Optional[str] = None
    items: Dict[str, str] = field(default_factory=dict)  # Item 1A, Item 7, Item 8, etc.

    def __post_init__(self):
        """Validate entity invariants"""
        if not self.company_cik.isdigit() or len(self.company_cik) < 10:
            raise ValueError("CIK must be a valid 10-digit number")
        if self.filing_date > datetime.now():
            raise ValueError("Filing date cannot be in the future")
        if self.period_end > datetime.now():
            raise ValueError("Period end cannot be in the future")

    def get_item(self, item_name: str) -> Optional[str]:
        """Get specific item content"""
        return self.items.get(item_name.upper())

    def contains_item(self, item_name: str) -> bool:
        """Check if the document contains the specific item"""
        return item_name.upper() in self.items

    @property
    def is_annual(self) -> bool:
        """Check if this is an annual filing"""
        return self.filing_type in ["10-K", "20-F", "40-F"]

    @property
    def is_quarterly(self) -> bool:
        """Check if this is a quarterly filing"""
        return self.filing_type in ["10-Q", "6-K"]


@dataclass
class FinancialAnalysis(Entity):
    """Financial Analysis Aggregate Root"""
    analysis_id: str
    agent_id: str
    company_cik: str
    analysis_type: AnalysisType
    analysis_date: datetime = field(default_factory=datetime.now)
    content: str = ""
    metrics_used: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    risks_identified: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate analysis invariants"""
        if not self.company_cik.isdigit():
            raise ValueError("Company CIK must be numeric")
        if not self.content.strip():
            raise ValueError("Analysis content cannot be empty")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")

    def add_metric(self, metric_name: str):
        """Add metric used in analysis"""
        if metric_name not in self.metrics_used:
            self.metrics_used.append(metric_name)

    def add_source_document(self, document_id: str):
        """Add source document reference"""
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)

    def add_conclusion(self, conclusion: str):
        """Add conclusion to analysis"""
        if conclusion not in self.conclusions:
            self.conclusions.append(conclusion)

    def validate_analysis(self) -> List[str]:
        """Validate analysis completeness"""
        errors = []

        if len(self.metrics_used) == 0:
            errors.append("No metrics used in analysis")
        if len(self.source_documents) == 0:
            errors.append("No source documents referenced")
        if len(self.conclusions) == 0:
            errors.append("No conclusions provided")
        if len(self.risks_identified) == 0:
            errors.append("No risks identified")

        return errors

    @property
    def is_valid(self) -> bool:
        """Check if analysis is valid"""
        return len(self.validate_analysis()) == 0


@dataclass
class Company(Entity):
    """Company Aggregate Root"""
    cik: str
    name: str
    ticker: Optional[str] = None
    sic_code: Optional[str] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate company invariants"""
        if not self.cik.isdigit() or len(self.cik) < 10:
            raise ValueError("CIK must be a valid 10-digit number")
        if not self.name.strip():
            raise ValueError("Company name cannot be empty")

    @property
    def formatted_cik(self) -> str:
        """Return formatted CIK with leading zeros"""
        return self.cik.zfill(10)

    def update_from_filing(self, filing_data: Dict[str, Any]):
        """Update company data from filing"""
        if "companyName" in filing_data:
            self.name = filing_data["companyName"]
        if "ticker" in filing_data:
            self.ticker = filing_data["ticker"]
        if "sic" in filing_data:
            self.sic_code = filing_data["sic"]
        self.updated_at = datetime.now()