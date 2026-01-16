# domain/models/finance.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class FilingType(str, Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_S1 = "S-1"
    FORM_4 = "4"


class FinancialMetric(str, Enum):
    REVENUE = "revenue"
    NET_INCOME = "net_income"
    EPS = "eps"
    ASSETS = "total_assets"
    LIABILITIES = "total_liabilities"
    EQUITY = "shareholders_equity"
    CASH_FLOW = "operating_cash_flow"
    DEBT = "total_debt"


@dataclass
class SECDocument:
    """Domain entity for SEC documents"""
    document_id: str
    company_cik: str
    company_name: str
    filing_type: FilingType
    filing_date: datetime
    period_end: datetime
    document_url: str
    content: Dict[str, Any]  # Parsed document sections
    raw_text: Optional[str] = None
    items: Dict[str, str] = field(default_factory=dict)  # Item 1A, 7, etc.

    def get_item_content(self, item_number: str) -> Optional[str]:
        """Get content for specific item"""
        return self.items.get(f"Item {item_number}")

    def validate_completeness(self) -> List[str]:
        """Validate document has required sections"""
        missing_items = []
        required_items = ["1A", "7", "8"] if self.filing_type == FilingType.FORM_10K else ["2"]

        for item in required_items:
            if f"Item {item}" not in self.items:
                missing_items.append(f"Item {item}")

        return missing_items


@dataclass
class FinancialAnalysis:
    """Aggregate root for financial analysis"""
    analysis_id: str
    agent_id: str
    company_ticker: str
    analysis_date: datetime
    content: str
    metrics_used: List[Any]  # Can be FinancialMetric or str
    source_documents: List[Any]  # Can be SECDocument or dict
    conclusions: List[str]
    risks_identified: List[str]
    assumptions: List[str]
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cited_sources(self) -> List[Dict[str, str]]:
        """Extract cited sources from analysis"""
        sources = []
        for doc in self.source_documents:
            sources.append({
                "cik": doc.company_cik,
                "type": doc.filing_type.value,
                "date": doc.filing_date.isoformat(),
                "url": doc.document_url
            })
        return sources

    def validate_source_citations(self) -> bool:
        """Validate that analysis properly cites sources"""
        # This would implement logic to check if numbers/facts are cited
        return len(self.source_documents) > 0