# domain/models/entities.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum
from uuid import uuid4
from .value_objects import FinancialMetric, FinancialRatio, RiskFactor, TrendAnalysis


class AnalysisType(Enum):
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    RISK = "risk"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    QUICK = "quick"


@dataclass
class SECDocument:
    """Entity for SEC Document (has identity)"""
    document_id: str
    company_cik: str
    company_name: str
    filing_type: str
    filing_date: datetime
    period_end: datetime
    document_url: str
    content: Dict[str, Any]
    raw_text: str
    items: Dict[str, str] = field(default_factory=dict)

    def validate_completeness(self) -> bool:
        """Validate document has required sections"""
        required_items = ["Item 1A", "Item 7", "Item 8"] if self.filing_type == "10-K" else ["Item 2"]
        return all(item in self.items for item in required_items)


@dataclass
class FinancialAnalysis:  # Aggregate Root
    """Aggregate Root for Financial Analysis"""
    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    company_cik: str = ""
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    analysis_date: datetime = field(default_factory=datetime.now)

    # Value Objects
    metrics: List[FinancialMetric] = field(default_factory=list)
    ratios: List[FinancialRatio] = field(default_factory=list)
    risk_factors: List[RiskFactor] = field(default_factory=list)
    trends: List[TrendAnalysis] = field(default_factory=list)

    # Business Logic
    conclusions: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Domain Events
    _domain_events: List[Any] = field(default_factory=list, init=False)

    def add_metric(self, metric: FinancialMetric):
        """Add financial metric with validation"""
        if not any(m.name == metric.name for m in self.metrics):
            self.metrics.append(metric)
            self._domain_events.append(MetricAddedEvent(self.analysis_id, metric))

    def calculate_ratios(self):
        """Calculate financial ratios from metrics"""
        # This would implement ratio calculation logic
        pass

    def assess_risk_level(self) -> str:
        """Assess overall risk level"""
        if not self.risk_factors:
            return "unknown"

        high_risks = sum(1 for r in self.risk_factors if r.severity == "high")
        total_risks = len(self.risk_factors)

        if high_risks / total_risks > 0.3:
            return "high"
        elif high_risks / total_risks > 0.1:
            return "medium"
        return "low"

    def get_domain_events(self) -> List[Any]:
        """Get and clear domain events"""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events