# contracts/finance_contracts.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class FilingStatus(str, Enum):
    FILED = "filed"
    AMENDED = "amended"
    WITHDRAWN = "withdrawn"


class FinancialStatementType(str, Enum):
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    CHANGES_IN_EQUITY = "changes_in_equity"


class SECFilingRequest(BaseModel):
    """Contract for SEC filing requests"""
    company_cik: str = Field(..., description="Central Index Key")
    filing_type: str = Field(..., description="10-K, 10-Q, 8-K, etc.")
    period_end: Optional[datetime] = Field(None, description="Filing period end date")
    accession_number: Optional[str] = Field(None, description="SEC accession number")
    include_attachments: bool = Field(default=False)

    @validator('company_cik')
    def validate_cik(cls, v):
        if not v.isdigit() or len(v) > 10:
            raise ValueError("CIK must be numeric and up to 10 digits")
        return v.zfill(10)


class FinancialMetricData(BaseModel):
    """Financial metric data point"""
    metric_name: str = Field(...)
    value: float = Field(...)
    unit: str = Field(default="USD")
    period: datetime = Field(...)
    source_document: str = Field(...)
    footnote: Optional[str] = None
    is_estimated: bool = Field(default=False)
    confidence: float = Field(default=1.0, ge=0, le=1)


class CompanyFinancials(BaseModel):
    """Complete company financial data"""
    company_cik: str = Field(...)
    company_name: str = Field(...)
    ticker: Optional[str] = None
    fiscal_year_end: str = Field(...)
    filings: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, List[FinancialMetricData]] = Field(default_factory=dict)
    risk_factors: List[str] = Field(default_factory=list)
    management_discussion: Optional[str] = None
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)

    def get_latest_metric(self, metric_name: str) -> Optional[FinancialMetricData]:
        """Get the latest value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sorted(self.metrics[metric_name],
                          key=lambda x: x.period,
                          reverse=True)[0]
        return None


class MarketDataRequest(BaseModel):
    """Contract for market data requests"""
    ticker: str = Field(...)
    start_date: datetime = Field(...)
    end_date: datetime = Field(...)
    interval: str = Field(default="1d", pattern="^(1d|1wk|1mo)$")
    metrics: List[str] = Field(default_factory=lambda: ["close", "volume"])

    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v


class RiskAssessment(BaseModel):
    """Risk assessment contract"""
    assessment_id: str = Field(...)
    company_cik: str = Field(...)
    assessment_date: datetime = Field(default_factory=datetime.now)
    risk_categories: Dict[str, float] = Field(...)  # category -> score
    overall_risk_score: float = Field(..., ge=0, le=1)
    mitigations: List[str] = Field(default_factory=list)
    monitoring_indicators: List[str] = Field(default_factory=list)
    next_review_date: Optional[datetime] = None