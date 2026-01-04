# contracts/api/requests/analysis_requests.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisType(str, Enum):
    COMPREHENSIVE = "comprehensive"
    FINANCIAL = "financial"
    RISK = "risk"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    QUICK = "quick"


class AnalyzeCompanyRequest(BaseModel):
    """Request contract for company analysis"""

    company_cik: str = Field(..., description="Company CIK number")
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE)
    start_date: Optional[datetime] = Field(default=None, description="Start date for filings")
    end_date: Optional[datetime] = Field(default_factory=datetime.now)
    metrics: Optional[List[str]] = Field(default=None, description="Specific metrics to extract")
    include_trends: bool = Field(default=True)

    @validator('company_cik')
    def validate_cik(cls, v):
        if not v.isdigit() or len(v) > 10:
            raise ValueError("CIK must be numeric and up to 10 digits")
        return v.zfill(10)

    @validator('start_date')
    def validate_dates(cls, v, values):
        if v and 'end_date' in values and values['end_date'] and v > values['end_date']:
            raise ValueError("start_date must be before end_date")
        return v


class CompareCompaniesRequest(BaseModel):
    """Request contract for company comparison"""

    company_ciks: List[str] = Field(..., min_items=2, max_items=10)
    comparison_type: AnalysisType = Field(default=AnalysisType.FINANCIAL)
    metrics: List[str] = Field(default=["revenue", "net_income", "current_ratio", "debt_to_equity"])
    benchmark_company: Optional[str] = Field(default=None)

    @validator('company_ciks')
    def validate_unique_ciks(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("Company CIKs must be unique")
        return v


class GetFinancialMetricsRequest(BaseModel):
    """Request contract for financial metrics"""

    company_cik: str
    metric_names: List[str]
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    include_historical: bool = Field(default=False)

    class Config:
        schema_extra = {
            "example": {
                "company_cik": "0000320193",
                "metric_names": ["revenue", "net_income", "total_assets"],
                "period_start": "2022-01-01T00:00:00Z",
                "period_end": "2023-12-31T23:59:59Z"
            }
        }