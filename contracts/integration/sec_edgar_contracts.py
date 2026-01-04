# contracts/integration/sec_edgar_contracts.py
"""SEC EDGAR integration contracts"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SECFilingType(str, Enum):
    """SEC filing types"""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_20F = "20-F"
    FORM_6K = "6-K"
    FORM_S1 = "S-1"
    FORM_4 = "4"
    FORM_5 = "5"


class SECFilingRequest(BaseModel):
    """Request for SEC filing data"""

    company_cik: str = Field(..., description="Company CIK number")
    filing_type: Optional[SECFilingType] = Field(None, description="Type of filing")
    start_date: Optional[datetime] = Field(None, description="Start date for filings")
    end_date: Optional[datetime] = Field(None, description="End date for filings")
    accession_number: Optional[str] = Field(None, description="Specific accession number")
    include_attachments: bool = Field(False, description="Include filing attachments")
    include_xbrl: bool = Field(True, description="Include XBRL data")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of filings")

    @validator('company_cik')
    def validate_cik(cls, v):
        if not v.isdigit() or len(v) > 10:
            raise ValueError("CIK must be numeric and up to 10 digits")
        return v.zfill(10)

    @validator('end_date')
    def validate_dates(cls, v, values):
        if v and 'start_date' in values and values['start_date'] and v < values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v


class SECFilingMetadata(BaseModel):
    """Metadata for SEC filing"""

    accession_number: str = Field(..., description="Accession number")
    filing_date: datetime = Field(..., description="Filing date")
    report_date: Optional[datetime] = Field(None, description="Report date")
    acceptance_datetime: datetime = Field(..., description="Acceptance datetime")
    form: str = Field(..., description="Form type")
    filing_url: str = Field(..., description="Filing URL")
    size_bytes: int = Field(..., description="Filing size in bytes")
    is_xbrl: bool = Field(False, description="Whether filing includes XBRL")
    is_inline_xbrl: bool = Field(False, description="Whether filing uses inline XBRL")


class SECCompanyInfo(BaseModel):
    """Company information from SEC"""

    cik: str = Field(..., description="Central Index Key")
    name: str = Field(..., description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker")
    sic: Optional[str] = Field(None, description="Standard Industrial Classification")
    sic_description: Optional[str] = Field(None, description="SIC description")
    state_location: Optional[str] = Field(None, description="State of location")
    state_incorporation: Optional[str] = Field(None, description="State of incorporation")
    fiscal_year_end: Optional[str] = Field(None, description="Fiscal year end")


class SECFilingData(BaseModel):
    """SEC filing data"""

    metadata: SECFilingMetadata = Field(..., description="Filing metadata")
    company_info: SECCompanyInfo = Field(..., description="Company information")
    content: Optional[Dict[str, Any]] = Field(None, description="Filing content (structured)")
    raw_text: Optional[str] = Field(None, description="Raw filing text")
    xbrl_data: Optional[Dict[str, Any]] = Field(None, description="XBRL data")
    extracted_items: Dict[str, str] = Field(default_factory=dict, description="Extracted items")

    class Config:
        arbitrary_types_allowed = True


class SECFilingResponse(BaseModel):
    """Response with SEC filing data"""

    request: SECFilingRequest = Field(..., description="Original request")
    filings: List[SECFilingData] = Field(default_factory=list, description="Filing data")
    total_filings: int = Field(0, description="Total filings available")
    retrieved_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    source: str = Field("SEC EDGAR", description="Data source")


class SECCompanyFactsRequest(BaseModel):
    """Request for SEC company facts"""

    company_cik: str = Field(..., description="Company CIK")
    concepts: Optional[List[str]] = Field(None, description="Specific concepts to retrieve")
    include_all: bool = Field(False, description="Include all concepts")
    taxonomy: str = Field("us-gaap", description="XBRL taxonomy")

    @validator('company_cik')
    def validate_cik(cls, v):
        if not v.isdigit() or len(v) > 10:
            raise ValueError("CIK must be numeric and up to 10 digits")
        return v.zfill(10)


class SECConceptData(BaseModel):
    """Data for a specific XBRL concept"""

    concept: str = Field(..., description="Concept name")
    label: str = Field(..., description="Concept label")
    description: Optional[str] = Field(None, description="Concept description")
    units: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Concept data by unit")
    taxonomy: str = Field(..., description="XBRL taxonomy")


class SECCompanyFacts(BaseModel):
    """SEC company facts (XBRL data)"""

    cik: str = Field(..., description="Company CIK")
    name: str = Field(..., description="Company name")
    facts: Dict[str, Dict[str, SECConceptData]] = Field(default_factory=dict, description="Company facts")
    retrieved_at: datetime = Field(default_factory=datetime.now)


class SECSearchRequest(BaseModel):
    """Request for SEC search"""

    query: str = Field(..., description="Search query")
    filing_types: Optional[List[SECFilingType]] = Field(None, description="Filter by filing types")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    companies: Optional[List[str]] = Field(None, description="Filter by companies")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results")


class SECSearchResult(BaseModel):
    """SEC search result"""

    rank: int = Field(..., description="Result rank")
    score: float = Field(..., description="Search score")
    filing: SECFilingMetadata = Field(..., description="Filing metadata")
    company_info: SECCompanyInfo = Field(..., description="Company information")
    snippets: List[str] = Field(default_factory=list, description="Search snippets")


class SECSearchResponse(BaseModel):
    """Response with SEC search results"""

    request: SECSearchRequest = Field(..., description="Original request")
    results: List[SECSearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(0, description="Total results available")
    retrieved_at: datetime = Field(default_factory=datetime.now)
    search_time_ms: float = Field(..., description="Search time in milliseconds")