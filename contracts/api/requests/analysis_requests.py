# contracts/api/requests/analysis_requests.py
from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeCompanyRequest(BaseModel):
    company_cik: str = Field(..., description="SEC CIK identifier")
    analysis_type: str = Field(..., description="Type of analysis requested")
    include_risk: bool = Field(default=True)
    include_operational: bool = Field(default=True)
    include_strategic: bool = Field(default=True)


class CompareCompaniesRequest(BaseModel):
    primary_cik: str = Field(..., description="Primary company CIK")
    peer_ciks: list[str] = Field(..., description="Peer company CIKs")
    analysis_type: Optional[str] = None
