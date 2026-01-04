# contracts/events/analysis_events.py
"""Analysis event contracts"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisEventType(str, Enum):
    """Types of analysis events"""
    STARTED = "analysis_started"
    DOCUMENTS_FETCHED = "documents_fetched"
    METRICS_EXTRACTED = "metrics_extracted"
    RISKS_IDENTIFIED = "risks_identified"
    COMPLETED = "analysis_completed"
    FAILED = "analysis_failed"
    VALIDATED = "analysis_validated"
    PUBLISHED = "analysis_published"


class AnalysisEvent(BaseModel):
    """Base analysis event contract"""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: AnalysisEventType = Field(..., description="Type of event")
    analysis_id: str = Field(..., description="Analysis identifier")
    agent_id: str = Field(..., description="Agent that performed the analysis")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class AnalysisStartedEvent(AnalysisEvent):
    """Event raised when analysis starts"""

    event_type: AnalysisEventType = AnalysisEventType.STARTED
    company_cik: str = Field(..., description="Company CIK being analyzed")
    analysis_type: str = Field(..., description="Type of analysis")
    requested_by: Optional[str] = Field(None, description="Who requested the analysis")
    expected_completion_time: Optional[datetime] = Field(None, description="Expected completion time")


class AnalysisProgressEvent(AnalysisEvent):
    """Event raised during analysis progress"""

    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage")
    current_step: str = Field(..., description="Current analysis step")
    steps_completed: int = Field(..., description="Number of steps completed")
    total_steps: int = Field(..., description="Total number of steps")


class DocumentsFetchedEvent(AnalysisEvent):
    """Event raised when documents are fetched"""

    event_type: AnalysisEventType = AnalysisEventType.DOCUMENTS_FETCHED
    document_count: int = Field(..., description="Number of documents fetched")
    document_types: List[str] = Field(default_factory=list, description="Types of documents fetched")
    fetch_duration_seconds: float = Field(..., description="Time taken to fetch documents")
    source: str = Field("SEC EDGAR", description="Source of documents")


class MetricsExtractedEvent(AnalysisEvent):
    """Event raised when metrics are extracted"""

    event_type: AnalysisEventType = AnalysisEventType.METRICS_EXTRACTED
    metric_count: int = Field(..., description="Number of metrics extracted")
    metric_categories: List[str] = Field(default_factory=list, description="Categories of metrics")
    extraction_quality: float = Field(..., ge=0, le=1, description="Quality of extraction")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors if any")


class RisksIdentifiedEvent(AnalysisEvent):
    """Event raised when risks are identified"""

    event_type: AnalysisEventType = AnalysisEventType.RISKS_IDENTIFIED
    risk_count: int = Field(..., description="Number of risks identified")
    risk_categories: Dict[str, int] = Field(default_factory=dict, description="Risk categories and counts")
    overall_risk_level: str = Field(..., description="Overall risk level")
    high_severity_count: int = Field(0, description="Number of high severity risks")


class AnalysisCompletedEvent(AnalysisEvent):
    """Event raised when analysis is completed"""

    event_type: AnalysisEventType = AnalysisEventType.COMPLETED
    analysis_duration_seconds: float = Field(..., description="Total analysis duration")
    conclusions_count: int = Field(..., description="Number of conclusions generated")
    overall_score: float = Field(..., ge=0, le=2, description="Overall analysis score")
    passed: bool = Field(..., description="Whether analysis passed requirements")
    recommendations_count: int = Field(..., description="Number of recommendations")
    warnings_count: int = Field(..., description="Number of warnings")


class AnalysisFailedEvent(AnalysisEvent):
    """Event raised when analysis fails"""

    event_type: AnalysisEventType = AnalysisEventType.FAILED
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    retry_count: int = Field(0, description="Number of retry attempts")
    can_retry: bool = Field(True, description="Whether analysis can be retried")


class AnalysisValidatedEvent(AnalysisEvent):
    """Event raised when analysis is validated"""

    event_type: AnalysisEventType = AnalysisEventType.VALIDATED
    validator_id: str = Field(..., description="Validator agent ID")
    validation_score: float = Field(..., ge=0, le=1, description="Validation score")
    validation_issues: List[str] = Field(default_factory=list, description="Validation issues found")
    is_approved: bool = Field(..., description="Whether analysis was approved")


class AnalysisPublishedEvent(AnalysisEvent):
    """Event raised when analysis is published"""

    event_type: AnalysisEventType = AnalysisEventType.PUBLISHED
    published_to: List[str] = Field(default_factory=list, description="Where analysis was published")
    access_level: str = Field("private", description="Access level for published analysis")
    version: str = Field("1.0", description="Analysis version")
    checksum: Optional[str] = Field(None, description="Checksum for integrity verification")