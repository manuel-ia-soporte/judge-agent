# contracts/events/risk_events.py
"""Risk event contracts"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskEventType(str, Enum):
    """Types of risk events"""
    ASSESSMENT_STARTED = "risk_assessment_started"
    FACTOR_IDENTIFIED = "risk_factor_identified"
    MITIGATION_PROPOSED = "risk_mitigation_proposed"
    ASSESSMENT_COMPLETED = "risk_assessment_completed"
    THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    MONITORING_TRIGGERED = "risk_monitoring_triggered"


class RiskEvent(BaseModel):
    """Base risk event contract"""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: RiskEventType = Field(..., description="Type of event")
    assessment_id: str = Field(..., description="Risk assessment identifier")
    company_cik: str = Field(..., description="Company CIK")
    agent_id: str = Field(..., description="Agent that performed the assessment")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class RiskAssessmentStartedEvent(RiskEvent):
    """Event raised when risk assessment starts"""

    event_type: RiskEventType = RiskEventType.ASSESSMENT_STARTED
    assessment_type: str = Field(..., description="Type of risk assessment")
    scope: List[str] = Field(default_factory=list, description="Assessment scope")
    timeframe: str = Field("current", description="Timeframe for assessment")
    requested_by: Optional[str] = Field(None, description="Who requested the assessment")


class RiskFactorIdentifiedEvent(RiskEvent):
    """Event raised when a risk factor is identified"""

    event_type: RiskEventType = RiskEventType.FACTOR_IDENTIFIED
    risk_id: str = Field(..., description="Risk factor identifier")
    description: str = Field(..., description="Risk description")
    category: str = Field(..., description="Risk category")
    severity: str = Field(..., description="Risk severity")
    probability: float = Field(..., ge=0, le=1, description="Risk probability")
    impact: str = Field(..., description="Risk impact")
    source: str = Field(..., description="Source of risk information")
    is_new: bool = Field(True, description="Whether this is a new risk")


class RiskMitigationProposedEvent(RiskEvent):
    """Event raised when risk mitigation is proposed"""

    event_type: RiskEventType = RiskEventType.MITIGATION_PROPOSED
    risk_id: str = Field(..., description="Risk factor identifier")
    mitigation_id: str = Field(..., description="Mitigation identifier")
    description: str = Field(..., description="Mitigation description")
    effectiveness: float = Field(..., ge=0, le=1, description="Expected effectiveness")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost")
    implementation_timeframe: Optional[str] = Field(None, description="Implementation timeframe")
    proposed_by: str = Field(..., description="Who proposed the mitigation")


class RiskAssessmentCompletedEvent(RiskEvent):
    """Event raised when risk assessment is completed"""

    event_type: RiskEventType = RiskEventType.ASSESSMENT_COMPLETED
    risk_count: int = Field(..., description="Total number of risks identified")
    overall_risk_score: float = Field(..., ge=0, le=1, description="Overall risk score")
    risk_level: str = Field(..., description="Overall risk level")
    high_severity_count: int = Field(0, description="Number of high severity risks")
    mitigations_proposed: int = Field(0, description="Number of mitigations proposed")
    assessment_duration_seconds: float = Field(..., description="Assessment duration")
    next_review_date: Optional[datetime] = Field(None, description="Next review date")


class RiskThresholdExceededEvent(RiskEvent):
    """Event raised when a risk threshold is exceeded"""

    event_type: RiskEventType = RiskEventType.THRESHOLD_EXCEEDED
    risk_id: str = Field(..., description="Risk factor identifier")
    threshold_type: str = Field(..., description="Type of threshold")
    threshold_value: float = Field(..., description="Threshold value")
    actual_value: float = Field(..., description="Actual value")
    exceeded_by: float = Field(..., description="Amount exceeded by")
    severity: str = Field(..., description="Severity of threshold breach")
    automatic_actions: List[str] = Field(default_factory=list, description="Automatic actions triggered")


class RiskMonitoringTriggeredEvent(RiskEvent):
    """Event raised when risk monitoring is triggered"""

    event_type: RiskEventType = RiskEventType.MONITORING_TRIGGERED
    monitoring_id: str = Field(..., description="Monitoring identifier")
    trigger_type: str = Field(..., description="Type of trigger")
    trigger_value: float = Field(..., description="Trigger value")
    monitoring_frequency: str = Field(..., description="Monitoring frequency")
    monitoring_duration: Optional[str] = Field(None, description="Monitoring duration")
    alert_recipients: List[str] = Field(default_factory=list, description="Who to alert")