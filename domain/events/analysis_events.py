# domain/events/analysis_events.py
"""Domain events for the analysis system"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base domain event class"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    aggregate_id: str = ""
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "event_type": self.__class__.__name__,
            "version": self.version
        }


@dataclass
class AnalysisStarted(DomainEvent):
    """Event raised when analysis starts"""
    company_cik: str = ""
    analysis_type: str = ""
    agent_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "company_cik": self.company_cik,
            "analysis_type": self.analysis_type,
            "agent_id": self.agent_id,
            "metadata": self.metadata
        })
        return base


@dataclass
class DocumentsFetched(DomainEvent):
    """Event raised when SEC documents are fetched"""
    document_count: int = 0
    document_types: List[str] = field(default_factory=list)
    fetch_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "document_count": self.document_count,
            "document_types": self.document_types,
            "fetch_duration_seconds": self.fetch_duration_seconds
        })
        return base


@dataclass
class MetricsExtracted(DomainEvent):
    """Event raised when financial metrics are extracted"""
    metric_count: int = 0
    metric_categories: List[str] = field(default_factory=list)
    extraction_quality: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "metric_count": self.metric_count,
            "metric_categories": self.metric_categories,
            "extraction_quality": self.extraction_quality
        })
        return base


@dataclass
class RiskFactorsIdentified(DomainEvent):
    """Event raised when risk factors are identified"""
    risk_count: int = 0
    risk_categories: Dict[str, int] = field(default_factory=dict)
    overall_risk_level: str = ""

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "risk_count": self.risk_count,
            "risk_categories": self.risk_categories,
            "overall_risk_level": self.overall_risk_level
        })
        return base


@dataclass
class AnalysisCompleted(DomainEvent):
    """Event raised when analysis is completed"""
    analysis_duration_seconds: float = 0.0
    conclusions_count: int = 0
    overall_score: float = 0.0
    status: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "analysis_duration_seconds": self.analysis_duration_seconds,
            "conclusions_count": self.conclusions_count,
            "overall_score": self.overall_score,
            "status": self.status
        })
        return base


@dataclass
class AnalysisFailed(DomainEvent):
    """Event raised when analysis fails"""
    error_message: str = ""
    error_type: str = ""
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count
        })
        return base


@dataclass
class MetricAddedEvent(DomainEvent):
    """Event raised when a metric is added to analysis"""
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    metric_period: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_unit": self.metric_unit,
            "metric_period": self.metric_period.isoformat()
        })
        return base


@dataclass
class RiskAssessmentUpdated(DomainEvent):
    """Event raised when risk assessment is updated"""
    previous_risk_level: str = ""
    new_risk_level: str = ""
    risk_score_change: float = 0.0
    updated_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "previous_risk_level": self.previous_risk_level,
            "new_risk_level": self.new_risk_level,
            "risk_score_change": self.risk_score_change,
            "updated_factors": self.updated_factors
        })
        return base


@dataclass
class AnalysisPublished(DomainEvent):
    """Event raised when analysis is published"""
    published_to: List[str] = field(default_factory=list)
    access_level: str = "private"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "published_to": self.published_to,
            "access_level": self.access_level,
            "version": self.version
        })
        return base


class EventHandler:
    """Base event handler"""

    @staticmethod
    def handle_event(event: DomainEvent) -> None:
        """Handle domain event"""
        # This would be implemented by concrete event handlers
        # For now, just log or process as needed
        pass


class EventPublisher:
    """Simple event publisher"""

    def __init__(self):
        self._subscribers = []

    def subscribe(self, handler):
        """Subscribe to events"""
        self._subscribers.append(handler)

    def publish(self, event: DomainEvent):
        """Publish event to all subscribers"""
        for subscriber in self._subscribers:
            try:
                subscriber.handle_event(event)
            except Exception as e:
                # Log error but continue with other subscribers
                print(f"Error in event handler: {e}")