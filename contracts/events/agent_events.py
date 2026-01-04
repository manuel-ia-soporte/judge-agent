# contracts/events/agent_events.py
"""Agent event contracts"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AgentEventType(str, Enum):
    """Types of agent events"""
    REGISTERED = "agent_registered"
    DEREGISTERED = "agent_deregistered"
    STATUS_CHANGED = "agent_status_changed"
    CAPABILITIES_UPDATED = "agent_capabilities_updated"
    TASK_ASSIGNED = "agent_task_assigned"
    TASK_COMPLETED = "agent_task_completed"
    TASK_FAILED = "agent_task_failed"
    HEALTH_CHANGED = "agent_health_changed"
    METRICS_UPDATED = "agent_metrics_updated"


class AgentEvent(BaseModel):
    """Base agent event contract"""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: AgentEventType = Field(..., description="Type of event")
    agent_id: str = Field(..., description="Agent identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class AgentRegisteredEvent(AgentEvent):
    """Event raised when an agent is registered"""

    event_type: AgentEventType = AgentEventType.REGISTERED
    agent_name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Agent type")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    max_concurrent_tasks: int = Field(5, description="Maximum concurrent tasks")
    version: str = Field("1.0.0", description="Agent version")
    host: Optional[str] = Field(None, description="Agent host")
    port: Optional[int] = Field(None, description="Agent port")


class AgentDeregisteredEvent(AgentEvent):
    """Event raised when an agent is deregistered"""

    event_type: AgentEventType = AgentEventType.DEREGISTERED
    reason: str = Field(..., description="Reason for deregistration")
    deregistered_by: Optional[str] = Field(None, description="Who deregistered the agent")
    tasks_transferred: int = Field(0, description="Number of tasks transferred")
    uptime_seconds: Optional[float] = Field(None, description="Total uptime")


class AgentStatusChangedEvent(AgentEvent):
    """Event to raise when agent status changes"""

    event_type: AgentEventType = AgentEventType.STATUS_CHANGED
    old_status: str = Field(..., description="Previous status")
    new_status: str = Field(..., description="New status")
    reason: Optional[str] = Field(None, description="Reason for status change")
    expected_duration: Optional[str] = Field(None, description="Expected duration of status")
    tasks_affected: List[str] = Field(default_factory=list, description="Tasks affected by status change")


class AgentCapabilitiesUpdatedEvent(AgentEvent):
    """Event raised when agent capabilities are updated"""

    event_type: AgentEventType = AgentEventType.CAPABILITIES_UPDATED
    added_capabilities: List[str] = Field(default_factory=list, description="Added capabilities")
    removed_capabilities: List[str] = Field(default_factory=list, description="Removed capabilities")
    updated_capabilities: List[str] = Field(default_factory=list, description="Updated capabilities")
    reason: Optional[str] = Field(None, description="Reason for update")
    updated_by: Optional[str] = Field(None, description="Who updated the capabilities")


class AgentTaskAssignedEvent(AgentEvent):
    """Event raised when a task is assigned to an agent"""

    event_type: AgentEventType = AgentEventType.TASK_ASSIGNED
    task_id: str = Field(..., description="Task identifier")
    task_type: str = Field(..., description="Task type")
    priority: int = Field(1, description="Task priority (1-10)")
    expected_duration_seconds: Optional[float] = Field(None, description="Expected duration")
    assigned_by: Optional[str] = Field(None, description="Who assigned the task")
    queue_position: Optional[int] = Field(None, description="Position in queue")


class AgentTaskCompletedEvent(AgentEvent):
    """Event raised when an agent completes a task"""

    event_type: AgentEventType = AgentEventType.TASK_COMPLETED
    task_id: str = Field(..., description="Task identifier")
    processing_time_seconds: float = Field(..., description="Processing time")
    success: bool = Field(True, description="Whether task was successful")
    result_size_bytes: Optional[int] = Field(None, description="Size of result")
    quality_score: Optional[float] = Field(None, description="Quality score (0-1)")
    next_task_id: Optional[str] = Field(None, description="Next task if any")


class AgentTaskFailedEvent(AgentEvent):
    """Event raised when an agent fails a task"""

    event_type: AgentEventType = AgentEventType.TASK_FAILED
    task_id: str = Field(..., description="Task identifier")
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    retry_count: int = Field(0, description="Number of retry attempts")
    can_retry: bool = Field(True, description="Whether task can be retried")
    assigned_to_new_agent: Optional[str] = Field(None, description="New agent if reassigned")


class AgentHealthChangedEvent(AgentEvent):
    """Event to raise when agent health changes"""

    event_type: AgentEventType = AgentEventType.HEALTH_CHANGED
    old_health_score: float = Field(..., ge=0, le=100, description="Previous health score")
    new_health_score: float = Field(..., ge=0, le=100, description="New health score")
    health_status: str = Field(..., description="Health status")
    issues: List[str] = Field(default_factory=list, description="Health issues")
    warnings: List[str] = Field(default_factory=list, description="Health warnings")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat")


class AgentMetricsUpdatedEvent(AgentEvent):
    """Event raised when agent metrics are updated"""

    event_type: AgentEventType = AgentEventType.METRICS_UPDATED
    metrics: Dict[str, Any] = Field(..., description="Updated metrics")
    update_interval_seconds: float = Field(60.0, description="Update interval")
    significant_changes: List[str] = Field(default_factory=list, description="Significant changes")
    performance_score: Optional[float] = Field(None, description="Performance score")
    resource_utilization: Optional[Dict[str, float]] = Field(None, description="Resource utilization")