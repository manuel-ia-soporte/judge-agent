# domain/models/agent.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from enum import Enum


class AgentStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class AgentCapability(str, Enum):
    FINANCIAL_ANALYSIS = "financial_analysis"
    SEC_FILING_ANALYSIS = "sec_filing_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    VALUATION = "valuation"
    FORECASTING = "forecasting"
    DATA_EXTRACTION = "data_extraction"


@dataclass
class AgentCapabilities:
    """Value object for agent capabilities"""
    capabilities: List[AgentCapability]
    max_concurrent_tasks: int = 5
    processing_timeout: int = 300  # seconds
    supports_batch: bool = False
    requires_grounding: bool = True

    def can_perform(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability"""
        return capability in self.capabilities


@dataclass
class AgentMetrics:
    """Value object for agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 100.0
    last_active: Optional[datetime] = None
    error_count: int = 0

    def update_success(self, processing_time: float):
        """Update metrics for the successful task"""
        self.tasks_completed += 1
        self.average_processing_time = (
                (self.average_processing_time * (self.tasks_completed - 1) + processing_time)
                / self.tasks_completed
        )
        self.success_rate = (self.tasks_completed /
                             (self.tasks_completed + self.tasks_failed)) * 100
        self.last_active = datetime.now(UTC)

    def update_failure(self):
        """Update metrics for failed task"""
        self.tasks_failed += 1
        self.success_rate = (self.tasks_completed /
                             (self.tasks_completed + self.tasks_failed)) * 100
        self.error_count += 1


@dataclass
class Agent:
    """Domain entity for an agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    capabilities: AgentCapabilities
    status: AgentStatus = AgentStatus.REGISTERED
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    configuration: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None

    def activate(self):
        """Activate the agent"""
        if self.status == AgentStatus.ERROR:
            raise ValueError("Cannot activate agent in error state")
        self.status = AgentStatus.ACTIVE
        self.last_heartbeat = datetime.now(UTC)

    def deactivate(self):
        """Deactivate the agent"""
        self.status = AgentStatus.OFFLINE

    def mark_busy(self):
        """Mark agent as busy"""
        if self.status == AgentStatus.ACTIVE:
            self.status = AgentStatus.BUSY

    def mark_idle(self):
        """Mark agent as idle"""
        if self.status == AgentStatus.BUSY:
            self.status = AgentStatus.IDLE

    def update_heartbeat(self):
        """Update agent heartbeat"""
        self.last_heartbeat = datetime.now(UTC)
        if self.status == AgentStatus.IDLE:
            self.status = AgentStatus.ACTIVE

    def validate_capability(self, required_capability: AgentCapability) -> bool:
        """Validate if agent has required capability"""
        return self.capabilities.can_perform(required_capability)