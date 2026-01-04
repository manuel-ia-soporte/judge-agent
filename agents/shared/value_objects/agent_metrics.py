# agents/shared/value_objects/agent_metrics.py
"""Shared value objects for agent metrics"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent"""

    # Task counts
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    timeout_tasks: int = 0

    # Timing metrics
    average_processing_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    fastest_task_ms: float = 0.0
    slowest_task_ms: float = 0.0

    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    # Resource metrics
    average_memory_usage_mb: float = 0.0
    average_cpu_usage_percent: float = 0.0
    peak_memory_usage_mb: float = 0.0
    peak_cpu_usage_percent: float = 0.0

    # Timestamps
    first_task_time: Optional[datetime] = None
    last_task_time: Optional[datetime] = None
    metrics_calculated_at: datetime = field(default_factory=datetime.now)

    def update_task_completed(self, processing_time_ms: float):
        """Update metrics for the completed task"""
        self.total_tasks += 1
        self.completed_tasks += 1

        # Update timing metrics
        self.total_processing_time_ms += processing_time_ms
        self.average_processing_time_ms = (
                self.total_processing_time_ms / self.completed_tasks
        )

        if self.fastest_task_ms == 0 or processing_time_ms < self.fastest_task_ms:
            self.fastest_task_ms = processing_time_ms

        if processing_time_ms > self.slowest_task_ms:
            self.slowest_task_ms = processing_time_ms

        # Update timestamps
        now = datetime.now(UTC)
        if not self.first_task_time:
            self.first_task_time = now
        self.last_task_time = now

        # Recalculate rates
        self._recalculate_rates()

    def update_task_failed(self, processing_time_ms: float = 0):
        """Update metrics for failed task"""
        self.total_tasks += 1
        self.failed_tasks += 1

        # Update timestamps
        now = datetime.now(UTC)
        if not self.first_task_time:
            self.first_task_time = now
        self.last_task_time = now

        # Recalculate rates
        self._recalculate_rates()

    def update_task_cancelled(self):
        """Update metrics for the canceled task"""
        self.total_tasks += 1
        self.cancelled_tasks += 1
        self._recalculate_rates()

    def update_task_timeout(self):
        """Update metrics for the timed out task"""
        self.total_tasks += 1
        self.timeout_tasks += 1
        self._recalculate_rates()

    def update_resource_usage(self, memory_mb: float, cpu_percent: float):
        """Update resource usage metrics"""
        # Update memory metrics
        self.average_memory_usage_mb = (
            (self.average_memory_usage_mb * (self.total_tasks - 1) + memory_mb) / self.total_tasks
            if self.total_tasks > 0 else memory_mb
        )

        if memory_mb > self.peak_memory_usage_mb:
            self.peak_memory_usage_mb = memory_mb

        # Update CPU metrics
        self.average_cpu_usage_percent = (
            (self.average_cpu_usage_percent * (self.total_tasks - 1) + cpu_percent) / self.total_tasks
            if self.total_tasks > 0 else cpu_percent
        )

        if cpu_percent > self.peak_cpu_usage_percent:
            self.peak_cpu_usage_percent = cpu_percent

    def _recalculate_rates(self):
        """Recalculate success, error, and timeout rates"""
        if self.total_tasks > 0:
            self.success_rate = self.completed_tasks / self.total_tasks
            self.error_rate = (self.failed_tasks + self.cancelled_tasks) / self.total_tasks
            self.timeout_rate = self.timeout_tasks / self.total_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "timeout_tasks": self.timeout_tasks,
            "average_processing_time_ms": self.average_processing_time_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "fastest_task_ms": self.fastest_task_ms,
            "slowest_task_ms": self.slowest_task_ms,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
            "average_memory_usage_mb": self.average_memory_usage_mb,
            "average_cpu_usage_percent": self.average_cpu_usage_percent,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "peak_cpu_usage_percent": self.peak_cpu_usage_percent,
            "first_task_time": self.first_task_time.isoformat() if self.first_task_time else None,
            "last_task_time": self.last_task_time.isoformat() if self.last_task_time else None,
            "metrics_calculated_at": self.metrics_calculated_at.isoformat()
        }


@dataclass
class AgentHealthStatus:
    """Health status of an agent"""

    agent_id: str
    status: AgentStatus = AgentStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    is_healthy: bool = False
    health_score: float = 0.0  # 0-100 scale

    # Health checks
    can_connect: bool = False
    can_process_tasks: bool = False
    resources_available: bool = False
    dependencies_healthy: bool = False

    # Issues
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
    checked_at: datetime = field(default_factory=datetime.now)

    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.now(UTC)

    def check_health(self, max_heartbeat_age: int = 60) -> bool:
        """Check agent health"""
        issues = []
        warnings = []

        # Check heartbeat
        if not self.last_heartbeat:
            issues.append("No heartbeat received")
            self.can_connect = False
        else:
            age = (datetime.now(UTC) - self.last_heartbeat).total_seconds()
            if age > max_heartbeat_age:
                issues.append(f"Heartbeat too old: {age:.1f}s")
                self.can_connect = False
            else:
                self.can_connect = True

        # Check status
        if self.status == AgentStatus.ERROR:
            issues.append("Agent in error state")
            self.can_process_tasks = False
        elif self.status == AgentStatus.OFFLINE:
            issues.append("Agent offline")
            self.can_process_tasks = False
        elif self.status == AgentStatus.MAINTENANCE:
            warnings.append("Agent in maintenance")
            self.can_process_tasks = False
        else:
            self.can_process_tasks = True

        # Determine overall health
        self.issues = issues
        self.warnings = warnings

        self.is_healthy = (
                self.can_connect and
                self.can_process_tasks and
                self.resources_available and
                self.dependencies_healthy and
                len(issues) == 0
        )

        # Calculate health score
        self.health_score = self._calculate_health_score()

        return self.is_healthy

    def _calculate_health_score(self) -> float:
        """Calculate health score (0-100)"""
        score = 100.0

        # Deductions for issues
        if not self.can_connect:
            score -= 40
        if not self.can_process_tasks:
            score -= 30
        if not self.resources_available:
            score -= 20
        if not self.dependencies_healthy:
            score -= 10

        # Deductions for specific issues
        for issue in self.issues:
            if "error" in issue.lower():
                score -= 5
            elif "timeout" in issue.lower():
                score -= 3
            else:
                score -= 2

        # Deductions for warnings
        for warning in self.warnings:
            score -= 1

        return max(0.0, min(100.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "is_healthy": self.is_healthy,
            "health_score": self.health_score,
            "can_connect": self.can_connect,
            "can_process_tasks": self.can_process_tasks,
            "resources_available": self.resources_available,
            "dependencies_healthy": self.dependencies_healthy,
            "issues": self.issues,
            "warnings": self.warnings,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "checked_at": self.checked_at.isoformat()
        }


@dataclass
class TaskMetrics:
    """Metrics for individual tasks"""

    task_id: str
    agent_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Performance metrics
    processing_time_ms: Optional[float] = None
    queue_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None

    # Resource usage
    memory_used_mb: Optional[float] = None
    cpu_used_percent: Optional[float] = None

    # Quality metrics
    success: Optional[bool] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    # Results
    result_size_bytes: Optional[int] = None
    result_quality_score: Optional[float] = None

    def start_task(self):
        """Mark task as started"""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.now(UTC)

        # Calculate queue time
        if self.started_at and self.created_at:
            self.queue_time_ms = (self.started_at - self.created_at).total_seconds() * 1000

    def complete_task(self, success: bool = True, error_message: Optional[str] = None):
        """Mark task as completed"""
        self.completed_at = datetime.now(UTC)
        self.success = success
        self.error_message = error_message

        if success:
            self.status = TaskStatus.COMPLETED
        else:
            self.status = TaskStatus.FAILED

        # Calculate processing and total times
        if self.started_at and self.completed_at:
            self.processing_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000

        if self.created_at and self.completed_at:
            self.total_time_ms = (self.completed_at - self.created_at).total_seconds() * 1000

    def cancel_task(self):
        """Mark task as canceled"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now(UTC)
        self.success = False
        self.error_message = "Task cancelled"

    def timeout_task(self):
        """Mark task as timed out"""
        self.status = TaskStatus.TIMEOUT
        self.completed_at = datetime.now(UTC)
        self.success = False
        self.error_message = "Task timed out"

    def retry_task(self):
        """Retry the task"""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.processing_time_ms = None
        self.queue_time_ms = None
        self.total_time_ms = None
        self.success = None
        self.error_message = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_ms": self.processing_time_ms,
            "queue_time_ms": self.queue_time_ms,
            "total_time_ms": self.total_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "cpu_used_percent": self.cpu_used_percent,
            "success": self.success,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "result_size_bytes": self.result_size_bytes,
            "result_quality_score": self.result_quality_score
        }