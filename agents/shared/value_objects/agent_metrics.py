# agents/shared/value_objects/agent_metrics.py
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentMetrics:
    processing_time_ms: int
    tokens_used: int
    warnings_count: int = 0
