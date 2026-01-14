"""Benchmark contracts for Finance Agent Benchmark v2.

Defines schemas for Task-Accuracy and Analysis-Quality benchmark tracks.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TrackType(str, Enum):
    """Benchmark track type."""
    TASK = "task"
    QUALITY = "quality"
    BOTH = "both"


class JudgeMode(str, Enum):
    """Correctness judge mode."""
    DETERMINISTIC = "deterministic"
    LLM = "llm"


class RubricKind(str, Enum):
    """Kind of correctness rubric check."""
    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    LLM = "llm"


class TaskRubric(BaseModel):
    """A single rubric/criterion for evaluating task correctness."""
    rubric_id: str = Field(..., description="Unique rubric identifier")
    description: str = Field("", description="Human-readable description")
    kind: RubricKind = Field(RubricKind.CONTAINS, description="Type of check")
    expected_text: Optional[str] = Field(None, description="Expected text for exact/contains/regex")
    pattern: Optional[str] = Field(None, description="Regex pattern (for kind=regex)")
    expected_number: Optional[float] = Field(None, description="Expected number (for numeric_tolerance)")
    tolerance: Optional[float] = Field(None, description="Numeric tolerance (for numeric_tolerance)")
    llm_prompt: Optional[str] = Field(None, description="LLM judge prompt (for kind=llm)")
    weight: float = Field(1.0, ge=0, description="Weight for aggregation")


class TaskBenchmarkCase(BaseModel):
    """A single benchmark case for the Task-Accuracy track."""
    case_id: str = Field(..., description="Unique case identifier")
    category: str = Field("general", description="Category for class-balanced accuracy")
    question: str = Field(..., description="The question to answer")
    ground_truth: Optional[str] = Field(None, description="Ground truth answer (for simple checks)")
    rubrics: Optional[List[TaskRubric]] = Field(
        default_factory=list,
        description="List of rubrics for correctness evaluation"
    )
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context (e.g., target company CIK, filing period)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arbitrary metadata"
    )


class LLMCallRecord(BaseModel):
    """Record of an LLM call."""
    provider: str = Field(..., description="LLM provider (e.g., 'openai', 'anthropic')")
    model: str = Field(..., description="Model name")
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    latency_ms: int = Field(0, ge=0)
    cost_usd: float = Field(0.0, ge=0)


class ToolCallRecord(BaseModel):
    """Record of a tool call within the harness."""
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    ok: bool = Field(True)
    elapsed_ms: int = Field(0, ge=0)
    output_preview: Optional[str] = Field(None, description="Truncated output for logging")
    error: Optional[str] = Field(None)


class TrajectoryEvent(BaseModel):
    """A single event in the agent's trajectory."""
    ts: datetime = Field(default_factory=datetime.utcnow)
    event_type: Literal["llm", "tool", "error", "final"] = Field(..., description="Event type")
    message: str = Field("", description="Human-readable message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")


class TaskBenchmarkCaseResult(BaseModel):
    """Result of running a single benchmark case."""
    case_id: str
    ok: bool = Field(..., description="Whether the case ran without errors")
    elapsed_ms: int = Field(0, ge=0)
    answer: Optional[str] = Field(None, description="Agent's final answer")
    sources: Optional[List[str]] = Field(default_factory=list)
    correctness: Optional[bool] = Field(None, description="True if all rubrics passed")
    correctness_details: Optional[Dict[str, Any]] = Field(None)
    quality_result: Optional[Dict[str, Any]] = Field(None, description="JudgeAgent evaluation result")
    llm: Optional[LLMCallRecord] = Field(None)
    tool_calls: Optional[List[ToolCallRecord]] = Field(default_factory=list)
    trace_path: Optional[str] = Field(None, description="Path to persisted trace file")
    error: Optional[str] = Field(None)


class TaskBenchmarkRunConfig(BaseModel):
    """Configuration for a benchmark run."""
    track: Literal["task", "quality", "both"] = Field("task")
    provider: str = Field("auto", description="LLM provider for harness")
    model: Optional[str] = Field(None, description="LLM model for harness")
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(2048, ge=1)
    max_steps: int = Field(12, ge=1, description="Max ReAct iterations")
    timeout_s: float = Field(60.0, gt=0)
    live: bool = Field(True, description="Enable live EDGAR/Web calls")
    cache_dir: str = Field(".cache/benchmark")
    judge_mode: Literal["deterministic", "llm"] = Field("deterministic")
    subcall_provider: Optional[str] = Field(None, description="Provider for retrieve.information subcalls")
    subcall_model: Optional[str] = Field(None, description="Model for retrieve.information subcalls")
    truncate_observations: int = Field(2000, ge=0, description="Truncate observations in traces")


class TaskBenchmarkSuiteResult(BaseModel):
    """Result of running a full benchmark suite."""
    run_id: str
    started_at: datetime
    config: TaskBenchmarkRunConfig
    results: List[TaskBenchmarkCaseResult]
    summary: Dict[str, Any] = Field(default_factory=dict)


# ----- A2A / MCP capabilities -----

class BenchmarkRunSuiteRequest(BaseModel):
    """Request to run a benchmark suite via A2A/MCP."""
    dataset: str = Field("demo", description="'demo' or path to dataset, or inline list")
    config: Optional[TaskBenchmarkRunConfig] = Field(None)
    limit: Optional[int] = Field(None, ge=1)


class BenchmarkRunSuiteResponse(BaseModel):
    """Response from running a benchmark suite."""
    run_id: str
    status: Literal["completed", "failed", "partial"]
    summary: Dict[str, Any]
    results_path: Optional[str] = Field(None)
    error: Optional[str] = Field(None)


class BenchmarkGetTraceRequest(BaseModel):
    """Request to retrieve a trace."""
    run_id: str
    case_id: str


class BenchmarkGetTraceResponse(BaseModel):
    """Response with trace data."""
    run_id: str
    case_id: str
    trace: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)
