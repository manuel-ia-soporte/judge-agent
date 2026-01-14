"""
Contracts package - Defines interfaces and protocols for the system.
"""
from .evaluation_contracts import RubricCategory, ScoringScale, EvaluationRequest, RubricScore, EvaluationResult, A2AMessage
from .finance_contracts import FilingStatus, FinancialStatementType, SECFilingRequest, FinancialMetricData, CompanyFinancials, MarketDataRequest, RiskAssessment
from .judge_contracts import JudgeCapabilities, JudgeMetrics, JudgeConfiguration
from .benchmark_contracts import (
    TaskBenchmarkCase,
    TaskBenchmarkCaseResult,
    TaskBenchmarkRunConfig,
    TaskBenchmarkSuiteResult,
    TaskRubric,
    RubricKind,
    TrackType,
    JudgeMode,
    LLMCallRecord,
    ToolCallRecord,
    TrajectoryEvent,
    BenchmarkRunSuiteRequest,
    BenchmarkRunSuiteResponse,
    BenchmarkGetTraceRequest,
    BenchmarkGetTraceResponse,
)

__all__ = [
    "RubricCategory",
    "ScoringScale",
    "EvaluationRequest",
    "RubricScore",
    "EvaluationResult",
    "A2AMessage",
    "FilingStatus",
    "FinancialStatementType",
    "SECFilingRequest",
    "FinancialMetricData",
    "CompanyFinancials",
    "MarketDataRequest",
    "RiskAssessment",
    "JudgeCapabilities",
    "JudgeMetrics",
    "JudgeConfiguration",
    # Benchmark contracts
    "TaskBenchmarkCase",
    "TaskBenchmarkCaseResult",
    "TaskBenchmarkRunConfig",
    "TaskBenchmarkSuiteResult",
    "TaskRubric",
    "RubricKind",
    "TrackType",
    "JudgeMode",
    "LLMCallRecord",
    "ToolCallRecord",
    "TrajectoryEvent",
    "BenchmarkRunSuiteRequest",
    "BenchmarkRunSuiteResponse",
    "BenchmarkGetTraceRequest",
    "BenchmarkGetTraceResponse",
]
