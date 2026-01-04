# agents/finance_agent/core/agent_capabilities.py
"""Agent capabilities definitions"""

from dataclasses import dataclass, field
from typing import Dict, Any, Set
from enum import Enum


class AnalysisCapability(str, Enum):
    """Analysis capabilities that agents can have"""
    FINANCIAL_STATEMENT_ANALYSIS = "financial_statement_analysis"
    RATIO_CALCULATION = "ratio_calculation"
    TREND_ANALYSIS = "trend_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    OPERATIONAL_ANALYSIS = "operational_analysis"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    VALUATION = "valuation"
    FORECASTING = "forecasting"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    BENCHMARKING = "benchmarking"
    DATA_VALIDATION = "data_validation"
    QUALITY_ASSURANCE = "quality_assurance"


class DataSourceCapability(str, Enum):
    """Data source capabilities"""
    SEC_EDGAR_API = "sec_edgar_api"
    YAHOO_FINANCE_API = "yahoo_finance_api"
    MARKET_DATA_API = "market_data_api"
    INTERNAL_DATABASE = "internal_database"
    REAL_TIME_DATA = "real_time_data"
    HISTORICAL_DATA = "historical_data"


class ProcessingCapability(str, Enum):
    """Processing capabilities"""
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_PROCESSING = "real_time_processing"
    STREAM_PROCESSING = "stream_processing"
    DISTRIBUTED_PROCESSING = "distributed_processing"
    PARALLEL_PROCESSING = "parallel_processing"


@dataclass
class AgentCapabilities:
    """Defines capabilities of an agent"""

    # Core capabilities
    analysis_capabilities: Set[AnalysisCapability] = field(default_factory=set)
    data_source_capabilities: Set[DataSourceCapability] = field(default_factory=set)
    processing_capabilities: Set[ProcessingCapability] = field(default_factory=set)

    # Performance characteristics
    max_concurrent_analyses: int = 5
    analysis_timeout_seconds: int = 300
    max_documents_per_analysis: int = 10
    supports_async: bool = True
    supports_batch: bool = False

    # Quality characteristics
    accuracy_score: float = 0.0  # 0-1 scale
    reliability_score: float = 0.0  # 0-1 scale
    speed_score: float = 0.0  # 0-1 scale

    # Configuration
    enabled: bool = True
    priority: int = 1  # 1-10, higher is more important
    cost_per_analysis: float = 0.0

    def can_perform(self, capability: AnalysisCapability) -> bool:
        """Check if agent has specific capability"""
        return capability in self.analysis_capabilities

    def can_access(self, data_source: DataSourceCapability) -> bool:
        """Check if agent can access the data source"""
        return data_source in self.data_source_capabilities

    def get_performance_score(self) -> float:
        """Calculate overall performance score"""
        weights = {
            "accuracy": 0.4,
            "reliability": 0.3,
            "speed": 0.3
        }

        return (
                self.accuracy_score * weights["accuracy"] +
                self.reliability_score * weights["reliability"] +
                self.speed_score * weights["speed"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "analysis_capabilities": [c.value for c in self.analysis_capabilities],
            "data_source_capabilities": [c.value for c in self.data_source_capabilities],
            "processing_capabilities": [c.value for c in self.processing_capabilities],
            "max_concurrent_analyses": self.max_concurrent_analyses,
            "analysis_timeout_seconds": self.analysis_timeout_seconds,
            "max_documents_per_analysis": self.max_documents_per_analysis,
            "supports_async": self.supports_async,
            "supports_batch": self.supports_batch,
            "accuracy_score": self.accuracy_score,
            "reliability_score": self.reliability_score,
            "speed_score": self.speed_score,
            "performance_score": self.get_performance_score(),
            "enabled": self.enabled,
            "priority": self.priority,
            "cost_per_analysis": self.cost_per_analysis
        }

    @classmethod
    def create_finance_agent_capabilities(cls) -> "AgentCapabilities":
        """Create capabilities for a finance agent"""
        return cls(
            analysis_capabilities={
                AnalysisCapability.FINANCIAL_STATEMENT_ANALYSIS,
                AnalysisCapability.RATIO_CALCULATION,
                AnalysisCapability.TREND_ANALYSIS,
                AnalysisCapability.RISK_ASSESSMENT,
                AnalysisCapability.OPERATIONAL_ANALYSIS,
                AnalysisCapability.STRATEGIC_ANALYSIS,
                AnalysisCapability.COMPARATIVE_ANALYSIS,
                AnalysisCapability.BENCHMARKING,
                AnalysisCapability.DATA_VALIDATION
            },
            data_source_capabilities={
                DataSourceCapability.SEC_EDGAR_API,
                DataSourceCapability.YAHOO_FINANCE_API,
                DataSourceCapability.MARKET_DATA_API
            },
            processing_capabilities={
                ProcessingCapability.BATCH_PROCESSING,
                ProcessingCapability.PARALLEL_PROCESSING
            },
            max_concurrent_analyses=3,
            analysis_timeout_seconds=600,
            max_documents_per_analysis=20,
            supports_async=True,
            supports_batch=True,
            accuracy_score=0.85,
            reliability_score=0.90,
            speed_score=0.75,
            enabled=True,
            priority=5,
            cost_per_analysis=0.0
        )

    @classmethod
    def create_risk_agent_capabilities(cls) -> "AgentCapabilities":
        """Create capabilities for a risk agent"""
        return cls(
            analysis_capabilities={
                AnalysisCapability.RISK_ASSESSMENT,
                AnalysisCapability.DATA_VALIDATION,
                AnalysisCapability.QUALITY_ASSURANCE
            },
            data_source_capabilities={
                DataSourceCapability.SEC_EDGAR_API,
                DataSourceCapability.INTERNAL_DATABASE
            },
            processing_capabilities={
                ProcessingCapability.REAL_TIME_PROCESSING,
                ProcessingCapability.STREAM_PROCESSING
            },
            max_concurrent_analyses=10,
            analysis_timeout_seconds=120,
            max_documents_per_analysis=5,
            supports_async=True,
            supports_batch=False,
            accuracy_score=0.95,
            reliability_score=0.98,
            speed_score=0.90,
            enabled=True,
            priority=7,
            cost_per_analysis=0.0
        )

    @classmethod
    def create_valuation_agent_capabilities(cls) -> "AgentCapabilities":
        """Create capabilities for a valuation agent"""
        return cls(
            analysis_capabilities={
                AnalysisCapability.VALUATION,
                AnalysisCapability.FORECASTING,
                AnalysisCapability.FINANCIAL_STATEMENT_ANALYSIS
            },
            data_source_capabilities={
                DataSourceCapability.SEC_EDGAR_API,
                DataSourceCapability.MARKET_DATA_API,
                DataSourceCapability.REAL_TIME_DATA
            },
            processing_capabilities={
                ProcessingCapability.BATCH_PROCESSING
            },
            max_concurrent_analyses=2,
            analysis_timeout_seconds=900,
            max_documents_per_analysis=15,
            supports_async=True,
            supports_batch=False,
            accuracy_score=0.80,
            reliability_score=0.85,
            speed_score=0.60,
            enabled=True,
            priority=3,
            cost_per_analysis=0.0
        )