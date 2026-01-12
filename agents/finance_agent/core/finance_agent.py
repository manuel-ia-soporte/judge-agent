# agents/finance_agent/core/finance_agent.py
from typing import Dict, Any, List

from agents.finance_agent.factories.analyzer_factory import AnalyzerFactory
from agents.finance_agent.strategies.analysis_strategy import AnalysisStrategy
from agents.finance_agent.strategies.comparison_strategy import ComparisonStrategy
from agents.finance_agent.core.agent_capabilities import (
    CapabilityRegistry,
    AgentCapability,
    CapabilitySchema,
)
from application.use_cases.analyze_company_use_case import AnalyzeCompanyCommand
from application.dtos.analysis_dtos import AnalysisResultDTO


class FinanceAgent:
    """
    Finance agent with dynamic, self-describing capabilities.
    """

    def __init__(self, strategy: AnalysisStrategy) -> None:
        self._strategy = strategy
        self._risk_analyzer = AnalyzerFactory.create_risk()
        self._comparison_strategy = ComparisonStrategy()

        self._capabilities = CapabilityRegistry()
        self._register_capabilities()

    async def quick_analyze(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        """
        Quick analysis variant (e.g., uses fewer documents, skips deep risk).
        Reuses main analysis strategy with adjusted command.
        """
        # Optionally override command fields for "quick" mode
        command.depth_level = "executive"
        command.include_trends = False
        command.include_comparisons = False
        return await self.analyze(command)

    def list_capabilities(self) -> Dict[str, AgentCapability]:
        return self._capabilities.list()

    def get_capability(self, name: str) -> AgentCapability:
        return self._capabilities.get(name)

    async def analyze(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        return await self._strategy.execute(command)

    def compare(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._comparison_strategy.compare(analyses)

    def _register_capabilities(self) -> None:
        self._capabilities.register(
            AgentCapability(
                name="analyze_company",
                schema=CapabilitySchema(
                    input_type=AnalyzeCompanyCommand,
                    output_type=dict,
                    description="Analyze a company and assess risk",
                ),
                handler=self.analyze,
                required_permission="finance:analyze",
            )
        )

        self._capabilities.register(
            AgentCapability(
                name="compare_analyses",
                schema=CapabilitySchema(
                    input_type=list,
                    output_type=dict,
                    description="Compare multiple analysis results",
                ),
                handler=self.compare,
                required_permission="finance:compare",
            )
        )

        self._capabilities.register(
            AgentCapability(
                name="quick_analyze",
                schema=CapabilitySchema(
                    input_type=AnalyzeCompanyCommand,
                    output_type=dict,
                    description="Quick company analysis",
                ),
                handler=self.quick_analyze,
                required_permission="finance:analyze",
            )
        )

