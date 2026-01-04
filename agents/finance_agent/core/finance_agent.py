# agents/finance_agent/core/finance_agent.py
from typing import Dict, Any, List
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from application.use_cases.analyze_company_use_case import AnalyzeCompanyUseCase
from application.use_cases.compare_companies_use_case import CompareCompaniesUseCase
from .agent_capabilities import AgentCapabilities
from ..factories.analyzer_factory import AnalyzerFactory
from ..strategies.analysis_strategy import AnalysisStrategy, ComprehensiveStrategy, FinancialStrategy, RiskStrategy


@dataclass
class FinanceAgent:
    """Finance Agent using Clean Architecture"""

    agent_id: str
    agent_name: str = "FinanceAnalysisAgent"
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)

    # Dependencies (Dependency Injection)
    analyze_use_case: AnalyzeCompanyUseCase = None
    compare_use_case: CompareCompaniesUseCase = None

    # Strategy Pattern for analysis
    analysis_strategy: AnalysisStrategy = field(default_factory=ComprehensiveStrategy)

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_active = False

        # Factory for creating analyzers
        self.analyzer_factory = AnalyzerFactory()

    async def analyze_company(
            self,
            company_cik: str,
            analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze a company using 'use cases'"""
        self.logger.info(f"Starting analysis for CIK: {company_cik}")

        try:
            # Set analysis strategy based on type
            self._set_analysis_strategy(analysis_type)

            # Execute the use case
            from application.commands import AnalyzeCompanyCommand
            command = AnalyzeCompanyCommand(
                company_cik=company_cik,
                analysis_type=analysis_type,
                start_date=datetime(2022, 1, 1),
                end_date=datetime.now(UTC)
            )

            result = await self.analyzer_factory.create_analyzer(analysis_type).analyze(command)

            self.logger.info(f"Analysis completed for CIK: {company_cik}")
            return result.to_dict()

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}

    async def compare_companies(
            self,
            company_ciks: List[str],
            comparison_metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple companies"""
        try:
            from application.commands import CompareCompaniesCommand
            command = CompareCompaniesCommand(
                company_ciks=company_ciks,
                metrics=comparison_metrics,
                analysis_type="financial"
            )

            result = await self.compare_use_case.execute(command)
            return result.to_dict()

        except Exception as e:
            self.logger.error(f"Comparison failed: {e}")
            return {"error": str(e)}

    async def get_financial_metrics(
            self,
            company_cik: str,
            metric_names: List[str]
    ) -> Dict[str, Any]:
        """Get specific financial metrics"""
        from application.queries import GetFinancialMetricsQuery
        query = GetFinancialMetricsQuery(
            company_cik=company_cik,
            metric_names=metric_names
        )

        # Use query handler pattern
        return await self._handle_query(query)

    def _set_analysis_strategy(self, analysis_type: str):
        """Set analysis strategy based on type"""
        strategies = {
            "comprehensive": ComprehensiveStrategy,
            "financial": FinancialStrategy,
            "risk": RiskStrategy,
            "operational": OperationalStrategy,
            "strategic": StrategicStrategy,
            "quick": QuickAnalysisStrategy
        }

        strategy_class = strategies.get(analysis_type, ComprehensiveStrategy)
        self.analysis_strategy = strategy_class()

    async def _handle_query(self, query):
        """Handle query using CQRS pattern"""
        # Implementation would use query handlers
        pass

    async def start(self):
        """Start the agent"""
        self.is_active = True
        self.logger.info(f"Finance agent {self.agent_id} started")

    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.logger.info(f"Finance agent {self.agent_id} stopped")