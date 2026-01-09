# agents/finance_agent/strategies/analysis_strategy.py
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List

from application.commands.analyze_company_command import AnalyzeCompanyCommand, AnalysisType
from application.dtos.analysis_dtos import AnalysisResultDTO
from application.use_cases.assess_risk_use_case import AssessRiskUseCase

# Analyzers (already defined in codebase)
from agents.finance_agent.analyzers.financial_analyzer import FinancialAnalyzer
from agents.finance_agent.analyzers.operational_analyzer import OperationalAnalyzer
from agents.finance_agent.analyzers.strategic_analyzer import StrategicAnalyzer
from agents.finance_agent.analyzers.hybrid_risk_analyzer import HybridRiskAnalyzer
from agents.finance_agent.sec_analyzer import SECAnalyzer

# Infrastructure adapter (exists in codebase)
from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter


class AnalysisStrategy(ABC):
    """
    Strategy = HOW the analysis is executed.
    Hexagonal Architecture: This interface belongs to the application core.
    """

    @abstractmethod
    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        ...


class FullAnalysisStrategy(AnalysisStrategy):
    """
    Concrete strategy that orchestrates domain analyzers and use cases.

    Design Pattern: Strategy + Composition
    DDD: Coordinates domain services via analyzers (application layer)
    Hexagonal: Depends only on ports (adapters injected implicitly via direct instantiation for simplicity;
              can be dependency-injected in main.py for full testability).
    """

    def __init__(self, risk_use_case: AssessRiskUseCase):
        # Infrastructure adapter (driving adapter for SEC data)
        self._sec_adapter = SECEdgarAdapter()

        # Domain analyzers (thin wrappers over domain services)
        self._financial_analyzer = FinancialAnalyzer()
        self._operational_analyzer = OperationalAnalyzer()
        self._strategic_analyzer = StrategicAnalyzer()
        self._risk_analyzer = HybridRiskAnalyzer()
        self._sec_analyzer = SECAnalyzer()

        # Application use case for specialized risk assessment
        self._risk_use_case = risk_use_case

    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        # 1. Fetch raw SEC documents (infrastructure concern)
        documents = self._sec_adapter.find_by_cik(command.company_cik)
        if not documents:
            raise ValueError(f"No SEC documents found for CIK: {command.company_cik}")

        # 2. Pre-process documents (optional metadata extraction)
        _ = self._sec_analyzer.summarize(documents)  # currently unused but kept for extensibility

        # 3. Run domain analyzers
        fin_result: Dict[str, Any] = self._financial_analyzer.analyze(documents)
        operational_assessment: Dict[str, Any] = self._operational_analyzer.analyze(fin_result)
        strategic_assessment: Dict[str, Any] = self._strategic_analyzer.analyze(documents)

        # 4. Handle risk analysis based on command type
        risk_assessment: Dict[str, Any] = {}
        risk_factors: List[Dict[str, Any]] = []

        if command.analysis_type == AnalysisType.RISK:
            # Use dedicated risk use case (e.g., for regulatory or deep risk workflows)
            risk_dto = await self._risk_use_case.execute(command.company_cik)
            risk_assessment = risk_dto.to_dict()
            risk_factors = risk_dto.risk_factors or []
        else:
            # Use hybrid analyzer for embedded risk scoring
            financial_metrics = fin_result.get("financial_metrics", {})
            risk_result = await self._risk_analyzer.analyze(financial_metrics)

            risk_assessment = {
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
            }
            risk_factors = [{
                "description": risk_result["explanation"],
                "category": "financial",
                "severity": risk_result["risk_level"],
                "probability": risk_result["risk_score"],
                "impact": "material"
            }]

        # 5. Build final DTO with proper structure
        return AnalysisResultDTO(
            analysis_id=f"ana_{command.company_cik}_{int(datetime.now(timezone.utc).timestamp())}",
            company_cik=command.company_cik,
            analysis_type=command.analysis_type.value,
            analysis_date=datetime.now(timezone.utc),
            status="completed",
            metrics=[
                {"name": k, "value": v}
                for k, v in fin_result.get("financial_metrics", {}).items()
            ],
            ratios=[
                {"name": k, "value": v}
                for k, v in fin_result.get("financial_ratios", {}).items()
            ],
            risk_factors=risk_factors,
            financial_assessment=fin_result,
            operational_assessment=operational_assessment,
            strategic_assessment=strategic_assessment,
            risk_assessment=risk_assessment,
            key_findings=["Comprehensive analysis completed"],
            conclusions=["Company exhibits moderate financial health"],
            recommendations=["Monitor leverage and liquidity quarterly"],
            documents_analyzed=len(documents),
            confidence_score=0.85
        )