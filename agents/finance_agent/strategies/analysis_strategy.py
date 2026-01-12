# agents/finance_agent/strategies/analysis_strategy.py
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from application.commands.analyze_company_command import AnalyzeCompanyCommand, AnalysisType
from application.dtos.analysis_dtos import AnalysisResultDTO, AnalysisStatus
from application.use_cases.assess_risk_use_case import AssessRiskUseCase

# Domain services
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService

# Analyzers (now properly initialized with services)
from agents.finance_agent.analyzers.financial_analyzer import FinancialAnalyzer
from agents.finance_agent.analyzers.operational_analyzer import OperationalAnalyzer
from agents.finance_agent.analyzers.strategic_analyzer import StrategicAnalyzer
from agents.finance_agent.analyzers.hybrid_risk_analyzer import HybridRiskAnalyzer

# Infrastructure
from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter

class AnalysisStrategy(ABC):
    @abstractmethod
    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        ...

class FullAnalysisStrategy(AnalysisStrategy):
    def __init__(
        self,
        risk_use_case: AssessRiskUseCase,
        financial_service: FinancialAnalysisService,
        operational_service: OperationalAnalysisService,
        strategic_service: StrategicAnalysisService,
    ):
        self._sec_adapter = SECEdgarAdapter()
        self._financial_analyzer = FinancialAnalyzer(financial_service)
        self._operational_analyzer = OperationalAnalyzer(operational_service)
        self._strategic_analyzer = StrategicAnalyzer(strategic_service)
        self._risk_analyzer = HybridRiskAnalyzer()
        self._risk_use_case = risk_use_case

    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        documents = self._sec_adapter.find_by_cik(command.company_cik)
        if not documents:
            raise ValueError(f"No SEC documents found for CIK: {command.company_cik}")

        fin_result = self._financial_analyzer.analyze(documents)
        operational_assessment = self._operational_analyzer.analyze(documents)
        strategic_assessment = self._strategic_analyzer.analyze(documents)

        risk_assessment = {}
        risk_factors = []

        if command.analysis_type == AnalysisType.RISK:
            risk_dto = await self._risk_use_case.execute(command.company_cik)
            risk_assessment = risk_dto.to_dict()
            risk_factors = risk_dto.risk_factors or []
        else:
            financial_metrics = fin_result.get("financial_metrics", {})
            risk_result = await self._risk_analyzer.analyze(financial_metrics)
            risk_assessment = {
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
            }
            risk_factors = [{
                "description": risk_result.get("explanation", "No explanation"),
                "category": "financial",
                "severity": risk_result["risk_level"],
                "probability": risk_result["risk_score"],
                "impact": "material"
            }]

        now = datetime.now(timezone.utc)
        return AnalysisResultDTO(
            analysis_id=f"ana_{command.company_cik}_{int(now.timestamp())}",
            company_cik=command.company_cik,
            analysis_type=command.analysis_type,
            analysis_date=now,
            status=AnalysisStatus.COMPLETED,
            metrics=[{"name": k, "value": v} for k, v in fin_result.get("financial_metrics", {}).items()],
            ratios=[{"name": k, "value": v} for k, v in fin_result.get("financial_ratios", {}).items()],
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