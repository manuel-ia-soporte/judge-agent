# application/use_cases/analyze_company_use_case.py
from typing import List
from dataclasses import dataclass

from domain.models.value_objects import FinancialRatio, RiskFactor
from ..commands import AnalyzeCompanyCommand
from ..dtos.analysis_dtos import AnalysisResultDTO
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.risk_assessment_service import RiskAssessmentService
from domain.repositories.sec_document_repository import SECDocumentRepository


@dataclass
class AnalyzeCompanyUseCase:
    """Use Case for analyzing a company"""

    sec_repository: SECDocumentRepository
    financial_service: FinancialAnalysisService
    risk_service: RiskAssessmentService

    async def execute(self, command: AnalyzeCompanyCommand) -> AnalysisResultDTO:
        """Execute company analysis use case"""

        # 1. Fetch documents (Hexagonal: Application -> Infrastructure)
        documents = await self.sec_repository.find_by_cik(
            cik=command.company_cik,
            filing_types=["10-K", "10-Q"],
            start_date=command.start_date,
            end_date=command.end_date
        )

        if not documents:
            raise ValueError(f"No filings found for CIK {command.company_cik}")

        # 2. Analyze based on type (Strategy Pattern)
        if command.analysis_type == "comprehensive":
            result = await self._comprehensive_analysis(documents, command)
        elif command.analysis_type == "financial":
            result = await self._financial_analysis(documents, command)
        elif command.analysis_type == "risk":
            result = await self._risk_analysis(documents, command)
        else:
            result = await self._quick_analysis(documents, command)

        return result

    async def _comprehensive_analysis(
            self,
            documents: List[SECDocument],
            command: AnalyzeCompanyCommand
    ) -> AnalysisResultDTO:
        """Perform comprehensive analysis"""

        # Extract metrics
        metrics = self.financial_service.extract_metrics(documents)

        # Calculate ratios
        ratios = self.financial_service.calculate_ratios(metrics)

        # Analyze risks
        risk_factors = self.risk_service.extract_risk_factors(documents)
        risk_categories = self.risk_service.categorize_risks(risk_factors)
        risk_level, risk_score = self.risk_service.assess_overall_risk(risk_factors)

        # Analyze trends
        trends = []
        for metric_name in ["Revenue", "NetIncomeLoss", "Assets"]:
            trend = self.financial_service.analyze_trends(metrics, metric_name)
            if trend:
                trends.append(trend)

        # Generate conclusions
        conclusions = self._generate_conclusions(metrics, ratios, risk_factors)

        return AnalysisResultDTO(
            company_cik=command.company_cik,
            analysis_type=command.analysis_type,
            metrics=[m.__dict__ for m in metrics],
            ratios=[r.__dict__ for r in ratios],
            risk_factors=[rf.__dict__ for rf in risk_factors],
            risk_categories=risk_categories,
            risk_level=risk_level,
            risk_score=risk_score,
            trends=[t.__dict__ for t in trends],
            conclusions=conclusions,
            documents_analyzed=len(documents)
        )

    async def _financial_analysis(
            self,
            documents: List[SECDocument],
            command: AnalyzeCompanyCommand
    ) -> AnalysisResultDTO:
        """Focus on financial analysis"""
        metrics = self.financial_service.extract_metrics(documents)
        ratios = self.financial_service.calculate_ratios(metrics)

        return AnalysisResultDTO(
            company_cik=command.company_cik,
            analysis_type=command.analysis_type,
            metrics=[m.__dict__ for m in metrics],
            ratios=[r.__dict__ for r in ratios],
            documents_analyzed=len(documents)
        )

    async def _risk_analysis(
            self,
            documents: List[SECDocument],
            command: AnalyzeCompanyCommand
    ) -> AnalysisResultDTO:
        """Focus on risk analysis"""
        risk_factors = self.risk_service.extract_risk_factors(documents)
        risk_categories = self.risk_service.categorize_risks(risk_factors)
        risk_level, risk_score = self.risk_service.assess_overall_risk(risk_factors)
        mitigations = self.risk_service.identify_mitigations(risk_factors)

        return AnalysisResultDTO(
            company_cik=command.company_cik,
            analysis_type=command.analysis_type,
            risk_factors=[rf.__dict__ for rf in risk_factors],
            risk_categories=risk_categories,
            risk_level=risk_level,
            risk_score=risk_score,
            mitigations=mitigations,
            documents_analyzed=len(documents)
        )

    async def _quick_analysis(
            self,
            documents: List[SECDocument],
            command: AnalyzeCompanyCommand
    ) -> AnalysisResultDTO:
        """Quick overview analysis"""
        latest_doc = max(documents, key=lambda d: d.filing_date)

        metrics = self.financial_service.extract_metrics([latest_doc])
        risk_factors = self.risk_service.extract_risk_factors([latest_doc])

        return AnalysisResultDTO(
            company_cik=command.company_cik,
            analysis_type=command.analysis_type,
            metrics=[m.__dict__ for m in metrics][:5],  # Top 5
            risk_factors=[rf.__dict__ for rf in risk_factors][:3],  # Top 3
            documents_analyzed=1
        )

    @staticmethod
    def _generate_conclusions(
            metrics: List[FinancialMetric],
            ratios: List[FinancialRatio],
            risk_factors: List[RiskFactor]
    ) -> List[str]:
        """Generate conclusions from analysis"""
        conclusions = []

        # Financial conclusions
        current_ratio = next((r for r in ratios if r.name == "current_ratio"), None)
        if current_ratio:
            if current_ratio.value < 1:
                conclusions.append("Potential liquidity concern: current ratio below 1")
            elif current_ratio.value > 2:
                conclusions.append("Strong liquidity position: current ratio above 2")

        debt_to_equity = next((r for r in ratios if r.name == "debt_to_equity"), None)
        if debt_to_equity and debt_to_equity.value > 2:
            conclusions.append("High leverage: debt-to-equity ratio above 2")

        # Risk conclusions
        if risk_factors:
            high_risks = sum(1 for r in risk_factors if r.severity == "high")
            if high_risks > 5:
                conclusions.append(f"Multiple high-severity risks identified: {high_risks}")

        return conclusions