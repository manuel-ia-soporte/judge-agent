# agents/finance_agent/analyzers/operational_analyzer.py
"""Operational analyzer implementation"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, UTC

from .analyzer_interface import Analyzer
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.repositories.sec_document_repository import SECDocumentRepository
from ....application.commands.analyze_company_command import  AnalyzeCompanyCommand


@dataclass
class OperationalAnalyzer(Analyzer):
    """Operational analyzer implementation"""

    sec_repository: SECDocumentRepository
    operational_service: OperationalAnalysisService

    async def analyze(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        """Perform operational analysis"""

        # Fetch documents
        documents = await self.sec_repository.find_by_cik(
            cik=command.company_cik,
            filing_types=["10-K", "10-Q"],
            start_date=command.start_date,
            end_date=command.end_date
        )

        if not documents:
            return {"error": "No documents found"}

        # Extract operational metrics
        operational_metrics = self.operational_service.extract_operational_metrics(documents)

        # Analyze operational efficiency
        efficiency_analysis = self.operational_service.analyze_operational_efficiency(
            operational_metrics
        )

        # Identify operational risks
        operational_risks = self.operational_service.identify_operational_risks(documents)

        # Analyze supply chain resilience
        supply_chain_analysis = self.operational_service.analyze_supply_chain_resilience(documents)

        # Calculate working capital metrics
        working_capital_metrics = self.operational_service.calculate_working_capital_metrics(
            self._extract_financial_metrics(documents)
        )

        # Generate operational insights
        insights = self._generate_operational_insights(
            efficiency_analysis,
            supply_chain_analysis,
            operational_risks
        )

        return {
            "analysis_type": "operational",
            "company_cik": command.company_cik,
            "operational_metrics": self._operational_metrics_to_dict(operational_metrics),
            "efficiency_analysis": efficiency_analysis,
            "operational_risks": [self._risk_factor_to_dict(rf) for rf in operational_risks],
            "supply_chain_analysis": supply_chain_analysis,
            "working_capital_metrics": working_capital_metrics,
            "insights": insights,
            "documents_analyzed": len(documents),
            "analysis_date": datetime.now(UTC).isoformat()
        }

    def can_handle(self, analysis_type: str) -> bool:
        """Check if analyzer can handle operational analysis"""
        return analysis_type in ["operational", "comprehensive"]

    @staticmethod
    def _extract_financial_metrics(documents: List[Any]) -> List[Any]:
        """Extract financial metrics from documents"""
        # This is a simplified version. In reality, you would use a financial service.

        metrics = []
        for doc in documents:
            # This would actually parse financial statements
            # For now, return empty list
            pass
        return metrics

    @staticmethod
    def _generate_operational_insights(
            efficiency_analysis: Dict[str, Any],
            supply_chain_analysis: Dict[str, Any],
            operational_risks: List[Any]
    ) -> List[str]:
        """Generate operational insights"""
        insights = []

        # Efficiency insights
        efficiency_score = efficiency_analysis.get("efficiency_score", 0)
        if efficiency_score >= 0.7:
            insights.append("Strong operational efficiency indicated by high score")
        elif efficiency_score <= 0.4:
            insights.append("Opportunities for operational efficiency improvements")

        # Supply chain insights
        resilience_score = supply_chain_analysis.get("resilience_score", 0)
        if resilience_score >= 0.7:
            insights.append("Robust supply chain resilience indicated")
        elif resilience_score <= 0.4:
            insights.append("Supply chain vulnerabilities identified")

        # Risk insights
        if operational_risks:
            insights.append(f"Identified {len(operational_risks)} operational risks")

        # Strengths and weaknesses
        strengths = efficiency_analysis.get("strengths", [])
        if strengths:
            insights.append(f"Operational strengths: {', '.join(strengths[:2])}")

        weaknesses = efficiency_analysis.get("weaknesses", [])
        if weaknesses:
            insights.append(f"Operational weaknesses: {', '.join(weaknesses[:2])}")

        return insights

    @staticmethod
    def _operational_metrics_to_dict(metrics: Any) -> Dict[str, Any]:
        """Convert operational metrics to dictionary"""
        return {
            "inventory_turnover": metrics.inventory_turnover,
            "days_sales_outstanding": metrics.days_sales_outstanding,
            "days_payable_outstanding": metrics.days_payable_outstanding,
            "operating_cycle": metrics.operating_cycle,
            "asset_turnover": metrics.asset_turnover,
            "employee_productivity": metrics.employee_productivity
        }

    @staticmethod
    def _risk_factor_to_dict(risk_factor: Any) -> Dict[str, Any]:
        """Convert the risk factor to dictionary"""
        return {
            "description": risk_factor.description,
            "category": risk_factor.category,
            "severity": risk_factor.severity,
            "probability": risk_factor.probability,
            "impact": risk_factor.impact,
            "mitigation": risk_factor.mitigation,
            "risk_score": risk_factor.risk_score()
        }