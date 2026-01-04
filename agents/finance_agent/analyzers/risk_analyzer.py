# agents/finance_agent/analyzers/risk_analyzer.py
"""Risk analyzer implementation"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, UTC

from .analyzer_interface import Analyzer
from ....domain.services.risk_assessment_service import RiskAssessmentService
from ....domain.repositories.sec_document_repository import SECDocumentRepository
from ....application.commands.analyze_company_command import AnalyzeCompanyCommand


@dataclass
class RiskAnalyzer(Analyzer):
    """Risk analyzer implementation"""

    sec_repository: SECDocumentRepository
    risk_service: RiskAssessmentService

    async def analyze(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        """Perform risk analysis"""

        # Fetch documents
        documents = await self.sec_repository.find_by_cik(
            cik=command.company_cik,
            filing_types=["10-K", "10-Q"],
            start_date=command.start_date,
            end_date=command.end_date
        )

        if not documents:
            return {"error": "No documents found"}

        # Extract risk factors
        risk_factors = self.risk_service.extract_risk_factors(documents)

        # Categorize risks
        risk_categories = self.risk_service.categorize_risks(risk_factors)

        # Assess overall risk
        risk_level, risk_score = self.risk_service.assess_overall_risk(risk_factors)

        # Identify mitigations
        mitigations = self.risk_service.identify_mitigations(risk_factors)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(risk_factors, risk_categories)

        # Generate risk insights
        insights = self._generate_risk_insights(risk_factors, risk_level)

        return {
            "analysis_type": "risk",
            "company_cik": command.company_cik,
            "risk_factors": [self._risk_factor_to_dict(rf) for rf in risk_factors],
            "risk_categories": {
                category: [self._risk_factor_to_dict(rf) for rf in risks]
                for category, risks in risk_categories.items()
                if risks  # Only include non-empty categories
            },
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_metrics": risk_metrics,
            "mitigations": mitigations,
            "insights": insights,
            "documents_analyzed": len(documents),
            "analysis_date": datetime.now(UTC).isoformat()
        }

    def can_handle(self, analysis_type: str) -> bool:
        """Check if analyzer can handle risk analysis"""
        return analysis_type in ["risk", "comprehensive"]

    @staticmethod
    def _calculate_risk_metrics(
            risk_factors: List[Any],
            risk_categories: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Calculate risk metrics"""
        metrics = {
            "total_risks": len(risk_factors),
            "high_severity_count": sum(1 for rf in risk_factors if rf.severity == "high"),
            "medium_severity_count": sum(1 for rf in risk_factors if rf.severity == "medium"),
            "low_severity_count": sum(1 for rf in risk_factors if rf.severity == "low"),
            "category_distribution": {
                category: len(risks) for category, risks in risk_categories.items()
            },
            "average_risk_score": sum(rf.risk_score() for rf in risk_factors) / len(risk_factors) if risk_factors else 0
        }

        # Add risk concentration metrics
        if risk_factors:
            metrics["risk_concentration"] = {
                "top_3_risks_score": sum(
                    sorted([rf.risk_score() for rf in risk_factors], reverse=True)[:3]
                ) / sum(rf.risk_score() for rf in risk_factors)
            }

        return metrics

    @staticmethod
    def _generate_risk_insights(
            risk_factors: List[Any],
            risk_level: str
    ) -> List[str]:
        """Generate risk insights"""
        insights = []

        # Count insights by category
        high_risks = [rf for rf in risk_factors if rf.severity == "high"]
        if high_risks:
            insights.append(f"Found {len(high_risks)} high-severity risks requiring immediate attention")

        # Check for risk categories with high concentration
        category_counts = {}
        for rf in risk_factors:
            category_counts[rf.category] = category_counts.get(rf.category, 0) + 1

        if category_counts:
            dominant_category = max(category_counts, key=category_counts.get)
            insights.append(f"Highest risk concentration in {dominant_category} category")

        # Risk level insight
        insights.append(f"Overall risk level assessed as: {risk_level}")

        # Mitigation insight
        mitigated_risks = [rf for rf in risk_factors if rf.mitigation]
        if mitigated_risks:
            insights.append(f"{len(mitigated_risks)} risks have mitigation strategies mentioned")
        else:
            insights.append("No explicit mitigation strategies mentioned in filings")

        return insights

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