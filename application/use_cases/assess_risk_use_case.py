# application/use_cases/assess_risk_use_case.py
"""Use the case for risk assessment"""

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, UTC

from ..commands import AssessRiskCommand
from ..dtos.risk_dtos import RiskAssessmentDTO
from domain.repositories.sec_document_repository import SECDocumentRepository
from domain.services.risk_assessment_service import RiskAssessmentService
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.operational_analysis_service import OperationalAnalysisService


@dataclass
class AssessRiskUseCase:
    """Use the case for comprehensive risk assessment"""

    sec_repository: SECDocumentRepository
    risk_service: RiskAssessmentService
    financial_service: FinancialAnalysisService
    operational_service: OperationalAnalysisService

    async def execute(self, command: AssessRiskCommand) -> RiskAssessmentDTO:
        """Execute the risk assessment use case"""

        # Fetch documents
        documents = await self.sec_repository.find_by_cik(
            cik=command.company_cik,
            filing_types=command.filing_types or ["10-K", "10-Q"],
            start_date=command.start_date,
            end_date=command.end_date
        )

        if not documents:
            raise ValueError(f"No filings found for CIK {command.company_cik}")

        # Extract risk factors
        risk_factors = self.risk_service.extract_risk_factors(documents)

        # Categorize risks
        risk_categories = self.risk_service.categorize_risks(risk_factors)

        # Assess overall risk
        risk_level, risk_score = self.risk_service.assess_overall_risk(risk_factors)

        # Identify mitigations
        mitigations = self.risk_service.identify_mitigations(risk_factors)

        # Financial risk assessment
        financial_risk = await self._assess_financial_risk(documents)

        # Operational risk assessment
        operational_risk = await self._assess_operational_risk(documents)

        # Calculate composite risk score
        composite_score = self._calculate_composite_risk_score(
            risk_score,
            financial_risk.get("score", 0),
            operational_risk.get("score", 0)
        )

        # Identify key risk indicators
        kris = self._identify_key_risk_indicators(
            risk_factors, financial_risk, operational_risk
        )

        # Generate risk mitigation recommendations
        recommendations = self._generate_risk_recommendations(
            risk_factors, financial_risk, operational_risk
        )

        return RiskAssessmentDTO(
            company_cik=command.company_cik,
            assessment_date=datetime.now(UTC),
            risk_factors=[self._risk_factor_to_dict(rf) for rf in risk_factors],
            risk_categories=risk_categories,
            risk_level=risk_level,
            risk_score=risk_score,
            composite_risk_score=composite_score,
            financial_risk_assessment=financial_risk,
            operational_risk_assessment=operational_risk,
            mitigations=mitigations,
            key_risk_indicators=kris,
            recommendations=recommendations,
            monitoring_indicators=self._generate_monitoring_indicators(risk_factors),
            next_review_date=self._calculate_next_review_date(risk_level),
            documents_analyzed=len(documents)
        )

    async def _assess_financial_risk(
            self,
            documents: List[Any]
    ) -> Dict[str, Any]:
        """Assess financial risk"""
        financial_risk = {
            "score": 0.0,
            "categories": {},
            "indicators": [],
            "concerns": []
        }

        # Extract financial metrics
        metrics = self.financial_service.extract_metrics(documents)

        # Calculate ratios
        ratios = self.financial_service.calculate_ratios(metrics)

        # Assess liquidity risk
        current_ratio = next((r for r in ratios if r.name == "current_ratio"), None)
        if current_ratio:
            if current_ratio.value < 1.0:
                financial_risk["score"] += 0.3
                financial_risk["concerns"].append("Liquidity risk: current ratio below 1")
                financial_risk["indicators"].append(f"Current ratio: {current_ratio.value:.2f}")
            financial_risk["categories"]["liquidity"] = current_ratio.value

        # Assess solvency risk
        debt_to_equity = next((r for r in ratios if r.name == "debt_to_equity"), None)
        if debt_to_equity:
            if debt_to_equity.value > 2.0:
                financial_risk["score"] += 0.4
                financial_risk["concerns"].append("High leverage: debt-to-equity above 2")
                financial_risk["indicators"].append(f"Debt-to-equity: {debt_to_equity.value:.2f}")
            financial_risk["categories"]["solvency"] = debt_to_equity.value

        # Assess profitability risk
        profit_margin = next((r for r in ratios if r.name == "profit_margin"), None)
        if profit_margin:
            if profit_margin.value < 0.05:
                financial_risk["score"] += 0.3
                financial_risk["concerns"].append("Low profitability: margin below 5%")
                financial_risk["indicators"].append(f"Profit margin: {profit_margin.value:.2%}")
            financial_risk["categories"]["profitability"] = profit_margin.value

        # Normalize score
        financial_risk["score"] = min(1.0, financial_risk["score"])

        return financial_risk

    async def _assess_operational_risk(
            self,
            documents: List[Any]
    ) -> Dict[str, Any]:
        """Assess operational risk"""
        operational_risk = {
            "score": 0.0,
            "categories": {},
            "indicators": [],
            "concerns": []
        }

        # Extract operational metrics
        operational_metrics = self.operational_service.extract_operational_metrics(documents)

        # Analyze supply chain resilience
        supply_chain = self.operational_service.analyze_supply_chain_resilience(documents)
        resilience_score = supply_chain.get("resilience_score", 0)

        if resilience_score < 0.5:
            operational_risk["score"] += 0.3
            operational_risk["concerns"].append("Supply chain resilience concerns")
            operational_risk["indicators"].append(f"Supply chain resilience: {resilience_score:.2f}")

        operational_risk["categories"]["supply_chain"] = resilience_score

        # Identify operational risks
        operational_risks = self.operational_service.identify_operational_risks(documents)
        if operational_risks:
            operational_risk["score"] += min(0.4, len(operational_risks) * 0.1)
            operational_risk["concerns"].append(
                f"{len(operational_risks)} operational risks identified"
            )

        # Normalize score
        operational_risk["score"] = min(1.0, operational_risk["score"])

        return operational_risk

    @staticmethod
    def _calculate_composite_risk_score(
            risk_score: float,
            financial_risk: float,
            operational_risk: float
    ) -> float:
        """Calculate composite risk score"""
        # Weighted average
        return (
                risk_score * 0.5 +
                financial_risk * 0.3 +
                operational_risk * 0.2
        )

    @staticmethod
    def _identify_key_risk_indicators(
            risk_factors: List[Any],
            financial_risk: Dict[str, Any],
            operational_risk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify key risk indicators"""
        kris = []

        # High severity risk factors
        high_severity = [rf for rf in risk_factors if rf.severity == "high"]
        for risk in high_severity[:3]:  # Top 3 high the severity
            kris.append({
                "type": "risk_factor",
                "description": risk.description[:100],
                "severity": risk.severity,
                "category": risk.category
            })

        # Financial indicators
        financial_concerns = financial_risk.get("concerns", [])
        for concern in financial_concerns[:2]:  # Top 2 concerns
            kris.append({
                "type": "financial",
                "description": concern,
                "severity": "high" if "below" in concern.lower() else "medium"
            })

        # Operational indicators
        operational_concerns = operational_risk.get("concerns", [])
        for concern in operational_concerns[:2]:  # Top 2 concerns
            kris.append({
                "type": "operational",
                "description": concern,
                "severity": "medium"
            })

        return kris

    @staticmethod
    def _generate_risk_recommendations(
            risk_factors: List[Any],
            financial_risk: Dict[str, Any],
            operational_risk: Dict[str, Any]
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        # General recommendations based on risk count
        if len(risk_factors) > 20:
            recommendations.append(
                "Consider establishing a dedicated risk management committee"
            )

        # Financial recommendations
        financial_concerns = financial_risk.get("concerns", [])
        for concern in financial_concerns:
            if "liquidity" in concern.lower():
                recommendations.append(
                    "Review working capital management and explore financing options"
                )
            if "leverage" in concern.lower():
                recommendations.append(
                    "Consider debt restructuring or equity financing to improve capital structure"
                )

        # Operational recommendations
        operational_concerns = operational_risk.get("concerns", [])
        for concern in operational_concerns:
            if "supply chain" in concern.lower():
                recommendations.append(
                    "Diversify supplier base and develop contingency plans"
                )

        # Risk monitoring recommendations
        if risk_factors:
            recommendations.append(
                "Implement regular risk assessment reviews and update mitigation strategies"
            )

        return list(set(recommendations))[:5]  # Deduplicate and limit to 5

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

    @staticmethod
    def _generate_monitoring_indicators(
            risk_factors: List[Any]
    ) -> List[str]:
        """Generate monitoring indicators"""
        indicators = []

        # Monitor high severity risks
        high_severity = [rf for rf in risk_factors if rf.severity == "high"]
        for risk in high_severity[:3]:
            keywords = risk.description.lower().split()[:3]
            indicators.append(f"Monitor developments related to {' '.join(keywords)}")

        # General monitoring indicators
        if any(rf.category == "financial" for rf in risk_factors):
            indicators.append("Monitor quarterly financial results and ratios")

        if any(rf.category == "regulatory" for rf in risk_factors):
            indicators.append("Monitor regulatory changes and compliance requirements")

        return indicators[:5]  # Limit to 5

    @staticmethod
    def _calculate_next_review_date(risk_level: str) -> datetime:
        """Calculate the next review date based on risk level"""
        from datetime import timedelta

        if risk_level == "high":
            days = 30  # Monthly review for high risk
        elif risk_level == "medium":
            days = 90  # Quarterly review for medium risk
        else:
            days = 180  # Semi-annual review for low risk

        return datetime.now() + timedelta(days=days)