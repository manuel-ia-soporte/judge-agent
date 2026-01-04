# agents/finance_agent/analyzers/strategic_analyzer.py
"""Strategic analyzer implementation"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime , UTC

from .analyzer_interface import Analyzer
from ....domain.services.strategic_analysis_service import StrategicAnalysisService
from ....domain.repositories.sec_document_repository import SECDocumentRepository
from ....application.commands.analyze_company_command import AnalyzeCompanyCommand


@dataclass
class StrategicAnalyzer(Analyzer):
    """Strategic analyzer implementation"""

    sec_repository: SECDocumentRepository
    strategic_service: StrategicAnalysisService

    async def analyze(self, command: AnalyzeCompanyCommand) -> Dict[str, Any]:
        """Perform strategic analysis"""

        # Fetch documents
        documents = await self.sec_repository.find_by_cik(
            cik=command.company_cik,
            filing_types=["10-K", "10-Q"],
            start_date=command.start_date,
            end_date=command.end_date
        )

        if not documents:
            return {"error": "No documents found"}

        # Analyze strategic position
        strategic_position = self.strategic_service.analyze_strategic_position(documents)

        # Assess competitive advantage
        competitive_advantage = self.strategic_service.assess_competitive_advantage(documents)

        # Analyze growth strategies
        growth_strategies = self.strategic_service.analyze_growth_strategies(documents)

        # Identify strategic risks
        strategic_risks = self.strategic_service.identify_strategic_risks(documents)

        # Assess innovation capability
        innovation_capability = self.strategic_service.assess_innovation_capability(documents)

        # Analyze market position
        market_position = self.strategic_service.analyze_market_position(documents)

        # Generate strategic insights
        insights = self._generate_strategic_insights(
            strategic_position,
            competitive_advantage,
            growth_strategies,
            market_position
        )

        return {
            "analysis_type": "strategic",
            "company_cik": command.company_cik,
            "strategic_position": self._strategic_position_to_dict(strategic_position),
            "competitive_advantage": competitive_advantage,
            "growth_strategies": growth_strategies,
            "strategic_risks": [self._risk_factor_to_dict(rf) for rf in strategic_risks],
            "innovation_capability": innovation_capability,
            "market_position": market_position,
            "insights": insights,
            "documents_analyzed": len(documents),
            "analysis_date": datetime.now(UTC).isoformat()
        }

    def can_handle(self, analysis_type: str) -> bool:
        """Check if analyzer can handle strategic analysis"""
        return analysis_type in ["strategic", "comprehensive"]

    @staticmethod
    def _generate_strategic_insights(
            strategic_position: Any,
            competitive_advantage: Dict[str, Any],
            growth_strategies: Dict[str, Any],
            market_position: Dict[str, Any]
    ) -> List[str]:
        """Generate strategic insights"""
        insights = []

        # Competitive advantage insights
        advantage_score = competitive_advantage.get("score", 0)
        if advantage_score >= 0.7:
            insights.append("Strong competitive advantage with wide moat")
        elif advantage_score <= 0.4:
            insights.append("Limited competitive advantage, susceptible to competition")

        # Market position insights
        position = market_position.get("market_position", "unknown")
        if position == "leader":
            insights.append("Market leadership position indicated")
        elif position == "challenger":
            insights.append("Positioned as market challenger")
        elif position == "niche":
            insights.append("Niche market position identified")

        # Growth strategy insights
        primary_strategy = growth_strategies.get("primary_strategy")
        if primary_strategy:
            insights.append(f"Primary growth strategy: {primary_strategy}")

        # Strategic position insights
        if strategic_position.competitive_advantage:
            insights.append("Explicit competitive advantage mentioned")

        if strategic_position.growth_strategy:
            insights.append("Growth strategy discussed in filings")

        return insights

    @staticmethod
    def _strategic_position_to_dict(position: Any) -> Dict[str, Any]:
        """Convert strategic position to dictionary"""
        return {
            "competitive_advantage": position.competitive_advantage,
            "market_position": position.market_position,
            "growth_strategy": position.growth_strategy,
            "innovation_capability": position.innovation_capability,
            "strategic_agility": position.strategic_agility
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