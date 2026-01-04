# domain/services/operational_analysis_service.py
"""Operational analysis domain service"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..models.entities import SECDocument
from ..models.value_objects import FinancialMetric, RiskFactor
import re


@dataclass
class OperationalMetrics:
    """Operational performance metrics"""
    inventory_turnover: Optional[float] = None
    days_sales_outstanding: Optional[float] = None
    days_payable_outstanding: Optional[float] = None
    operating_cycle: Optional[float] = None
    asset_turnover: Optional[float] = None
    employee_productivity: Optional[float] = None


class OperationalAnalysisService:
    """Domain service for operational analysis"""

    @staticmethod
    def extract_operational_metrics(
            documents: List[SECDocument]
    ) -> OperationalMetrics:
        """Extract operational metrics from SEC documents"""
        metrics = OperationalMetrics()

        # Parse documents for operational data
        for doc in documents:
            content = doc.content

            # Look for operational metrics in MD&A and footnotes
            if "Item 7" in doc.items:
                mdna_text = doc.items["Item 7"]
                metrics = OperationalAnalysisService._parse_mdna_for_metrics(
                    mdna_text, metrics
                )

        return metrics

    @staticmethod
    def analyze_operational_efficiency(
            metrics: OperationalMetrics,
            historical_metrics: List[OperationalMetrics] = None
    ) -> Dict[str, Any]:
        """Analyze operational efficiency"""
        analysis = {
            "efficiency_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Calculate efficiency score based on metrics
        score_components = []

        if metrics.inventory_turnover:
            if metrics.inventory_turnover > 5:
                score_components.append(1.0)
                analysis["strengths"].append("High inventory turnover indicates efficient inventory management")
            elif metrics.inventory_turnover < 2:
                score_components.append(0.0)
                analysis["weaknesses"].append("Low inventory turnover suggests potential overstocking")
            else:
                score_components.append(0.5)

        if metrics.days_sales_outstanding:
            if metrics.days_sales_outstanding < 45:
                score_components.append(1.0)
                analysis["strengths"].append("Effective accounts receivable collection")
            elif metrics.days_sales_outstanding > 90:
                score_components.append(0.0)
                analysis["weaknesses"].append("Slow collections may indicate credit policy issues")
            else:
                score_components.append(0.5)

        if metrics.asset_turnover:
            if metrics.asset_turnover > 1.0:
                score_components.append(1.0)
                analysis["strengths"].append("Effective asset utilization")
            elif metrics.asset_turnover < 0.5:
                score_components.append(0.0)
                analysis["weaknesses"].append("Low asset turnover suggests underutilized assets")
            else:
                score_components.append(0.5)

        if score_components:
            analysis["efficiency_score"] = sum(score_components) / len(score_components)

        # Generate recommendations
        if analysis["efficiency_score"] < 0.5:
            analysis["recommendations"].append(
                "Review operational processes for efficiency improvements"
            )
        if metrics.inventory_turnover and metrics.inventory_turnover < 3:
            analysis["recommendations"].append(
                "Consider inventory optimization strategies"
            )

        return analysis

    @staticmethod
    def identify_operational_risks(
            documents: List[SECDocument]
    ) -> List[RiskFactor]:
        """Identify operational risks from filings"""
        risks = []

        for doc in documents:
            # Look in the risk factors section
            if "Item 1A" in doc.items:
                risk_text = doc.items["Item 1A"]
                operational_risks = (
                    OperationalAnalysisService._extract_operational_risks(risk_text)
                )
                risks.extend(operational_risks)

            # Look in MD&A for operational challenges
            if "Item 7" in doc.items:
                mdna_text = doc.items["Item 7"]
                mdna_risks = (
                    OperationalAnalysisService._extract_mdna_operational_risks(mdna_text)
                )
                risks.extend(mdna_risks)

        return risks[:10]  # Return top 10

    @staticmethod
    def calculate_working_capital_metrics(
            metrics: List[FinancialMetric]
    ) -> Dict[str, Any]:
        """Calculate working capital related metrics"""
        current_assets = next(
            (m for m in metrics if m.name == "AssetsCurrent"), None
        )
        current_liabilities = next(
            (m for m in metrics if m.name == "LiabilitiesCurrent"), None
        )
        inventory = next(
            (m for m in metrics if m.name == "InventoryNet"), None
        )
        receivables = next(
            (m for m in metrics if m.name == "AccountsReceivableNetCurrent"), None
        )

        results = {}

        if current_assets and current_liabilities:
            working_capital = float(current_assets.value - current_liabilities.value)
            results["working_capital"] = working_capital
            results["current_ratio"] = float(
                current_assets.value / current_liabilities.value
                if current_liabilities.value != 0 else 0
            )

        if inventory and receivables and current_assets and current_liabilities:
            # Quick ratio (Acid-test ratio)
            quick_assets = float(current_assets.value - inventory.value)
            results["quick_ratio"] = (
                quick_assets / float(current_liabilities.value)
                if current_liabilities.value != 0 else 0
            )

        return results

    @staticmethod
    def analyze_supply_chain_resilience(
            documents: List[SECDocument]
    ) -> Dict[str, Any]:
        """Analyze supply chain resilience from disclosures"""
        analysis = {
            "resilience_score": 0.0,
            "key_findings": [],
            "vulnerabilities": []
        }

        supply_chain_mentions = 0
        diversification_mentions = 0
        contingency_mentions = 0

        for doc in documents:
            text = doc.raw_text.lower() if doc.raw_text else ""

            # Look for supply chain discussions
            if "supply chain" in text or "supplier" in text:
                supply_chain_mentions += 1

            if "diversif" in text or "multiple supplier" in text:
                diversification_mentions += 1

            if "contingency" in text or "backup" in text or "alternative" in text:
                contingency_mentions += 1

            # Check for specific supply chain risks
            risk_keywords = [
                "single source", "sole source", "supply disruption",
                "logistics", "transportation", "raw material"
            ]

            for keyword in risk_keywords:
                if keyword in text:
                    analysis["vulnerabilities"].append(keyword)

        # Calculate resilience score
        if supply_chain_mentions > 0:
            score = 0.5  # Base score for mentioning the supply chain
            if diversification_mentions > 0:
                score += 0.25
            if contingency_mentions > 0:
                score += 0.25
            analysis["resilience_score"] = min(1.0, score)

        if diversification_mentions > 0:
            analysis["key_findings"].append(
                "Company discusses supplier diversification"
            )

        if contingency_mentions > 0:
            analysis["key_findings"].append(
                "Company has contingency plans mentioned"
            )

        return analysis

    # Private helper methods
    @staticmethod
    def _parse_mdna_for_metrics(
            mdna_text: str,
            metrics: OperationalMetrics
    ) -> OperationalMetrics:
        """Parse MD&A text for operational metrics"""
        text_lower = mdna_text.lower()

        # Look for inventory turnover mentions
        if "inventory turnover" in text_lower or "inventory days" in text_lower:
            # Try to extract number
            import re
            turnover_match = re.search(r'inventory.*?(\d+\.?\d*)', text_lower)
            if turnover_match:
                try:
                    metrics.inventory_turnover = float(turnover_match.group(1))
                except ValueError:
                    pass

        # Look for DSO mentions
        if "days sales outstanding" in text_lower or "dso" in text_lower:
            dso_match = re.search(r'dso.*?(\d+\.?\d*)', text_lower)
            if not dso_match:
                dso_match = re.search(r'days sales.*?(\d+\.?\d*)', text_lower)
            if dso_match:
                try:
                    metrics.days_sales_outstanding = float(dso_match.group(1))
                except ValueError:
                    pass

        return metrics

    @staticmethod
    def _extract_operational_risks(risk_text: str) -> List[RiskFactor]:
        """Extract operational risks from risk factor text"""
        from ..models.value_objects import RiskFactor
        from ..models.enums import SeverityLevel

        risks = []
        operational_keywords = [
            "operational", "supply chain", "production", "manufacturing",
            "quality control", "safety", "equipment", "facility",
            "business interruption", "capacity", "efficiency"
        ]

        lines = risk_text.split('.')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in operational_keywords):
                risk = RiskFactor(
                    description=line.strip()[:200],
                    category="operational",
                    severity=SeverityLevel.MEDIUM,
                    probability=0.5,
                    impact="operational"
                )
                risks.append(risk)

        return risks

    @staticmethod
    def _extract_mdna_operational_risks(mdna_text: str) -> List[RiskFactor]:
        """Extract operational risks from MD&A"""
        from ..models.value_objects import RiskFactor
        from ..models.enums import SeverityLevel

        risks = []
        challenge_keywords = [
            "challenge", "difficulty", "issue", "problem",
            "constraint", "bottleneck", "shortage", "delay"
        ]

        lines = mdna_text.split('.')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in challenge_keywords):
                # Check if it's operational context
                operational_context = any(
                    ctx in line_lower for ctx in [
                        "production", "manufacturing", "supply",
                        "operational", "process", "efficiency"
                    ]
                )

                if operational_context:
                    risk = RiskFactor(
                        description=line.strip()[:200],
                        category="operational",
                        severity=SeverityLevel.LOW,
                        probability=0.4,
                        impact="operational"
                    )
                    risks.append(risk)

        return risks