# agents/finance_agent/strategies/comparison_strategy.py
"""Comparison strategy for finance agent"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from domain.models.entities import SECDocument
from ....application.commands.compare_companies_command import CompareCompaniesCommand


class ComparisonStrategy(ABC):
    """Strategy interface for company comparison"""

    @abstractmethod
    async def execute(
            self,
            command: CompareCompaniesCommand,
            companies_data: Dict[str, List[SECDocument]]
    ) -> Dict[str, Any]:
        """Execute comparison strategy"""
        pass

    @abstractmethod
    def get_comparison_metrics(self) -> List[str]:
        """Get metrics used for comparison"""
        pass

    @abstractmethod
    def get_required_documents(self) -> List[str]:
        """Get required document types for comparison"""
        pass


class FinancialComparisonStrategy(ComparisonStrategy):
    """Financial comparison strategy"""

    async def execute(
            self,
            command: CompareCompaniesCommand,
            companies_data: Dict[str, List[SECDocument]]
    ) -> Dict[str, Any]:
        """Execute financial comparison"""
        comparison = {
            "comparison_type": "financial",
            "companies": list(companies_data.keys()),
            "metrics": command.metrics or self.get_comparison_metrics(),
            "results": {},
            "rankings": {},
            "insights": []
        }

        # Extract financial metrics for each company
        for company_cik, documents in companies_data.items():
            # This would use a financial service to extract metrics
            # For now, create placeholder results
            comparison["results"][company_cik] = {
                "metrics": self._extract_placeholder_metrics(documents),
                "document_count": len(documents)
            }

        # Calculate rankings
        comparison["rankings"] = self._calculate_rankings(comparison["results"])

        # Generate insights
        comparison["insights"] = self._generate_insights(comparison["results"], comparison["rankings"])

        return comparison

    def get_comparison_metrics(self) -> List[str]:
        return ["revenue", "net_income", "current_ratio", "debt_to_equity", "roe"]

    def get_required_documents(self) -> List[str]:
        return ["10-K", "10-Q"]

    @staticmethod
    def _extract_placeholder_metrics(documents: List[SECDocument]) -> Dict[str, float]:
        """Extract placeholder metrics (to be replaced with real implementation)"""
        return {
            "revenue": 1000.0,
            "net_income": 100.0,
            "current_ratio": 1.5,
            "debt_to_equity": 0.8,
            "roe": 0.15
        }

    @staticmethod
    def _calculate_rankings(results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Calculate rankings for each metric"""
        rankings = {}

        # For each metric, rank company
        metrics = ["revenue", "net_income", "current_ratio", "debt_to_equity", "roe"]

        for metric in metrics:
            company_scores = []
            for company, data in results.items():
                if "metrics" in data and metric in data["metrics"]:
                    company_scores.append((company, data["metrics"][metric]))

            # Sort by value (higher is better for most metrics, except debt_to_equity)
            if metric == "debt_to_equity":
                company_scores.sort(key=lambda x: x[1])  # Lower is better
            else:
                company_scores.sort(key=lambda x: x[1], reverse=True)  # Higher is better

            rankings[metric] = [company for company, _ in company_scores]

        return rankings

    @staticmethod
    def _generate_insights(
            results: Dict[str, Dict[str, Any]],
            rankings: Dict[str, List[str]]
    ) -> List[str]:
        """Generate comparison insights"""
        insights = []

        # Find leaders in key metrics
        key_metrics = ["revenue", "net_income", "roe"]
        for metric in key_metrics:
            if metric in rankings and rankings[metric]:
                leader = rankings[metric][0]
                insights.append(f"{leader} leads in {metric}")

        # Check for outliers
        for metric, ranking in rankings.items():
            if len(ranking) >= 3:
                # Check if top company is far ahead
                top_company = ranking[0]
                second_company = ranking[1]

                top_value = results[top_company]["metrics"].get(metric, 0)
                second_value = results[second_company]["metrics"].get(metric, 0)

                if top_value > 0 and second_value > 0:
                    ratio = top_value / second_value
                    if ratio > 1.5:  # 50% higher than next
                        insights.append(f"{top_company} dominates in {metric} ({(ratio - 1) * 100:.0f}% ahead)")

        return insights


class RiskComparisonStrategy(ComparisonStrategy):
    """Risk comparison strategy"""

    async def execute(
            self,
            command: CompareCompaniesCommand,
            companies_data: Dict[str, List[SECDocument]]
    ) -> Dict[str, Any]:
        """Execute risk comparison"""
        comparison = {
            "comparison_type": "risk",
            "companies": list(companies_data.keys()),
            "results": {},
            "risk_rankings": [],
            "insights": []
        }

        # Extract risk factors for each company
        for company_cik, documents in companies_data.items():
            # This would use a risk service to extract risk factors
            comparison["results"][company_cik] = {
                "risk_factors": self._extract_placeholder_risk_factors(documents),
                "risk_score": self._calculate_risk_score(documents),
                "document_count": len(documents)
            }

        # Rank by risk score (lower is better)
        risk_scores = [
            (company, data["risk_score"])
            for company, data in comparison["results"].items()
        ]
        risk_scores.sort(key=lambda x: x[1])
        comparison["risk_rankings"] = [company for company, _ in risk_scores]

        # Generate insights
        comparison["insights"] = self._generate_risk_insights(comparison["results"], comparison["risk_rankings"])

        return comparison

    def get_comparison_metrics(self) -> List[str]:
        return ["risk_score", "risk_factor_count", "high_risk_count"]

    def get_required_documents(self) -> List[str]:
        return ["10-K"]

    @staticmethod
    def _extract_placeholder_risk_factors(documents: List[SECDocument]) -> List[Dict[str, Any]]:
        """Extract placeholder risk factors"""
        return [
            {"description": "Market competition risk", "severity": "high"},
            {"description": "Regulatory compliance risk", "severity": "medium"},
            {"description": "Supply chain disruption risk", "severity": "low"}
        ]

    @staticmethod
    def _calculate_risk_score(documents: List[SECDocument]) -> float:
        """Calculate risk score (placeholder)"""
        return 0.5  # Placeholder

    @staticmethod
    def _generate_risk_insights(
            results: Dict[str, Dict[str, Any]],
            rankings: List[str]
    ) -> List[str]:
        """Generate risk comparison insights"""
        insights = []

        if rankings:
            lowest_risk = rankings[0]
            highest_risk = rankings[-1]

            lowest_score = results[lowest_risk]["risk_score"]
            highest_score = results[highest_risk]["risk_score"]

            insights.append(f"{lowest_risk} has the lowest risk profile (score: {lowest_score:.2f})")
            insights.append(f"{highest_risk} has the highest risk profile (score: {highest_score:.2f})")

        # Compare risk factor counts
        risk_factor_counts = {
            company: len(data["risk_factors"])
            for company, data in results.items()
        }

        if risk_factor_counts:
            max_company = max(risk_factor_counts, key=risk_factor_counts.get)
            min_company = min(risk_factor_counts, key=risk_factor_counts.get)

            insights.append(f"{max_company} discloses the most risk factors ({risk_factor_counts[max_company]})")
            insights.append(f"{min_company} discloses the fewest risk factors ({risk_factor_counts[min_company]})")

        return insights


class ComprehensiveComparisonStrategy(ComparisonStrategy):
    """Comprehensive comparison strategy"""

    async def execute(
            self,
            command: CompareCompaniesCommand,
            companies_data: Dict[str, List[SECDocument]]
    ) -> Dict[str, Any]:
        """Execute comprehensive comparison"""
        # Use multiple strategies
        financial_strategy = FinancialComparisonStrategy()
        risk_strategy = RiskComparisonStrategy()

        # Execute both comparisons
        financial_comparison = await financial_strategy.execute(command, companies_data)
        risk_comparison = await risk_strategy.execute(command, companies_data)

        # Combine results
        comparison = {
            "comparison_type": "comprehensive",
            "companies": list(companies_data.keys()),
            "financial_comparison": financial_comparison,
            "risk_comparison": risk_comparison,
            "overall_rankings": self._calculate_overall_rankings(financial_comparison, risk_comparison),
            "comprehensive_insights": self._generate_comprehensive_insights(financial_comparison, risk_comparison)
        }

        return comparison

    def get_comparison_metrics(self) -> List[str]:
        return FinancialComparisonStrategy().get_comparison_metrics() + RiskComparisonStrategy().get_comparison_metrics()

    def get_required_documents(self) -> List[str]:
        return list(set(
            FinancialComparisonStrategy().get_required_documents() +
            RiskComparisonStrategy().get_required_documents()
        ))

    @staticmethod
    def _calculate_overall_rankings(
            financial_comparison: Dict[str, Any],
            risk_comparison: Dict[str, Any]
    ) -> List[str]:
        """Calculate overall rankings"""
        companies = financial_comparison.get("companies", [])

        if not companies:
            return []

        # Calculate composite scores (weighted average of financial and risk rankings)
        scores = {}
        for company in companies:
            # Financial ranking score (1 for top, decreasing)
            financial_rankings = financial_comparison.get("rankings", {})
            financial_score = 0
            for metric, ranking in financial_rankings.items():
                if company in ranking:
                    position = ranking.index(company) + 1
                    # Invert so higher is better (1st place gets len(companies) points)
                    financial_score += len(companies) - position + 1

            # Risk ranking score (lower risk is better)
            risk_rankings = risk_comparison.get("risk_rankings", [])
            if company in risk_rankings:
                risk_position = risk_rankings.index(company) + 1
                risk_score = len(companies) - risk_position + 1
            else:
                risk_score = 0

            # Weighted composite score (70% financial, 30% risk)
            composite_score = (financial_score * 0.7) + (risk_score * 0.3)
            scores[company] = composite_score

        # Sort by composite score
        sorted_companies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [company for company, _ in sorted_companies]

    @staticmethod
    def _generate_comprehensive_insights(
            financial_comparison: Dict[str, Any],
            risk_comparison: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive insights"""
        insights = []

        # Combine insights from both comparisons
        insights.extend(financial_comparison.get("insights", []))
        insights.extend(risk_comparison.get("insights", []))

        # Add integrated insights
        financial_rankings = financial_comparison.get("rankings", {})
        risk_rankings = risk_comparison.get("risk_rankings", [])

        if financial_rankings.get("roe") and risk_rankings:
            # Check for companies with both high ROE and low risk
            top_roe_companies = financial_rankings["roe"][:2]  # Top 2 by ROE
            low_risk_companies = risk_rankings[:2]  # Top 2 by low risk

            intersection = set(top_roe_companies) & set(low_risk_companies)
            if intersection:
                insights.append(f"Companies with both high ROE and low risk: {', '.join(intersection)}")

        return insights