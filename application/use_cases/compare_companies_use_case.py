# application/use_cases/compare_companies_use_case.py
"""Use the case for comparing companies"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, UTC

from ..commands import CompareCompaniesCommand
from ..dtos.analysis_dtos import ComparisonResultDTO
from domain.repositories.sec_document_repository import SECDocumentRepository
from domain.services.financial_analysis_service import FinancialAnalysisService
from domain.services.risk_assessment_service import RiskAssessmentService
from domain.services.operational_analysis_service import OperationalAnalysisService
from domain.services.strategic_analysis_service import StrategicAnalysisService


@dataclass
class CompareCompaniesUseCase:
    """Use the case for comparing multiple companies"""

    sec_repository: SECDocumentRepository
    financial_service: FinancialAnalysisService
    risk_service: RiskAssessmentService
    operational_service: OperationalAnalysisService
    strategic_service: StrategicAnalysisService

    async def execute(self, command: CompareCompaniesCommand) -> ComparisonResultDTO:
        """Execute the company comparison use case"""

        # Validate input
        if len(command.company_ciks) < 2:
            raise ValueError("At least two companies required for comparison")

        if len(command.company_ciks) > 10:
            raise ValueError("Maximum 10 companies allowed for comparison")

        # Fetch data for all companies
        companies_data = await self._fetch_companies_data(command)

        # Perform comparative analysis
        comparison = await self._compare_companies(companies_data, command)

        # Generate insights
        insights = await self._generate_comparison_insights(comparison, command)

        # Prepare the result
        return ComparisonResultDTO(
            company_ciks=command.company_ciks,
            comparison_type=command.comparison_type,
            metrics=command.metrics,
            benchmark_company=command.benchmark_company,
            comparisons=comparison,
            insights=insights,
            generated_at=datetime.now(UTC)
        )

    async def _fetch_companies_data(
            self,
            command: CompareCompaniesCommand
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch data for all companies"""
        companies_data = {}

        for cik in command.company_ciks:
            try:
                # Fetch documents
                documents = await self.sec_repository.find_by_cik(
                    cik=cik,
                    filing_types=["10-K", "10-Q"],
                    start_date=command.start_date,
                    end_date=command.end_date
                )

                if not documents:
                    companies_data[cik] = {"error": f"No filings found for CIK {cik}"}
                    continue

                # Extract metrics
                metrics = self.financial_service.extract_metrics(documents)

                # Calculate ratios
                ratios = self.financial_service.calculate_ratios(metrics)

                # Extract risk factors
                risk_factors = self.risk_service.extract_risk_factors(documents)
                risk_categories = self.risk_service.categorize_risks(risk_factors)
                risk_level, risk_score = self.risk_service.assess_overall_risk(risk_factors)

                # Operational analysis
                operational_metrics = self.operational_service.extract_operational_metrics(documents)
                operational_analysis = self.operational_service.analyze_operational_efficiency(
                    operational_metrics
                )

                # Strategic analysis
                strategic_position = self.strategic_service.analyze_strategic_position(documents)
                competitive_advantage = self.strategic_service.assess_competitive_advantage(documents)

                companies_data[cik] = {
                    "documents": documents,
                    "metrics": metrics,
                    "ratios": ratios,
                    "risk_factors": risk_factors,
                    "risk_categories": risk_categories,
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "operational_metrics": operational_metrics,
                    "operational_analysis": operational_analysis,
                    "strategic_position": strategic_position,
                    "competitive_advantage": competitive_advantage,
                    "documents_count": len(documents)
                }

            except Exception as e:
                companies_data[cik] = {"error": str(e)}

        return companies_data

    async def _compare_companies(
            self,
            companies_data: Dict[str, Dict[str, Any]],
            command: CompareCompaniesCommand
    ) -> Dict[str, Any]:
        """Perform comparative analysis"""
        comparison = {
            "financial_comparison": {},
            "risk_comparison": {},
            "operational_comparison": {},
            "strategic_comparison": {},
            "rankings": {}
        }

        # Filter out companies with errors
        valid_companies = {
            cik: data for cik, data in companies_data.items()
            if "error" not in data
        }

        if len(valid_companies) < 2:
            return comparison

        # Financial comparison
        comparison["financial_comparison"] = await self._compare_financials(
            valid_companies, command.metrics
        )

        # Risk comparison
        comparison["risk_comparison"] = await self._compare_risks(valid_companies)

        # Operational comparison
        comparison["operational_comparison"] = await self._compare_operational(
            valid_companies
        )

        # Strategic comparison
        comparison["strategic_comparison"] = await self._compare_strategic(
            valid_companies
        )

        # Overall rankings
        comparison["rankings"] = await self._calculate_rankings(valid_companies)

        return comparison

    async def _compare_financials(
            self,
            companies_data: Dict[str, Dict[str, Any]],
            metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare financial metrics"""
        financial_comparison = {
            "metrics_summary": {},
            "ratios_comparison": {},
            "trends_comparison": {},
            "outliers": []
        }

        # Compare each metric
        for metric in metrics:
            values = {}
            for cik, data in companies_data.items():
                # Find metric value
                metric_value = self._find_metric_value(data["metrics"], metric)
                if metric_value is not None:
                    values[cik] = metric_value

            if values:
                # Calculate statistics
                all_values = list(values.values())
                avg = sum(all_values) / len(all_values)
                max_val = max(all_values)
                min_val = min(all_values)

                financial_comparison["metrics_summary"][metric] = {
                    "values": values,
                    "average": avg,
                    "range": max_val - min_val,
                    "max_company": max(values, key=values.get),
                    "min_company": min(values, key=values.get)
                }

        # Compare ratios
        common_ratios = ["current_ratio", "debt_to_equity", "profit_margin", "roe"]
        for ratio in common_ratios:
            ratio_values = {}
            for cik, data in companies_data.items():
                if "ratios" in data:
                    ratio_obj = next(
                        (r for r in data["ratios"] if r.name == ratio),
                        None
                    )
                    if ratio_obj:
                        ratio_values[cik] = ratio_obj.value

            if ratio_values:
                financial_comparison["ratios_comparison"][ratio] = ratio_values

        return financial_comparison

    @staticmethod
    async def _compare_risks(
            companies_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare risk profiles"""
        risk_comparison = {
            "risk_scores": {},
            "risk_levels": {},
            "risk_categories_comparison": {},
            "top_risks_by_company": {}
        }

        for cik, data in companies_data.items():
            risk_comparison["risk_scores"][cik] = data.get("risk_score", 0)
            risk_comparison["risk_levels"][cik] = data.get("risk_level", "unknown")

            # Track risk categories
            risk_categories = data.get("risk_categories", {})
            for category, risks in risk_categories.items():
                if category not in risk_comparison["risk_categories_comparison"]:
                    risk_comparison["risk_categories_comparison"][category] = {}
                risk_comparison["risk_categories_comparison"][category][cik] = len(risks)

            # Top risks
            risk_factors = data.get("risk_factors", [])
            if risk_factors:
                top_risks = [
                    rf.description[:100] for rf in risk_factors[:3]
                ]
                risk_comparison["top_risks_by_company"][cik] = top_risks

        return risk_comparison

    async def _compare_operational(
            self,
            companies_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare operational metrics"""
        operational_comparison = {
            "efficiency_scores": {},
            "working_capital_metrics": {},
            "supply_chain_resilience": {}
        }

        for cik, data in companies_data.items():
            # Efficiency scores
            operational_analysis = data.get("operational_analysis", {})
            operational_comparison["efficiency_scores"][cik] = (
                operational_analysis.get("efficiency_score", 0)
            )

            # Working capital metrics
            metrics = data.get("metrics", [])
            wc_metrics = self.operational_service.calculate_working_capital_metrics(metrics)
            operational_comparison["working_capital_metrics"][cik] = wc_metrics

        return operational_comparison

    async def _compare_strategic(
            self,
            companies_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare strategic positions"""
        strategic_comparison = {
            "competitive_advantage_scores": {},
            "innovation_capability": {},
            "market_positions": {},
            "growth_strategies": {}
        }

        for cik, data in companies_data.items():
            # Competitive advantage
            competitive_advantage = data.get("competitive_advantage", {})
            strategic_comparison["competitive_advantage_scores"][cik] = (
                competitive_advantage.get("score", 0)
            )

            # Innovation capability
            innovation = self.strategic_service.assess_innovation_capability(
                data.get("documents", [])
            )
            strategic_comparison["innovation_capability"][cik] = innovation.get("score", 0)

            # Market position
            market_position = self.strategic_service.analyze_market_position(
                data.get("documents", [])
            )
            strategic_comparison["market_positions"][cik] = market_position.get("market_position", "unknown")

        return strategic_comparison

    async def _calculate_rankings(
            self,
            companies_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Calculate overall rankings"""
        rankings = {
            "financial_strength": [],
            "risk_profile": [],
            "operational_efficiency": [],
            "strategic_position": [],
            "overall": []
        }

        # Collect scores
        scores = {}
        for cik, data in companies_data.items():
            scores[cik] = {
                "financial": self._calculate_financial_score(data),
                "risk": 1 - data.get("risk_score", 0.5),  # Invert risk score
                "operational": data.get("operational_analysis", {}).get("efficiency_score", 0),
                "strategic": data.get("competitive_advantage", {}).get("score", 0)
            }

        # Rank by category
        for category in ["financial", "risk", "operational", "strategic"]:
            sorted_companies = sorted(
                scores.items(),
                key=lambda x: x[1][category],
                reverse=True
            )
            rankings[f"{category}_strength" if category == "financial" else f"{category}_profile"] = [
                cik for cik, _ in sorted_companies
            ]

        # Overall ranking (weighted average)
        for cik, category_scores in scores.items():
            overall_score = (
                    category_scores["financial"] * 0.3 +
                    category_scores["risk"] * 0.3 +
                    category_scores["operational"] * 0.2 +
                    category_scores["strategic"] * 0.2
            )
            scores[cik]["overall"] = overall_score

        sorted_overall = sorted(
            scores.items(),
            key=lambda x: x[1]["overall"],
            reverse=True
        )
        rankings["overall"] = [cik for cik, _ in sorted_overall]

        return rankings

    @staticmethod
    async def _generate_comparison_insights(
            comparison: Dict[str, Any],
            command: CompareCompaniesCommand
    ) -> List[str]:
        """Generate insights from comparison"""
        insights = []

        financial_comparison = comparison.get("financial_comparison", {})
        risk_comparison = comparison.get("risk_comparison", {})
        rankings = comparison.get("rankings", {})

        # Financial insights
        metrics_summary = financial_comparison.get("metrics_summary", {})
        for metric, summary in metrics_summary.items():
            max_company = summary.get("max_company")
            min_company = summary.get("min_company")
            if max_company and min_company:
                insights.append(
                    f"{max_company} leads in {metric}, while {min_company} trails the group"
                )

        # Risk insights
        risk_scores = risk_comparison.get("risk_scores", {})
        if risk_scores:
            lowest_risk = min(risk_scores.items(), key=lambda x: x[1])
            highest_risk = max(risk_scores.items(), key=lambda x: x[1])
            insights.append(
                f"{lowest_risk[0]} has the lowest risk profile, "
                f"while {highest_risk[0]} has the highest risk"
            )

        # Ranking insights
        overall_rankings = rankings.get("overall", [])
        if len(overall_rankings) >= 2:
            insights.append(
                f"{overall_rankings[0]} ranks highest overall, "
                f"followed by {overall_rankings[1]}"
            )

        # Benchmark insights
        if command.benchmark_company and command.benchmark_company in command.company_ciks:
            insights.append(
                f"Analysis benchmarked against {command.benchmark_company}"
            )

        return insights

    # Helper methods
    @staticmethod
    def _find_metric_value(metrics: List[Any], metric_name: str) -> Optional[float]:
        """Find value for a specific metric"""
        for metric in metrics:
            if hasattr(metric, 'name') and metric.name.lower() == metric_name.lower():
                if hasattr(metric, 'value'):
                    return float(metric.value)
        return None

    @staticmethod
    def _calculate_financial_score(data: Dict[str, Any]) -> float:
        """Calculate financial strength score"""
        score = 0.5  # Base score

        # Consider profitability
        ratios = data.get("ratios", [])
        profit_margin = next((r for r in ratios if r.name == "profit_margin"), None)
        if profit_margin and profit_margin.value > 0.1:
            score += 0.2

        # Consider liquidity
        current_ratio = next((r for r in ratios if r.name == "current_ratio"), None)
        if current_ratio and current_ratio.value > 1.5:
            score += 0.2

        # Consider leverage
        debt_to_equity = next((r for r in ratios if r.name == "debt_to_equity"), None)
        if debt_to_equity and debt_to_equity.value < 2:
            score += 0.1

        return min(1.0, score)