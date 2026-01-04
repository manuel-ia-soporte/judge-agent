# domain/services/financial_analysis_service.py
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from ..models.entities import SECDocument
from ..models.value_objects import FinancialMetric, FinancialRatio, TrendAnalysis


@dataclass
class FinancialAnalysisService:
    """Domain Service for Financial Analysis (Stateless)"""

    # Strategy Pattern: Different analyzers for different types
    def extract_metrics(
            self,
            documents: List[SECDocument],
            metric_names: Optional[List[str]] = None
    ) -> List[FinancialMetric]:
        """Extract financial metrics from SEC documents"""
        metrics = []

        for doc in documents:
            statements = self._parse_financial_statements(doc.content)

            for statement_type, statement_data in statements.items():
                for metric_name, metric_data in statement_data.items():
                    if metric_names and metric_name not in metric_names:
                        continue

                    if isinstance(metric_data, dict) and "value" in metric_data:
                        metric = FinancialMetric(
                            name=metric_name,
                            value=Decimal(str(metric_data["value"])),
                            unit=metric_data.get("unit", "USD"),
                            period=doc.period_end,
                            source_document_id=doc.document_id,
                            footnote=metric_data.get("footnote"),
                            is_estimated=metric_data.get("is_estimated", False),
                            confidence=metric_data.get("confidence", 1.0)
                        )
                        metrics.append(metric)

        return metrics

    def calculate_ratios(
            self,
            metrics: List[FinancialMetric]
    ) -> List[FinancialRatio]:
        """Calculate financial ratios from metrics"""
        ratios = []

        # Group metrics by type and period
        metrics_by_period = self._group_metrics_by_period(metrics)

        for period, period_metrics in metrics_by_period.items():
            # Current Ratio
            current_assets = self._find_metric(period_metrics, "AssetsCurrent")
            current_liabilities = self._find_metric(period_metrics, "LiabilitiesCurrent")

            if current_assets and current_liabilities and current_liabilities.value > 0:
                current_ratio = float(current_assets.value / current_liabilities.value)
                ratios.append(FinancialRatio(
                    name="current_ratio",
                    value=current_ratio,
                    category="liquidity",
                    calculation_method="AssetsCurrent / LiabilitiesCurrent",
                    benchmark=1.5
                ))

            # Debt to Equity
            total_debt = self._find_metric(period_metrics, "LongTermDebt")
            total_equity = self._find_metric(period_metrics, "StockholdersEquity")

            if total_debt and total_equity and total_equity.value > 0:
                debt_to_equity = float(total_debt.value / total_equity.value)
                ratios.append(FinancialRatio(
                    name="debt_to_equity",
                    value=debt_to_equity,
                    category="solvency",
                    calculation_method="LongTermDebt / StockholdersEquity",
                    benchmark=2.0
                ))

            # Profit Margin
            revenue = self._find_metric(period_metrics, "RevenueFromContractWithCustomerExcludingAssessedTax")
            net_income = self._find_metric(period_metrics, "NetIncomeLoss")

            if revenue and net_income and revenue.value > 0:
                profit_margin = float(net_income.value / revenue.value)
                ratios.append(FinancialRatio(
                    name="profit_margin",
                    value=profit_margin,
                    category="profitability",
                    calculation_method="NetIncomeLoss / Revenue",
                    benchmark=0.1
                ))

        return ratios

    def analyze_trends(
            self,
            metrics: List[FinancialMetric],
            metric_name: str
    ) -> Optional[TrendAnalysis]:
        """Analyze trends for a specific metric"""
        # Filter and sort metrics
        relevant_metrics = [
            m for m in metrics
            if m.name.lower() == metric_name.lower()
        ]
        relevant_metrics.sort(key=lambda m: m.period)

        if len(relevant_metrics) < 2:
            return None

        # Extract values and periods
        values = [float(m.value) for m in relevant_metrics]
        periods = [m.period for m in relevant_metrics]

        # Calculate linear regression
        slope, intercept, r_squared = self._linear_regression(values)

        # Determine trend
        if slope > 0.05:
            trend = "increasing"
        elif slope < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        # Calculate volatility
        volatility = self._calculate_volatility(values)

        return TrendAnalysis(
            metric_name=metric_name,
            values=[Decimal(str(v)) for v in values],
            periods=periods,
            trend=trend,
            slope=slope,
            r_squared=r_squared,
            volatility=volatility
        )

    # Private helper methods
    @staticmethod
    def _parse_financial_statements(content: Dict[str, Any]) -> Dict[str, Any]:
        """Parse financial statements from content"""
        # Implementation using infrastructure adapter
        return {}

    @staticmethod
    def _group_metrics_by_period(metrics: List[FinancialMetric]) -> Dict[datetime, List[FinancialMetric]]:
        """Group metrics by period"""
        grouped = {}
        for metric in metrics:
            if metric.period not in grouped:
                grouped[metric.period] = []
            grouped[metric.period].append(metric)
        return grouped

    @staticmethod
    def _find_metric(metrics: List[FinancialMetric], name: str) -> Optional[FinancialMetric]:
        """Find metric by name"""
        for metric in metrics:
            if metric.name == name:
                return metric
        return None

    @staticmethod
    def _linear_regression(values: List[float]) -> Tuple[float, float, float]:
        """Calculate linear regression"""
        n = len(values)
        x = list(range(n))

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x_i * x_i for x_i in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_total = sum((y - y_mean) ** 2 for y in values)
        ss_residual = sum((values[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

        return slope, intercept, r_squared

    @staticmethod
    def _calculate_volatility(values: List[float]) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(values) < 2:
            return 0.0

        returns = []
        for i in range(1, len(values)):
            if values[i - 1] != 0:
                returns.append((values[i] - values[i - 1]) / values[i - 1])

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5