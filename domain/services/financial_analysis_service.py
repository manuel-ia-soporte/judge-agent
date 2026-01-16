# domain/services/financial_analysis_service.py
import re
from typing import Dict, List
from domain.models.entities import SECDocument


class FinancialAnalysisService:
    """
    Domain service responsible for financial signal extraction.
    """

    def extract_metrics(self, documents: List[SECDocument]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for doc in documents:
            content = doc.content

            # Extract revenue (handle "- Total Revenue: $X" format)
            revenue_match = re.search(r'Total Revenue:\s*\$?([\d,]+)', content)
            if revenue_match:
                metrics["revenue"] = float(revenue_match.group(1).replace(',', ''))

            # Extract net income
            income_match = re.search(r'Net Income:\s*\$?([\d,]+)', content)
            if income_match:
                metrics["net_income"] = float(income_match.group(1).replace(',', ''))

            # Extract total assets
            assets_match = re.search(r'Total Assets:\s*\$?([\d,]+)', content)
            if assets_match:
                metrics["total_assets"] = float(assets_match.group(1).replace(',', ''))

            # Extract stockholders equity
            equity_match = re.search(r"Stockholders' Equity:\s*\$?([\d,]+)", content)
            if equity_match:
                metrics["stockholders_equity"] = float(equity_match.group(1).replace(',', ''))

            # Extract cash
            cash_match = re.search(r'Cash and Cash Equivalents:\s*\$?([\d,]+)', content)
            if cash_match:
                metrics["cash"] = float(cash_match.group(1).replace(',', ''))

            # Extract long-term debt
            debt_match = re.search(r'Long-term Debt:\s*\$?([\d,]+)', content)
            if debt_match:
                metrics["long_term_debt"] = float(debt_match.group(1).replace(',', ''))

            # Extract margins
            gross_margin_match = re.search(r'Gross Margin:\s*([\d.]+)%', content)
            if gross_margin_match:
                metrics["gross_margin"] = float(gross_margin_match.group(1)) / 100

            operating_margin_match = re.search(r'Operating Margin:\s*([\d.]+)%', content)
            if operating_margin_match:
                metrics["operating_margin"] = float(operating_margin_match.group(1)) / 100

            # Extract ratios
            current_ratio_match = re.search(r'Current Ratio:\s*([\d.]+)', content)
            if current_ratio_match:
                metrics["current_ratio"] = float(current_ratio_match.group(1))

            debt_equity_match = re.search(r'Debt to Equity:\s*([\d.]+)', content)
            if debt_equity_match:
                metrics["debt_to_equity"] = float(debt_equity_match.group(1))

        return metrics

    def calculate_ratios(self, metrics: Dict[str, float]) -> Dict[str, float]:
        ratios = {}

        # Profit margin from net income and revenue
        if "net_income" in metrics and "revenue" in metrics and metrics["revenue"] > 0:
            ratios["net_profit_margin"] = metrics["net_income"] / metrics["revenue"]

        # Pass through extracted ratios
        if "gross_margin" in metrics:
            ratios["gross_margin"] = metrics["gross_margin"]
        if "operating_margin" in metrics:
            ratios["operating_margin"] = metrics["operating_margin"]
        if "current_ratio" in metrics:
            ratios["current_ratio"] = metrics["current_ratio"]
        if "debt_to_equity" in metrics:
            ratios["debt_to_equity"] = metrics["debt_to_equity"]

        return ratios
