# infrastructure/adapters/financial_data_adapter.py
"""Financial Data Adapter for market and financial data."""

import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from domain.models.value_objects import FinancialMetric
from decimal import Decimal


class FinancialDataAdapter:
    """Adapter for financial data from various sources"""

    def __init__(self, yahoo_finance_base: str = "https://query1.finance.yahoo.com",
                 alpha_vantage_key: Optional[str] = None):
        self.yahoo_base = yahoo_finance_base
        self.alpha_vantage_key = alpha_vantage_key
        self.session = None
        self.logger = logging.getLogger(__name__)
        self._cache = {}

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def get_company_financials(self, ticker: str) -> Dict[str, Any]:
        """Get company financial statements"""
        await self._ensure_session()

        try:
            # Get key statistics
            stats_url = f"{self.yahoo_base}/v10/finance/quoteSummary/{ticker}"
            params = {
                "modules": "financialData,defaultKeyStatistics"
            }

            async with self.session.get(stats_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("quoteSummary", {}).get("result", [{}])[0]

        except Exception as e:
            self.logger.error(f"Error fetching financials for {ticker}: {e}")

        return {}

    async def get_historical_prices(self, ticker: str,
                                    start_date: datetime,
                                    end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical stock prices"""
        await self._ensure_session()

        try:
            url = f"{self.yahoo_base}/v8/finance/chart/{ticker}"
            params = {
                "period1": int(start_date.timestamp()),
                "period2": int(end_date.timestamp()),
                "interval": "1d",
                "events": "history"
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    chart_data = data.get("chart", {}).get("result", [])
                    if chart_data:
                        quotes = chart_data[0].get("indicators", {}).get("quote", [{}])[0]
                        timestamps = chart_data[0].get("timestamp", [])

                        prices = []
                        for i, ts in enumerate(timestamps):
                            prices.append({
                                "date": datetime.fromtimestamp(ts),
                                "open": quotes.get("open", [])[i] if i < len(quotes.get("open", [])) else None,
                                "high": quotes.get("high", [])[i] if i < len(quotes.get("high", [])) else None,
                                "low": quotes.get("low", [])[i] if i < len(quotes.get("low", [])) else None,
                                "close": quotes.get("close", [])[i] if i < len(quotes.get("close", [])) else None,
                                "volume": quotes.get("volume", [])[i] if i < len(quotes.get("volume", [])) else None
                            })
                        return prices

        except Exception as e:
            self.logger.error(f"Error fetching historical prices for {ticker}: {e}")

        return []

    async def calculate_financial_metrics(self, ticker: str) -> List[FinancialMetric]:
        """Calculate financial metrics from available data"""
        financials = await self.get_company_financials(ticker)

        metrics = []
        current_time = datetime.now()

        # Extract from financialData
        financial_data = financials.get("financialData", {})

        if "currentPrice" in financial_data:
            metrics.append(FinancialMetric(
                name="current_price",
                value=Decimal(str(financial_data["currentPrice"]["raw"])),
                unit="USD",
                period=current_time,
                source_document_id=f"yahoo_{ticker}",
                is_estimated=False,
                confidence=0.8
            ))

        if "totalRevenue" in financial_data:
            metrics.append(FinancialMetric(
                name="total_revenue",
                value=Decimal(str(financial_data["totalRevenue"]["raw"])),
                unit="USD",
                period=current_time,
                source_document_id=f"yahoo_{ticker}",
                is_estimated=False,
                confidence=0.7
            ))

        if "grossProfits" in financial_data:
            metrics.append(FinancialMetric(
                name="gross_profit",
                value=Decimal(str(financial_data["grossProfits"]["raw"])),
                unit="USD",
                period=current_time,
                source_document_id=f"yahoo_{ticker}",
                is_estimated=False,
                confidence=0.7
            ))

        if "ebitda" in financial_data:
            metrics.append(FinancialMetric(
                name="ebitda",
                value=Decimal(str(financial_data["ebitda"]["raw"])),
                unit="USD",
                period=current_time,
                source_document_id=f"yahoo_{ticker}",
                is_estimated=True,
                confidence=0.6
            ))

        return metrics

    async def get_industry_comparison(self, industry: str) -> Dict[str, Any]:
        """Get industry comparison data"""
        # Simplified implementation
        return {
            "industry": industry,
            "average_pe_ratio": 15.0,
            "average_pb_ratio": 2.5,
            "average_debt_to_equity": 1.2,
            "average_profit_margin": 0.1
        }

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()