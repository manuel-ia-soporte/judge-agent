# infrastructure/external/market_data_client.py
"""Market Data API Client (External Service)"""

import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, UTC
import asyncio
import logging


class MarketDataClient:
    """Client for market data APIs (Alpha Vantage, IEX Cloud, etc.)"""

    def __init__(self, api_key: Optional[str] = None, provider: str = "alphavantage"):
        self.api_key = api_key
        self.provider = provider
        self.base_urls = {
            "alphavantage": "https://www.alphavantage.co/query",
            "iexcloud": "https://cloud.iexapis.com/stable"
        }
        self.rate_limits = {
            "alphavantage": 5,  # free tier: 5 requests per minute
            "iexcloud": 100  # depends on plan
        }
        self.semaphore = asyncio.Semaphore(self.rate_limits.get(provider, 5))
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)
        self.logger = logging.getLogger(__name__)
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company overview"""
        cache_key = f"overview_{symbol}_{self.provider}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                if self.provider == "alphavantage":
                    params = {
                        "function": "OVERVIEW",
                        "symbol": symbol,
                        "apikey": self.api_key
                    }
                    url = self.base_urls["alphavantage"]
                elif self.provider == "iexcloud":
                    params = {
                        "token": self.api_key
                    }
                    url = f"{self.base_urls['iexcloud']}/stock/{symbol}/company"
                else:
                    self.logger.error(f"Unsupported provider: {self.provider}")
                    return None

                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                self.cache[cache_key] = (data, datetime.now(UTC))
                return data

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Market Data API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting company overview for {symbol}: {e}")
                return None

    async def get_income_statement(
            self,
            symbol: str,
            period: str = "annual"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get income statement"""
        cache_key = f"income_{symbol}_{period}_{self.provider}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                if self.provider == "alphavantage":
                    params = {
                        "function": "INCOME_STATEMENT",
                        "symbol": symbol,
                        "apikey": self.api_key
                    }
                    url = self.base_urls["alphavantage"]
                elif self.provider == "iexcloud":
                    params = {
                        "token": self.api_key
                    }
                    url = f"{self.base_urls['iexcloud']}/stock/{symbol}/income"
                else:
                    self.logger.error(f"Unsupported provider: {self.provider}")
                    return None

                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # Normalize response format
                if self.provider == "alphavantage":
                    statements = data.get("annualReports" if period == "annual" else "quarterlyReports", [])
                else:  # iexcloud
                    statements = data.get("income", [])

                self.cache[cache_key] = (statements, datetime.now(UTC))
                return statements

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Market Data API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting income statement for {symbol}: {e}")
                return None

    async def get_balance_sheet(
            self,
            symbol: str,
            period: str = "annual"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get balance sheet"""
        cache_key = f"balance_{symbol}_{period}_{self.provider}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                if self.provider == "alphavantage":
                    params = {
                        "function": "BALANCE_SHEET",
                        "symbol": symbol,
                        "apikey": self.api_key
                    }
                    url = self.base_urls["alphavantage"]
                elif self.provider == "iexcloud":
                    params = {
                        "token": self.api_key
                    }
                    url = f"{self.base_urls['iexcloud']}/stock/{symbol}/balance-sheet"
                else:
                    self.logger.error(f"Unsupported provider: {self.provider}")
                    return None

                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # Normalize response format
                if self.provider == "alphavantage":
                    statements = data.get("annualReports" if period == "annual" else "quarterlyReports", [])
                else:  # iexcloud
                    statements = data.get("balancesheet", [])

                self.cache[cache_key] = (statements, datetime.now(UTC))
                return statements

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Market Data API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting balance sheet for {symbol}: {e}")
                return None

    async def get_cash_flow(
            self,
            symbol: str,
            period: str = "annual"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get the cash flow statement"""
        cache_key = f"cashflow_{symbol}_{period}_{self.provider}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                if self.provider == "alphavantage":
                    params = {
                        "function": "CASH_FLOW",
                        "symbol": symbol,
                        "apikey": self.api_key
                    }
                    url = self.base_urls["alphavantage"]
                elif self.provider == "iexcloud":
                    params = {
                        "token": self.api_key
                    }
                    url = f"{self.base_urls['iexcloud']}/stock/{symbol}/cash-flow"
                else:
                    self.logger.error(f"Unsupported provider: {self.provider}")
                    return None

                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()

                # Normalize response format
                if self.provider == "alphavantage":
                    statements = data.get("annualReports" if period == "annual" else "quarterlyReports", [])
                else:  # iexcloud
                    statements = data.get("cashflow", [])

                self.cache[cache_key] = (statements, datetime.now(UTC))
                return statements

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Market Data API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting cash flow for {symbol}: {e}")
                return None

    async def cleanup_cache(self):
        """Clean up expired cache entries"""
        now = datetime.now(UTC)
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > self.cache_ttl
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()