# infrastructure/external/yahoo_finance_client.py
"""Yahoo Finance API Client (External Service)"""

import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, UTC
import asyncio
import logging
from urllib.parse import urljoin


class YahooFinanceClient:
    """Client for Yahoo Finance API"""

    def __init__(self, rate_limit: int = 5):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance"
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)
        self.logger = logging.getLogger(__name__)
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current quote for a symbol"""
        cache_key = f"quote_{symbol}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                url = urljoin(self.base_url, f"/quote/{symbol}")

                response = await self.client.get(url)
                response.raise_for_status()

                data = response.json()
                quote = data.get("quoteResponse", {}).get("result", [{}])[0]

                self.cache[cache_key] = (quote, datetime.now(UTC))
                return quote

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Yahoo Finance API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting quote for {symbol}: {e}")
                return None

    async def get_historical_data(
            self,
            symbol: str,
            start_date: datetime,
            end_date: datetime,
            interval: str = "1d"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical price data"""
        cache_key = f"history_{symbol}_{start_date}_{end_date}_{interval}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                params = {
                    "period1": int(start_date.timestamp()),
                    "period2": int(end_date.timestamp()),
                    "interval": interval,
                    "events": "history"
                }

                url = f"{self.base_url}/chart/{symbol}"

                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                prices = data.get("chart", {}).get("result", [{}])[0].get("indicators", {}).get("quote", [{}])[0]
                timestamps = data.get("chart", {}).get("result", [{}])[0].get("timestamp", [])

                # Combine timestamps with price data
                historical_data = []
                for i, ts in enumerate(timestamps):
                    if i < len(prices.get("open", [])):
                        historical_data.append({
                            "date": datetime.fromtimestamp(ts),
                            "open": prices.get("open", [])[i],
                            "high": prices.get("high", [])[i],
                            "low": prices.get("low", [])[i],
                            "close": prices.get("close", [])[i],
                            "volume": prices.get("volume", [])[i]
                        })

                self.cache[cache_key] = (historical_data, datetime.now(UTC))
                return historical_data

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Yahoo Finance API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting historical data for {symbol}: {e}")
                return None

    async def get_financials(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get financial statements"""
        cache_key = f"financials_{symbol}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                url = f"{self.base_url}/quote/{symbol}/financials"

                response = await self.client.get(url)
                response.raise_for_status()

                data = response.json()
                financials = data.get("quoteResponse", {}).get("result", [{}])[0]

                self.cache[cache_key] = (financials, datetime.now(UTC))
                return financials

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Yahoo Finance API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting financials for {symbol}: {e}")
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