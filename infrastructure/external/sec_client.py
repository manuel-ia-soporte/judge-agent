# infrastructure/external/sec_client.py
"""SEC EDGAR API Client (External Service)"""

import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, UTC
import asyncio
import logging
from urllib.parse import urljoin


class SECClient:
    """Client for SEC EDGAR API with rate limiting and caching"""

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 10):
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.api_key = api_key
        self.rate_limit = rate_limit  # requests per second
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        self.logger = logging.getLogger(__name__)
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "FinanceJudgeSystem/1.0 (contact@example.com)",
                "Accept-Encoding": "gzip, deflate"
            }
        )

    async def search_filings(
            self,
            company: str,
            filing_types: List[str],
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Search for filings by company and filing types"""
        cache_key = f"search_{company}_{filing_types}_{start_date}_{end_date}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            self.logger.debug(f"Cache hit for {cache_key}")
            return cached[0]

        async with self.semaphore:
            try:
                # Construct query parameters
                params = {
                    "company": company,
                    "formTypes": ",".join(filing_types)
                }

                if start_date:
                    params["startDate"] = start_date.strftime("%Y-%m-%d")

                if end_date:
                    params["endDate"] = end_date.strftime("%Y-%m-%d")

                # SEC API endpoint for searching
                url = urljoin(self.base_url, "/submissions/search")

                response = await self.client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                filings = data.get("filings", [])

                # Cache the results
                self.cache[cache_key] = (filings, datetime.now(UTC))

                return filings

            except httpx.HTTPStatusError as e:
                self.logger.error(f"SEC API error: {e.response.status_code} - {e.response.text}")
                return []
            except Exception as e:
                self.logger.error(f"Error searching filings: {e}")
                return []

    async def get_filing_by_accession(self, accession_number: str) -> Optional[Dict[str, Any]]:
        """Get filing by accession number"""
        cache_key = f"filing_{accession_number}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                url = urljoin(self.base_url, f"/files/{accession_number}")

                response = await self.client.get(url)
                response.raise_for_status()

                filing = response.json()

                self.cache[cache_key] = (filing, datetime.now(UTC))
                return filing

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    self.logger.warning(f"Filing not found: {accession_number}")
                else:
                    self.logger.error(f"SEC API error: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting filing: {e}")
                return None

    async def download_filing_text(self, accession_number: str) -> Optional[str]:
        """Download full filing text"""
        cache_key = f"text_{accession_number}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                # Construct URL for the filing text
                # The accession number is like 0000320193-20-000096
                # Convert to URL: https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/0000320193-20-000096.txt
                parts = accession_number.split('-')
                if len(parts) < 3:
                    self.logger.error(f"Invalid accession number format: {accession_number}")
                    return None

                cik = parts[0].lstrip('0')
                year = parts[1]
                rest = parts[2]

                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{accession_number}.txt"

                response = await self.client.get(url)
                response.raise_for_status()

                text = response.text

                self.cache[cache_key] = (text, datetime.now(UTC))
                return text

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Error downloading filing text: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error downloading filing text: {e}")
                return None

    async def get_company_facts(self, cik: str) -> Optional[Dict[str, Any]]:
        """Get company facts (XBRL data)"""
        cache_key = f"facts_{cik}"

        cached = self.cache.get(cache_key)
        if cached and datetime.now(UTC) - cached[1] < self.cache_ttl:
            return cached[0]

        async with self.semaphore:
            try:
                url = urljoin(self.base_url, f"/companyfacts/CIK{cik.zfill(10)}.json")

                response = await self.client.get(url)
                response.raise_for_status()

                facts = response.json()

                self.cache[cache_key] = (facts, datetime.now(UTC))
                return facts

            except httpx.HTTPStatusError as e:
                self.logger.error(f"Error getting company facts: {e.response.status_code}")
                return None
            except Exception as e:
                self.logger.error(f"Error getting company facts: {e}")
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