# infrastructure/sec_edgar/sec_client.py
import httpx
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import logging
from sec_edgar_downloader import Downloader
import edgartools


class SECClient:
    """Client for SEC EDGAR API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.downloader = Downloader()
        self.edgar = edgartools.Edgar()

        # Cache for frequent queries
        self.cache: Dict[str, Any] = {}

    async def fetch_filing(self, cik: str, filing_type: str, date: str) -> Optional[Dict[str, Any]]:
        """Fetch specific filing from EDGAR"""
        cache_key = f"{cik}_{filing_type}_{date}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            async with httpx.AsyncClient() as client:
                # Construct URL for SEC API
                url = f"{self.base_url}/companyconcept/CIK{cik.zfill(10)}/us-gaap/Assets.json"

                headers = {
                    "User-Agent": "FinanceJudgeAgent/1.0",
                    "Accept-Encoding": "gzip, deflate"
                }

                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = await client.get(url, headers=headers, timeout=30.0)

                if response.status_code == 200:
                    data = response.json()
                    self.cache[cache_key] = data
                    return data
                else:
                    logging.error(f"SEC API error: {response.status_code}")
                    return None

        except Exception as e:
            logging.error(f"Failed to fetch filing: {e}")
            return None

    async def search_filings(self,
                             company: str,
                             filing_types: List[str],
                             start_date: str,
                             end_date: str) -> List[Dict[str, Any]]:
        """Search for filings"""
        try:
            filings = self.edgar.search(
                company_name=company,
                form_types=filing_types,
                start_date=start_date,
                end_date=end_date
            )

            return filings.to_dict('records')

        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

    async def download_filing(self, cik: str, accession_number: str) -> Optional[str]:
        """Download full filing text"""
        try:
            filing = self.downloader.get(
                cik=cik,
                accession_number=accession_number
            )
            return filing.text if filing else None

        except Exception as e:
            logging.error(f"Download failed: {e}")
            return None

    async def validate_filing_data(self,
                                   data: Dict[str, Any],
                                   metric: str) -> Optional[float]:
        """Extract and validate specific metric from filing"""
        try:
            # Navigate SEC JSON structure
            units = data.get("units", {})
            if "USD" in units and metric in units["USD"]:
                latest = units["USD"][metric][-1]
                return float(latest.get("val", 0))
            return None

        except Exception as e:
            logging.error(f"Validation failed for {metric}: {e}")
            return None