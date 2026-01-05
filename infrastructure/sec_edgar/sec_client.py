# infrastructure/sec_edgar/sec_client.py
from pathlib import Path
from typing import List

from sec_edgar_downloader import Downloader


class SECClient:
    """
    Infrastructure adapter for SEC EDGAR.
    Uses sec-edgar-downloader (disk-based, stable, CI-friendly).
    """

    def __init__(self, company_name: str, email_address: str, download_dir: str = "sec_data") -> None:
        self._company_name = company_name
        self._email = email_address
        self._download_dir = Path(download_dir)

        self._downloader = Downloader(
            self._company_name,
            self._email,
            self._download_dir,
        )

    def get_filings(self, cik: str, form_type: str = "10-K") -> List[Path]:
        """
        Downloads filings and returns local file paths.

        Infrastructure concern ONLY.
        """
        self._downloader.get(form_type, cik)

        filings_dir = self._download_dir / cik / form_type
        if not filings_dir.exists():
            return []

        return list(filings_dir.glob("*.txt"))
