# application/ports/sec_filing_port.py
"""
SEC Filing Port - Hexagonal Architecture port definition.
Defines the contract for SEC filing access.
"""

from typing import List, Protocol

from domain.models.entities import SECDocument


class SECFilingPort(Protocol):
    """Port for accessing SEC filings."""

    async def get_filings(self, cik: str) -> List[SECDocument]:
        """
        Get SEC filings for a company.

        Args:
            cik: Company CIK identifier

        Returns:
            List of SEC documents
        """
        ...

    def get_filings_sync(self, cik: str) -> List[SECDocument]:
        """
        Synchronous version of get_filings.

        Args:
            cik: Company CIK identifier

        Returns:
            List of SEC documents
        """
        ...
