# infrastructure/persistence/repositories/sec_document_repository_impl.py
"""SEC Document Repository Implementation (Adapter)"""

from typing import List, Optional, Dict, Any
from datetime import datetime, UTC, timedelta
import logging

from domain.repositories.sec_document_repository import SECDocumentRepository
from domain.models.entities import SECDocument
from infrastructure.external.sec_client import SECClient
from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter


class SECDocumentRepositoryImpl(SECDocumentRepository):
    """Implementation of SEC Document Repository using SEC EDGAR API"""

    def __init__(self, sec_client: SECClient, cache_ttl: int = 3600):
        self.sec_client = sec_client
        self.adapter = SECEdgarAdapter(sec_client)
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(__name__)

    async def find_by_cik(
            self,
            cik: str,
            filing_types: Optional[List[str]] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> List[SECDocument]:
        """Find documents by CIK using the SEC client"""
        cache_key = f"{cik}_{filing_types}_{start_date}_{end_date}"

        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Check if the cache is still valid
            if datetime.now(UTC) < cached_data['expiry']:
                self.logger.debug(f"Cache hit for {cache_key}")
                return cached_data['documents']

        self.logger.info(f"Fetching documents for CIK: {cik}")

        try:
            # Use the adapter to fetch and convert documents
            documents = await self.adapter.find_by_cik(
                cik=cik,
                filing_types=filing_types,
                start_date=start_date,
                end_date=end_date
            )

            # Cache the results
            self.cache[cache_key] = {
                'documents': documents,
                'expiry': datetime.now(UTC) + timedelta(seconds=self.cache_ttl)
            }

            return documents

        except Exception as e:
            self.logger.error(f"Error fetching documents for CIK {cik}: {e}")
            raise

    async def find_by_accession_number(self, accession_number: str) -> Optional[SECDocument]:
        """Find the document by accession number"""
        cache_key = f"accession_{accession_number}"

        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now(UTC) < cached_data['expiry']:
                return cached_data['document']

        try:
            document = await self.adapter.find_by_accession_number(accession_number)

            if document:
                self.cache[cache_key] = {
                    'document': document,
                    'expiry': datetime.now(UTC) + timedelta(seconds=self.cache_ttl)
                }

            return document

        except Exception as e:
            self.logger.error(f"Error fetching document {accession_number}: {e}")
            return None

    async def save(self, document: SECDocument) -> bool:
        """Save document (not implemented for read-only SEC API)"""
        raise NotImplementedError("SEC EDGAR is read-only")

    async def delete(self, document_id: str) -> bool:
        """Delete document (not implemented)"""
        raise NotImplementedError("SEC EDGAR is read-only")

    async def count_by_cik(self, cik: str) -> int:
        """Count documents for CIK"""
        documents = await self.find_by_cik(cik)
        return len(documents)

    async def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")

    async def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        now = datetime.now(UTC)
        expired_keys = [
            key for key, data in self.cache.items()
            if data['expiry'] < now
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")