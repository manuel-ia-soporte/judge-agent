# domain/repositories/sec_document_repository.py
"""Repository interface for SEC documents (Port in Hexagonal Architecture)."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.entities import SECDocument


class SECDocumentRepository(ABC):
    """Abstract repository for SEC documents following Repository Pattern"""

    @abstractmethod
    async def find_by_cik(self, cik: str,
                          filing_types: Optional[List[str]] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 10) -> List[SECDocument]:
        """Find documents by CIK with optional filters"""
        pass

    @abstractmethod
    async def find_by_accession_number(self, accession_number: str) -> Optional[SECDocument]:
        """Find the document by accession number"""
        pass

    @abstractmethod
    async def find_latest_by_cik(self, cik: str, filing_type: str = "10-K") -> Optional[SECDocument]:
        """Find the latest document of the specific type"""
        pass

    @abstractmethod
    async def search(self,
                     query: Dict[str, Any],
                     limit: int = 100,
                     offset: int = 0) -> List[SECDocument]:
        """Search documents with the complex query"""
        pass

    @abstractmethod
    async def save(self, document: SECDocument) -> SECDocument:
        """Save document (create or update)"""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document by ID"""
        pass

    @abstractmethod
    async def count_by_cik(self, cik: str) -> int:
        """Count documents for a CIK"""
        pass