# domain/repositories/sec_document_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ..models.entities import SECDocument


# Port (Interface)
class SECDocumentRepository(ABC):
    """Repository interface for SEC documents"""

    @abstractmethod
    async def find_by_cik(
            self,
            cik: str,
            filing_types: Optional[List[str]] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> List[SECDocument]:
        """Find documents by CIK with optional filters"""
        pass

    @abstractmethod
    async def find_by_accession_number(self, accession_number: str) -> Optional[SECDocument]:
        """Find the document by accession number"""
        pass

    @abstractmethod
    async def save(self, document: SECDocument) -> bool:
        """Save document to repository"""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document from repository"""
        pass

    @abstractmethod
    async def count_by_cik(self, cik: str) -> int:
        """Count documents for CIK"""
        pass