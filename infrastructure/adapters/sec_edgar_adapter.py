# infrastructure/adapters/sec_edgar_adapter.py
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
from domain.models.entities import SECDocument
from domain.repositories.sec_document_repository import SECDocumentRepository
from infrastructure.external.sec_client import SECClient


# Adapter (Implements Port)
class SECEdgarAdapter(SECDocumentRepository):
    """Adapter for SEC EDGAR API"""

    def __init__(self, sec_client: SECClient):
        self.sec_client = sec_client
        self.cache = {}  # Simple cache

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
            return self.cache[cache_key]

        # Fetch from SEC API
        filings = await self.sec_client.search_filings(
            company=cik,
            filing_types=filing_types or ["10-K", "10-Q"],
            start_date=start_date,
            end_date=end_date
        )

        # Convert to domain entities
        documents = []
        for filing in filings[:5]:  # Limit to 5 most recent
            try:
                doc = await self._convert_to_document(filing)
                documents.append(doc)
            except Exception as e:
                print(f"Log error {e}")
                continue  # Log and skip failed conversions

        self.cache[cache_key] = documents
        return documents

    async def find_by_accession_number(self, accession_number: str) -> Optional[SECDocument]:
        """Find the document by accession number"""
        filing = await self.sec_client.get_filing_by_accession(accession_number)
        if filing:
            return await self._convert_to_document(filing)
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

    async def _convert_to_document(self, filing_data: Dict[str, Any]) -> SECDocument:
        """Convert SEC API response to domain entity"""

        # Parse filing text
        filing_text = await self.sec_client.download_filing(
            filing_data.get("cik"),
            filing_data.get("accessionNumber")
        )

        # Extract items
        items = {}
        if filing_text:
            # Use regex to extract items (simplified)
            import re
            item_patterns = {
                "Item 1A": r"(ITEM\s+1A\.?\s*RISK\s+FACTORS)(.*?)(?=ITEM\s+1B|\Z)",
                "Item 7": r"(ITEM\s+7\.?\s*MANAGEMENT'S\s+DISCUSSION)(.*?)(?=ITEM\s+7A|\Z)",
                "Item 8": r"(ITEM\s+8\.?\s*FINANCIAL\s+STATEMENTS)(.*?)(?=ITEM\s+9|\Z)"
            }

            for item_name, pattern in item_patterns.items():
                match = re.search(pattern, filing_text, re.IGNORECASE | re.DOTALL)
                if match:
                    items[item_name] = match.group(2).strip()

        return SECDocument(
            document_id=filing_data.get("accessionNumber", ""),
            company_cik=filing_data.get("cik", ""),
            company_name=filing_data.get("companyName", ""),
            filing_type=filing_data.get("form", ""),
            filing_date=datetime.fromisoformat(filing_data.get("filingDate", datetime.now(UTC).isoformat())),
            period_end=datetime.fromisoformat(filing_data.get("period", {}).get("end", datetime.now(UTC).isoformat())),
            document_url=filing_data.get("filingUrl", ""),
            content=filing_data,
            raw_text=filing_text or "",
            items=items
        )