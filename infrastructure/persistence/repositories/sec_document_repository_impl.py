# infrastructure/persistence/repositories/sec_document_repository_impl.py

from infrastructure.adapters.sec_edgar_adapter import SECEdgarAdapter
from domain.models.entities import SECDocument
from typing import List

class SECDocumentRepositoryImpl:
    def __init__(self, adapter: SECEdgarAdapter) -> None:
        self._adapter = adapter

    def find_by_cik(self, cik: str) -> List[SECDocument]:
        return self._adapter.find_by_cik(cik)