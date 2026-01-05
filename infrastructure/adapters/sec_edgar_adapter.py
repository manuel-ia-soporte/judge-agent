# infrastructure/adapters/sec_edgar_adapter.py
from domain.models.entities import SECDocument
from infrastructure.sec_edgar.sec_client import SECClient


class SECEdgarAdapter:
    def __init__(self, client: SECClient) -> None:
        self._client = client

    def find_by_cik(self, cik: str) -> list[SECDocument]:
        filings = self._client.get_filings(cik)
        return [
            SECDocument(
                cik=cik,
                form_type=f.form,
                filing_date=f.filing_date,
                content=f.text,
            )
            for f in filings
        ]
