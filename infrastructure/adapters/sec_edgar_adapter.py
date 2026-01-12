# infrastructure/adapters/sec_edgar_adapter.py
from typing import List
from pathlib import Path
from datetime import date
from domain.entities.sec_filing import SECFiling
from domain.models.entities import SECDocument


class SECEdgarAdapter:
    """
    Infrastructure adapter: File system → Domain filing
    """

    def load_filing(self, filing_path: Path) -> SECFiling:
        if not filing_path.exists():
            raise FileNotFoundError(f"SEC filing not found: {filing_path}")

        text = filing_path.read_text(encoding="utf-8")

        # Example: infer metadata from filename (can be replaced later)
        form = filing_path.stem.split("_")[0]
        filing_date = date.fromtimestamp(filing_path.stat().st_mtime)

        return SECFiling(
            form=form,
            filing_date=filing_date,
            text=text,
            source_path=str(filing_path),
        )

    def find_by_cik(self, cik: str) -> List[SECDocument]:
        ### SHOULD IMPLEMENT
        # Placeholder: in real system, fetch from SEC or cache
        # For now, return an empty list or mock
        return []
