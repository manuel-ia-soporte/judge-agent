# domain/entities/sec_filing.py
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SECFiling:
    form: str
    filing_date: date
    text: str
    source_path: str
