# domain/models/entities.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass(eq=False)
class Entity:
    id: UUID = field(default_factory=uuid4, init=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.id == other.id


@dataclass(eq=False)
class SECDocument(Entity):
    cik: str
    form_type: str
    filing_date: datetime
    content: str
    accession_number: Optional[str] = None


@dataclass(eq=False)
class FinancialAnalysis(Entity):
    company_cik: str
    created_at: datetime
    summary: str
    confidence_score: float
