import pytest
from datetime import datetime

import pydantic
from pydantic import TypeAdapter
from pydantic.errors import PydanticImportError

from domain.models.finance import FinancialAnalysis, SECDocument, FilingType


try:
    pydantic.parse_raw_as  # type: ignore[attr-defined]
except (AttributeError, PydanticImportError):
    def parse_raw_as(type_, data, *, content_type=None, encoding="utf-8"):
        adapter = TypeAdapter(type_)
        if isinstance(data, (bytes, bytearray)):
            text = data.decode(encoding)
        else:
            text = data
        return adapter.validate_json(text)

    pydantic.parse_raw_as = parse_raw_as


@pytest.fixture
def sample_analysis():
    return FinancialAnalysis(
        analysis_id="fixture_analysis",
        agent_id="fixture_agent",
        company_ticker="AAPL",
        analysis_date=datetime.utcnow(),
        content=(
            "Apple's revenue in Q3 2023 was $81.8 billion with 2% growth. "
            "Liquidity remains strong with a current ratio of 1.5."
        ),
        metrics_used=["revenue", "current_ratio"],
        source_documents=[],
        conclusions=["Maintain overweight rating"],
        risks_identified=["Market competition", "Supply chain disruptions"],
        assumptions=["Growth continues at 2%"],
    )


@pytest.fixture
def sample_sec_document():
    return SECDocument(
        document_id="fixture_doc",
        company_cik="0000320193",
        company_name="Apple Inc.",
        filing_type=FilingType.FORM_10Q,
        filing_date=datetime(2023, 6, 30),
        period_end=datetime(2023, 6, 30),
        document_url="https://example.com",
        content={"revenue": 81800000000},
        raw_text="Revenue: $81.8B",
        items={},
    )
