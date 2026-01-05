# contracts/integration/__init__.py
"""Integration contracts for external systems."""

from .sec_edgar_contracts import (
    SECFilingType,
    SECFilingRequest,
    SECFilingMetadata,
    SECCompanyInfo,
    SECFilingData,
    SECFilingResponse,
    SECCompanyFactsRequest,
    SECConceptData,
    SECCompanyFacts,
    SECSearchRequest,
    SECSearchResult,
    SECSearchResponse,
)
from .market_data_contracts import (
    MarketDataProvider,
    MarketDataType,
    MarketDataRequest,
    MarketQuote,
    PriceDataPoint,
    PriceHistory,
    FundamentalData,
    MarketDataResponse,
    MarketNewsItem,
    MarketNewsResponse,
)
from .messaging_contracts import A2AMessage

__all__ = [
    "SECFilingType",
    "SECFilingRequest",
    "SECFilingMetadata",
    "SECCompanyInfo",
    "SECFilingData",
    "SECFilingResponse",
    "SECCompanyFactsRequest",
    "SECConceptData",
    "SECCompanyFacts",
    "SECSearchRequest",
    "SECSearchResult",
    "SECSearchResponse",
    "MarketDataProvider",
    "MarketDataType",
    "MarketDataRequest",
    "MarketQuote",
    "PriceDataPoint",
    "PriceHistory",
    "FundamentalData",
    "MarketDataResponse",
    "MarketNewsItem",
    "MarketNewsResponse",
    'A2AMessage',
]