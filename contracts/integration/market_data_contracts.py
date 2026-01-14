# contracts/integration/market_data_contracts.py
"""Market data integration contracts"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum


class MarketDataProvider(str, Enum):
    """Market data providers"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    MARKETSTACK = "marketstack"
    FINNHUB = "finnhub"


class MarketDataType(str, Enum):
    """Types of market data"""
    QUOTE = "quote"
    HISTORICAL = "historical"
    INTRADAY = "intraday"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    SENTIMENT = "sentiment"


class MarketDataRequest(BaseModel):
    """Request for market data"""

    symbol: str = Field(..., description="Stock symbol")
    data_type: MarketDataType = Field(..., description="Type of data requested")
    provider: MarketDataProvider = Field(MarketDataProvider.YAHOO_FINANCE, description="Data provider")

    # For historical data
    start_date: Optional[date] = Field(None, description="Start date")
    end_date: Optional[date] = Field(None, description="End date")
    interval: Optional[str] = Field("1d", description="Data interval")

    # For fundamentals
    period: Optional[str] = Field("annual", description="Period for fundamentals")

    # Provider-specific parameters
    provider_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")

    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v):
        valid_intervals = ["1d", "1wk", "1mo", "1m", "5m", "15m", "30m", "60m"]
        if v not in valid_intervals:
            raise ValueError(f"Interval must be one of: {', '.join(valid_intervals)}")
        return v

    @field_validator('period')
    @classmethod
    def validate_period(cls, v):
        if v not in ["annual", "quarterly"]:
            raise ValueError("Period must be 'annual' or 'quarterly'")
        return v


class MarketQuote(BaseModel):
    """Market quote data"""

    symbol: str = Field(..., description="Stock symbol")
    price: float = Field(..., description="Current price")
    change: float = Field(..., description="Price change")
    change_percent: float = Field(..., description="Percentage change")
    volume: Optional[int] = Field(None, description="Trading volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    high: Optional[float] = Field(None, description="Day's high")
    low: Optional[float] = Field(None, description="Day's low")
    open: Optional[float] = Field(None, description="Opening price")
    previous_close: Optional[float] = Field(None, description="Previous close")
    timestamp: datetime = Field(..., description="Quote timestamp")
    currency: str = Field("USD", description="Currency")


class PriceDataPoint(BaseModel):
    """Historical price data point"""

    date: date = Field(..., description="Date")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: Optional[int] = Field(None, description="Trading volume")
    adjusted_close: Optional[float] = Field(None, description="Adjusted close")
    dividend: Optional[float] = Field(None, description="Dividend amount")
    split_coefficient: Optional[float] = Field(None, description="Split coefficient")


class PriceHistory(BaseModel):
    """Historical price data"""

    symbol: str = Field(..., description="Stock symbol")
    interval: str = Field(..., description="Data interval")
    data: List[PriceDataPoint] = Field(default_factory=list, description="Price data")
    currency: str = Field("USD", description="Currency")


class FundamentalData(BaseModel):
    """Fundamental financial data"""

    symbol: str = Field(..., description="Stock symbol")
    period: str = Field(..., description="Period (annual/quarterly)")
    report_date: Optional[date] = Field(None, description="Report date")

    # Income statement
    revenue: Optional[float] = Field(None, description="Revenue")
    gross_profit: Optional[float] = Field(None, description="Gross profit")
    operating_income: Optional[float] = Field(None, description="Operating income")
    net_income: Optional[float] = Field(None, description="Net income")
    eps: Optional[float] = Field(None, description="Earnings per share")

    # Balance sheet
    total_assets: Optional[float] = Field(None, description="Total assets")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities")
    total_equity: Optional[float] = Field(None, description="Total equity")
    cash: Optional[float] = Field(None, description="Cash and equivalents")
    debt: Optional[float] = Field(None, description="Total debt")

    # Cash flow
    operating_cash_flow: Optional[float] = Field(None, description="Operating cash flow")
    investing_cash_flow: Optional[float] = Field(None, description="Investing cash flow")
    financing_cash_flow: Optional[float] = Field(None, description="Financing cash flow")

    # Ratios
    current_ratio: Optional[float] = Field(None, description="Current ratio")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity")
    roe: Optional[float] = Field(None, description="Return on equity")
    roa: Optional[float] = Field(None, description="Return on assets")

    # Metadata
    source: str = Field(..., description="Data source")
    retrieved_at: datetime = Field(default_factory=datetime.now)


class MarketDataResponse(BaseModel):
    """Response with market data"""

    request: MarketDataRequest = Field(..., description="Original request")
    success: bool = Field(..., description="Whether request was successful")

    # Response data (one of these based on data_type)
    quote: Optional[MarketQuote] = Field(None, description="Quote data")
    history: Optional[PriceHistory] = Field(None, description="Historical data")
    fundamentals: Optional[List[FundamentalData]] = Field(None, description="Fundamental data")

    # Metadata
    retrieved_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: float = Field(..., description="Processing time")
    provider: MarketDataProvider = Field(..., description="Data provider")
    error_message: Optional[str] = Field(None, description="Error message if any")

    @field_validator('quote', 'history', 'fundamentals')
    @classmethod
    def validate_data_type(cls, v, info):
        # This validator runs for each field individually in Pydantic v2
        # We skip complex cross-field validation here as it's better suited for model_validator
        return v


class MarketNewsItem(BaseModel):
    """Market news item"""

    id: str = Field(..., description="News item ID")
    title: str = Field(..., description="News title")
    summary: Optional[str] = Field(None, description="News summary")
    content: Optional[str] = Field(None, description="News content")
    source: str = Field(..., description="News source")
    url: str = Field(..., description="News URL")
    published_at: datetime = Field(..., description="Publication time")
    symbols: List[str] = Field(default_factory=list, description="Related symbols")
    sentiment: Optional[float] = Field(None, description="Sentiment score (-1 to 1)")


class MarketNewsResponse(BaseModel):
    """Response with market news"""

    symbol: str = Field(..., description="Stock symbol")
    news: List[MarketNewsItem] = Field(default_factory=list, description="News items")
    retrieved_at: datetime = Field(default_factory=datetime.now)
    total_items: int = Field(0, description="Total news items available")