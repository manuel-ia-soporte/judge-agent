# infrastructure/adapters/market_data_adapter.py
"""Adapter for market data transformations"""

from typing import List, Dict, Any, Optional


class MarketDataAdapter:
    """Adapter for transforming market data between different formats"""

    @staticmethod
    def alpha_vantage_to_company_data(
            alpha_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert Alpha Vantage company overview to standard format"""
        return {
            "symbol": alpha_data.get("Symbol", ""),
            "name": alpha_data.get("Name", ""),
            "description": alpha_data.get("Description", ""),
            "sector": alpha_data.get("Sector", ""),
            "industry": alpha_data.get("Industry", ""),
            "exchange": alpha_data.get("Exchange", ""),
            "currency": alpha_data.get("Currency", ""),
            "country": alpha_data.get("Country", ""),
            "market_cap": MarketDataAdapter._parse_number(alpha_data.get("MarketCapitalization")),
            "pe_ratio": MarketDataAdapter._parse_number(alpha_data.get("PERatio")),
            "pb_ratio": MarketDataAdapter._parse_number(alpha_data.get("PriceToBookRatio")),
            "dividend_yield": MarketDataAdapter._parse_percentage(alpha_data.get("DividendYield")),
            "eps": MarketDataAdapter._parse_number(alpha_data.get("EPS")),
            "beta": MarketDataAdapter._parse_number(alpha_data.get("Beta")),
            "52_week_high": MarketDataAdapter._parse_number(alpha_data.get("52WeekHigh")),
            "52_week_low": MarketDataAdapter._parse_number(alpha_data.get("52WeekLow"))
        }

    @staticmethod
    def iex_cloud_to_company_data(
            iex_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert IEX Cloud company data to standard format"""
        return {
            "symbol": iex_data.get("symbol", ""),
            "name": iex_data.get("companyName", ""),
            "description": iex_data.get("description", ""),
            "sector": iex_data.get("sector", ""),
            "industry": iex_data.get("industry", ""),
            "exchange": iex_data.get("exchange", ""),
            "website": iex_data.get("website", ""),
            "ceo": iex_data.get("CEO", ""),
            "employees": iex_data.get("employees"),
            "country": iex_data.get("country", ""),
            "market_cap": iex_data.get("marketCap"),
            "pe_ratio": iex_data.get("peRatio"),
            "dividend_yield": MarketDataAdapter._parse_percentage(iex_data.get("dividendYield")),
            "next_earnings_date": iex_data.get("nextEarningsDate")
        }

    @staticmethod
    def yahoo_finance_to_price_history(
            yahoo_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Yahoo Finance historical data to standard format"""
        price_history = []

        for day_data in yahoo_data:
            price_history.append({
                "date": day_data.get("date"),
                "open": day_data.get("open"),
                "high": day_data.get("high"),
                "low": day_data.get("low"),
                "close": day_data.get("close"),
                "volume": day_data.get("volume"),
                "adjusted_close": day_data.get("close")  # Yahoo doesn't always provide adjusted close
            })

        return price_history

    @staticmethod
    def calculate_technical_indicators(
            price_history: List[Dict[str, Any]],
            period: int = 20
    ) -> Dict[str, List[float]]:
        """Calculate technical indicators from price history"""
        if not price_history:
            return {}

        closes = [day["close"] for day in price_history if day.get("close") is not None]

        if len(closes) < period:
            return {}

        # Simple Moving Average
        sma = MarketDataAdapter._calculate_sma(closes, period)

        # Exponential Moving Average
        ema = MarketDataAdapter._calculate_ema(closes, period)

        # Relative Strength Index
        rsi = MarketDataAdapter._calculate_rsi(closes, 14)

        # Moving Average Convergence Divergence
        macd = MarketDataAdapter._calculate_macd(closes)

        return {
            "sma": sma,
            "ema": ema,
            "rsi": rsi,
            "macd": macd
        }

    @staticmethod
    def _parse_number(value: Optional[str]) -> Optional[float]:
        """Parse string number to float"""
        if not value:
            return None

        try:
            # Remove commas and convert
            return float(value.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _parse_percentage(value: Optional[str]) -> Optional[float]:
        """Parse percentage string to float"""
        if not value:
            return None

        try:
            # Remove % sign and convert
            return float(value.replace("%", "")) / 100
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _calculate_sma(prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma = []

        for i in range(len(prices)):
            if i < period - 1:
                sma.append(None)
            else:
                sma.append(sum(prices[i - period + 1:i + 1]) / period)

        return sma

    @staticmethod
    def _calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return [None] * len(prices)

        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]

        for price in prices[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])

        # Pad beginning with None
        return [None] * (period - 1) + ema

    @staticmethod
    def _calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return [None] * len(prices)

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi = [None] * period

        for i in range(period, len(gains)):
            if i >= len(gains):
                break

            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rs = 100
            else:
                rs = avg_gain / avg_loss

            rsi_value = 100 - (100 / (1 + rs))
            rsi.append(rsi_value)

        return rsi

    @staticmethod
    def _calculate_macd(prices: List[float]) -> Dict[str, List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < 26:
            return {"macd": [], "signal": [], "histogram": []}

        # Calculate EMAs
        ema12 = MarketDataAdapter._calculate_ema(prices, 12)
        ema26 = MarketDataAdapter._calculate_ema(prices, 26)

        # Calculate MACD line
        macd_line = []
        for i in range(len(prices)):
            if ema12[i] is None or ema26[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema12[i] - ema26[i])

        # Calculate Signal line (9-day EMA of MACD line)
        # Filter out None values for calculation
        macd_values = [v for v in macd_line if v is not None]
        if len(macd_values) < 9:
            signal_line = [None] * len(macd_line)
        else:
            signal_ema = MarketDataAdapter._calculate_ema(macd_values, 9)
            # Align with the original array
            signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema

        # Calculate Histogram
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(macd_line[i] - signal_line[i])

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }