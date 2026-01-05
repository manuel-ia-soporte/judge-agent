# infrastructure/adapters/market_data_adapter.py
from typing import List


class MarketDataAdapter:
    def normalize(self, values: List[float]) -> List[float]:
        return [v for v in values if v is not None]
