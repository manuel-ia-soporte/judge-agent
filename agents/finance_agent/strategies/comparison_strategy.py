# agents/finance_agent/strategies/comparison_strategy.py
from typing import Dict, Any, List


class ComparisonStrategy:
    def compare(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "peer_count": len(analyses),
            "note": "Comparison logic to be implemented in Phase 5",
        }
