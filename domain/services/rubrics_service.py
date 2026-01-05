# domain/services/rubrics_service.py
from typing import Dict

from domain.models.evaluation import RubricCategory, RubricScore


class RubricsService:
    def evaluate(self, signals: Dict[str, float]) -> Dict[RubricCategory, RubricScore]:
        results: Dict[RubricCategory, RubricScore] = {}

        for category in RubricCategory:
            raw_value: float = signals.get(category.value, 50.0)

            bounded_score: int = min(100, max(0, int(raw_value)))

            results[category] = RubricScore(
                score=bounded_score,
                rationale=f"Score derived from {category.value}",
            )

        return results
