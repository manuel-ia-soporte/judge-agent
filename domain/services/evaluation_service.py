# domain/services/evaluation_service.py
from typing import Dict
from domain.models.evaluation import RubricCategory, RubricScore


class EvaluationService:
    def aggregate(self, scores: Dict[RubricCategory, RubricScore]) -> float:
        if not scores:
            return 0.0
        return sum(s.score for s in scores.values()) / len(scores)
