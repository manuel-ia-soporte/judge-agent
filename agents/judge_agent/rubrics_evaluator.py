# agents/judge_agent/rubrics_evaluator.py
from domain.models.evaluation import RubricCategory, RubricScore


class RubricEvaluator:
    def evaluate(self, scores: dict) -> dict:
        result = {}
        for key, value in scores.items():
            # Normalize key to RubricCategory
            if isinstance(key, RubricCategory):
                category = key
            else:
                category = RubricCategory(key)

            # Handle value - it may already be a RubricScore or a numeric value
            if isinstance(value, RubricScore):
                result[category] = value
            else:
                result[category] = RubricScore(
                    score=int(value) if isinstance(value, (int, float)) else 1,
                    rationale="Evaluated by judge agent",
                )
        return result
