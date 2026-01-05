# agents/judge_agent/rubrics_evaluator.py
from domain.models.evaluation import RubricCategory, RubricScore


class RubricEvaluator:
    def evaluate(self, scores: dict) -> dict:
        return {
            RubricCategory(key): RubricScore(
                score=int(value),
                rationale="Evaluated by judge agent",
            )
            for key, value in scores.items()
        }
