# infrastructure/adapters/evaluation_adapter.py

class EvaluationAdapter:
    def to_dict(self, scores: dict) -> dict:
        return {
            category.value: {
                "score": score.score,
                "rationale": score.rationale,
            }
            for category, score in scores.items()
        }
