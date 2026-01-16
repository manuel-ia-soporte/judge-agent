# agents/judge_agent/rubrics_evaluator.py
from domain.models.evaluation import RubricCategory, RubricScore
from domain.services.rubrics_service import RubricEvaluator as DomainRubricEvaluator


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

    @staticmethod
    def evaluate_all_rubrics(analysis, source_documents=None, expected_values=None):
        return DomainRubricEvaluator.evaluate_all_rubrics(
            analysis,
            source_documents,
            expected_values,
        )

    @staticmethod
    def evaluate_factual_accuracy(analysis, expected_values=None):
        return DomainRubricEvaluator.evaluate_factual_accuracy(analysis, expected_values)

    @staticmethod
    def evaluate_source_fidelity(analysis, source_documents=None):
        return DomainRubricEvaluator.evaluate_source_fidelity(analysis, source_documents)

    @staticmethod
    def evaluate_regulatory_compliance(analysis):
        return DomainRubricEvaluator.evaluate_regulatory_compliance(analysis)

    @staticmethod
    def evaluate_completeness(analysis):
        return DomainRubricEvaluator.evaluate_completeness(analysis)

    @staticmethod
    def evaluate_clarity(analysis):
        return DomainRubricEvaluator.evaluate_clarity_interpretability(analysis)

    @staticmethod
    def evaluate_uncertainty(analysis):
        return DomainRubricEvaluator.evaluate_uncertainty_handling(analysis)

    @staticmethod
    def evaluate_consistency(analysis):
        return DomainRubricEvaluator.evaluate_consistency(analysis)
