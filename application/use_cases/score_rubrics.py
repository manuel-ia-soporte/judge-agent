# application/use_cases/score_rubrics.py
from typing import Dict, List, Any, Optional
from application.interfaces.mcp_interface import MCPClient
from application.interfaces.a2a_interface import A2AClient
from domain.models.evaluation import RubricEvaluation
from domain.services.rubrics_service import RubricEvaluator
from domain.services.evaluation_service import EvaluationOrchestrator
from contracts.evaluation_contracts import EvaluationRequest, RubricScore


class ScoreRubricsUseCase:
    """Use case for scoring individual rubrics"""

    def __init__(self, sec_data_provider: Any):
        self.sec_data_provider = sec_data_provider
        self.rubric_evaluators = {
            "factual_accuracy": RubricEvaluator.evaluate_factual_accuracy,
            "source_fidelity": RubricEvaluator.evaluate_source_fidelity,
            "regulatory_compliance": RubricEvaluator.evaluate_regulatory_compliance,
            "financial_reasoning": RubricEvaluator.evaluate_financial_reasoning,
            "materiality_relevance": RubricEvaluator.evaluate_materiality,
        }

    async def execute(
            self,
            analysis_content: str,
            rubrics: List[str],
            source_documents: List[Dict[str, Any]],
            expected_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, RubricScore]:
        """Score specified rubrics for analysis"""

        # Parse analysis
        from domain.models.finance import FinancialAnalysis
        analysis = FinancialAnalysis(
            analysis_id="scoring_analysis",
            agent_id="scoring_agent",
            company_ticker="",
            analysis_date=datetime.utcnow(),
            content=analysis_content,
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        # Fetch source documents if provided
        sec_docs = []
        if source_documents:
            sec_docs = await self._fetch_documents(source_documents)

        # Score each rubric
        rubric_scores = {}

        for rubric in rubrics:
            if rubric in self.rubric_evaluators:
                score_data = await self._score_rubric(
                    rubric, analysis, sec_docs, expected_values
                )
                rubric_scores[rubric] = score_data

        return rubric_scores

    async def _score_rubric(
            self,
            rubric: str,
            analysis: FinancialAnalysis,
            sec_docs: List[Any],
            expected_values: Optional[Dict[str, float]]
    ) -> RubricScore:
        """Score individual rubric"""

        if rubric == "factual_accuracy":
            eval_result = RubricEvaluator.evaluate_factual_accuracy(
                analysis, expected_values or {}
            )
        elif rubric == "source_fidelity":
            eval_result = RubricEvaluator.evaluate_source_fidelity(analysis, sec_docs)
        elif rubric == "regulatory_compliance":
            eval_result = RubricEvaluator.evaluate_regulatory_compliance(analysis)
        elif rubric == "financial_reasoning":
            eval_result = RubricEvaluator.evaluate_financial_reasoning(analysis)
        elif rubric == "materiality_relevance":
            material_items = ["risk factors", "financial statements", "md&a"]
            eval_result = RubricEvaluator.evaluate_materiality(analysis, material_items)
        else:
            # Default for other rubrics
            eval_result = RubricEvaluation(
                rubric_name=rubric,
                score=1.0,
                is_passed=True,
                feedback="Not implemented",
                evidence=[]
            )

        # Convert to RubricScore contract
        return RubricScore(
            rubric=eval_result.rubric_name,
            score=eval_result.score,
            passed=eval_result.is_passed,
            feedback=eval_result.feedback,
            evidence=eval_result.evidence,
            confidence=eval_result.confidence_score
        )

    async def _fetch_documents(self, source_refs: List[Dict]) -> List[Any]:
        """Fetch source documents"""
        documents = []
        for ref in source_refs:
            # Simplified - would use SEC client
            doc = {
                "cik": ref.get("cik"),
                "type": ref.get("filing_type"),
                "date": ref.get("date"),
                "content": ref.get("content", "")
            }
            documents.append(doc)
        return documents

    async def calculate_composite_score(
            self,
            rubric_scores: Dict[str, RubricScore],
            weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate composite score from rubric scores"""

        # Calculate weighted score
        total_weight = sum(weights.get(r, 1.0) for r in rubric_scores.keys())
        weighted_sum = sum(
            score.score * weights.get(rubric, 1.0)
            for rubric, score in rubric_scores.items()
        )

        composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine if passed
        passed = all(score.passed for score in rubric_scores.values())

        # Generate summary
        passed_rubrics = [r for r, s in rubric_scores.items() if s.passed]
        failed_rubrics = [r for r, s in rubric_scores.items() if not s.passed]

        return {
            "composite_score": composite_score,
            "passed": passed,
            "passed_rubrics": passed_rubrics,
            "failed_rubrics": failed_rubrics,
            "weighted_score_breakdown": {
                rubric: {
                    "score": score.score,
                    "weight": weights.get(rubric, 1.0),
                    "weighted_score": score.score * weights.get(rubric, 1.0)
                }
                for rubric, score in rubric_scores.items()
            }
        }