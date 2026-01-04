# application/use_cases/evaluate_analysis.py
from datetime import datetime, UTC
from typing import List, Dict, Any
from ...application.interfaces.mcp_interface import MCPClient
from ...application.interfaces.a2a_interface import A2AClient
from ...domain.models.evaluation import Evaluation
from ...domain.services.rubrics_service import RubricEvaluator
from ...contracts.evaluation_contracts import EvaluationRequest, EvaluationResult


class EvaluateAnalysisUseCase:
    """Use the case for evaluating financial analysis"""

    def __init__(
            self,
            mcp_client: MCPClient,
            a2a_client: A2AClient,
            sec_data_provider: Any
    ):
        self.mcp_client = mcp_client
        self.a2a_client = a2a_client
        self.sec_data_provider = sec_data_provider

    async def execute(self, request: EvaluationRequest) -> EvaluationResult:
        """Execute the evaluation use case"""

        # 1. Create domain evaluation entity
        evaluation = Evaluation(
            evaluation_id=f"eval_{request.analysis_id}",
            analysis_id=request.analysis_id,
            agent_id=request.agent_id
        )

        # 2. Fetch source documents for verification
        sec_docs = await self._fetch_source_documents(request.source_documents)

        # 3. Parse analysis content
        analysis = self._parse_analysis_content(
            request.analysis_content,
            sec_docs
        )

        # 4. Evaluate each rubric
        rubric_scores = {}

        if "factual_accuracy" in request.rubrics_to_evaluate:
            expected_values = self._extract_expected_values(sec_docs)
            rubric_scores["factual_accuracy"] = RubricEvaluator.evaluate_factual_accuracy(
                analysis, expected_values
            )

        if "source_fidelity" in request.rubrics_to_evaluate:
            rubric_scores["source_fidelity"] = RubricEvaluator.evaluate_source_fidelity(
                analysis, sec_docs
            )

        if "regulatory_compliance" in request.rubrics_to_evaluate:
            rubric_scores["regulatory_compliance"] = RubricEvaluator.evaluate_regulatory_compliance(analysis)

        if "financial_reasoning" in request.rubrics_to_evaluate:
            rubric_scores["financial_reasoning"] = RubricEvaluator.evaluate_financial_reasoning(analysis)

        if "materiality_relevance" in request.rubrics_to_evaluate:
            material_items = ["risk factors", "md&a", "financial statements", "legal proceedings"]
            rubric_scores["materiality_relevance"] = RubricEvaluator.evaluate_materiality(
                analysis, material_items
            )

        # 5. Add all rubric evaluations
        for rubric_name, rubric_eval in rubric_scores.items():
            evaluation.add_rubric_evaluation(rubric_eval)

        # 6. Complete evaluation
        evaluation.complete_evaluation()

        # 7. Convert to result contract
        return self._to_result_contract(evaluation, request)

    async def _fetch_source_documents(self, source_refs: List[Dict]) -> List[Any]:
        """Fetch SEC documents"""
        documents = []
        for ref in source_refs:
            # This would use the SEC client to fetch actual documents
            doc = await self.sec_data_provider.fetch_document(
                ref.get("cik"),
                ref.get("filing_type"),
                ref.get("date")
            )
            if doc:
                documents.append(doc)
        return documents

    @staticmethod
    def _parse_analysis_content(content: str, source_docs: List[Any]) -> Any:
        """Parse analysis content into domain model"""
        # Simplified parsing - would be more complex in production
        from domain.models.finance import FinancialAnalysis
        return FinancialAnalysis(
            analysis_id="temp",
            agent_id="unknown",
            company_ticker="",
            analysis_date=datetime.now(UTC),
            content=content,
            metrics_used=[],
            source_documents=source_docs,
            conclusions=[],
            risks_identified=[]
        )

    @staticmethod
    def _extract_expected_values(sec_docs: List[Any]) -> Dict[str, float]:
        """Extract expected values from SEC documents"""
        # This would parse actual values from SEC filings
        return {}

    def _to_result_contract(self, evaluation: Evaluation, request: EvaluationRequest) -> EvaluationResult:
        """Convert domain entity to contract"""
        return EvaluationResult(
            evaluation_id=evaluation.evaluation_id,
            request_id=request.analysis_id,
            agent_id=evaluation.agent_id,
            overall_score=evaluation.overall_score,
            passed=evaluation.is_passed,
            rubric_scores={k: self._rubric_to_contract(v)
                           for k, v in evaluation.rubric_evaluations.items()},
            recommendations=evaluation.recommendations,
            warnings=evaluation.warnings,
            metadata=evaluation.metadata
        )

    @staticmethod
    def _rubric_to_contract(rubric_eval: Any) -> Dict:
        """Convert rubric evaluation to contract"""
        from contracts.evaluation_contracts import RubricScore
        return RubricScore(
            rubric=rubric_eval.rubric_name,
            score=rubric_eval.score,
            passed=rubric_eval.is_passed,
            feedback=rubric_eval.feedback,
            evidence=rubric_eval.evidence,
            confidence=rubric_eval.confidence_score
        )