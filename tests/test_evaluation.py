# tests/test_evaluation.py
"""
Unit and integration tests for Evaluation system
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domain.models.evaluation import Evaluation, EvaluationStatus, RubricEvaluation
from domain.services.evaluation_service import EvaluationOrchestrator
from domain.services.rubrics_service import RubricEvaluator as DomainRubricEvaluator
from infrastructure.adapters.evaluation_adapter import EvaluationAdapter
from application.use_cases.evaluate_analysis import EvaluateAnalysisUseCase
from application.use_cases.score_rubrics import ScoreRubricsUseCase
from contracts.evaluation_contracts import EvaluationRequest, RubricCategory, ScoringScale


class TestEvaluationModels:
    """Test Evaluation domain models"""

    def test_evaluation_creation(self):
        """Test Evaluation entity creation"""
        evaluation = Evaluation(
            evaluation_id="test_eval_001",
            analysis_id="test_analysis_001",
            agent_id="test_agent_001"
        )

        assert evaluation.evaluation_id == "test_eval_001"
        assert evaluation.analysis_id == "test_analysis_001"
        assert evaluation.agent_id == "test_agent_001"
        assert evaluation.status == EvaluationStatus.PENDING
        assert evaluation.overall_score == 0.0
        assert evaluation.is_passed == False
        assert evaluation.created_at is not None

    def test_rubric_evaluation_creation(self):
        """Test RubricEvaluation entity creation"""
        rubric_eval = RubricEvaluation(
            rubric_name="factual_accuracy",
            score=1.8,
            is_passed=True,
            feedback="All facts accurate",
            evidence=["Revenue matches: $81.8B"],
            confidence_score=0.95
        )

        assert rubric_eval.rubric_name == "factual_accuracy"
        assert rubric_eval.score == 1.8
        assert rubric_eval.is_passed == True
        assert rubric_eval.feedback == "All facts accurate"
        assert rubric_eval.evidence == ["Revenue matches: $81.8B"]
        assert rubric_eval.confidence_score == 0.95

    def test_add_rubric_evaluation(self):
        """Test adding rubric evaluation to evaluation"""
        evaluation = Evaluation(
            evaluation_id="test_eval_002",
            analysis_id="test_analysis_002",
            agent_id="test_agent_002"
        )

        rubric_eval = RubricEvaluation(
            rubric_name="factual_accuracy",
            score=2.0,
            is_passed=True,
            feedback="Good",
            evidence=[],
            confidence_score=1.0
        )

        evaluation.add_rubric_evaluation(rubric_eval)

        assert "factual_accuracy" in evaluation.rubric_evaluations
        assert evaluation.overall_score == 2.0
        assert evaluation.is_passed == True  # Threshold is 1.5

    def test_complete_evaluation(self):
        """Test completing evaluation"""
        evaluation = Evaluation(
            evaluation_id="test_eval_003",
            analysis_id="test_analysis_003",
            agent_id="test_agent_003"
        )

        evaluation.complete_evaluation()

        assert evaluation.status == EvaluationStatus.COMPLETED
        assert evaluation.completed_at is not None

    def test_get_failed_rubrics(self):
        """Test getting failed rubrics"""
        evaluation = Evaluation(
            evaluation_id="test_eval_004",
            analysis_id="test_analysis_004",
            agent_id="test_agent_004"
        )

        # Add passed rubric
        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="passed_rubric",
            score=2.0,
            is_passed=True,
            feedback="Passed",
            evidence=[],
            confidence_score=1.0
        ))

        # Add failed rubric
        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="failed_rubric",
            score=0.5,
            is_passed=False,
            feedback="Failed",
            evidence=[],
            confidence_score=1.0
        ))

        failed = evaluation.get_failed_rubrics()

        assert "failed_rubric" in failed
        assert "passed_rubric" not in failed
        assert len(failed) == 1


class TestEvaluationService:
    """Test Evaluation service"""

    def test_calculate_weighted_score(self):
        """Test weighted score calculation"""
        rubric_scores = {
            "factual_accuracy": RubricEvaluation(
                rubric_name="factual_accuracy",
                score=2.0,
                is_passed=True,
                feedback="",
                evidence=[],
                confidence_score=1.0
            ),
            "source_fidelity": RubricEvaluation(
                rubric_name="source_fidelity",
                score=1.0,
                is_passed=True,
                feedback="",
                evidence=[],
                confidence_score=1.0
            )
        }

        weights = {
            "factual_accuracy": 0.7,
            "source_fidelity": 0.3
        }

        weighted_score = EvaluationOrchestrator.calculate_weighted_score(rubric_scores, weights)

        expected = (2.0 * 0.7 + 1.0 * 0.3) / (0.7 + 0.3)
        assert weighted_score == expected

    def test_determine_passed_status(self):
        """Test passed status determination"""
        rubric_scores = {
            "factual_accuracy": RubricEvaluation(
                rubric_name="factual_accuracy",
                score=1.8,
                is_passed=True,
                feedback="",
                evidence=[],
                confidence_score=1.0
            ),
            "source_fidelity": RubricEvaluation(
                rubric_name="source_fidelity",
                score=0.8,
                is_passed=False,
                feedback="",
                evidence=[],
                confidence_score=1.0
            )
        }

        overall_score = 1.3
        required_rubrics = ["factual_accuracy", "source_fidelity"]

        passed, failed = EvaluationOrchestrator.determine_passed_status(
            rubric_scores, overall_score, required_rubrics, min_required_score=1.5
        )

        assert passed == False
        assert "source_fidelity" in failed
        assert "overall_score" in failed

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        from domain.models.finance import FinancialAnalysis

        analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Test analysis",
            metrics_used=["revenue"],
            source_documents=[],
            conclusions=["Conclusion"],
            risks_identified=["Risk1", "Risk2"],
            assumptions=["Assumption"]
        )

        rubric_scores = {
            "factual_accuracy": RubricEvaluation(
                rubric_name="factual_accuracy",
                score=0.8,
                is_passed=False,
                feedback="Low score",
                evidence=[],
                confidence_score=1.0
            ),
            "risk_awareness": RubricEvaluation(
                rubric_name="risk_awareness",
                score=1.2,
                is_passed=True,
                feedback="OK",
                evidence=[],
                confidence_score=1.0
            )
        }

        recommendations = EvaluationOrchestrator.generate_recommendations(rubric_scores, analysis)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("Verify numerical values" in rec for rec in recommendations)

    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        rubric_scores = {
            "rubric1": RubricEvaluation(
                rubric_name="rubric1",
                score=1.5,
                is_passed=True,
                feedback="",
                evidence=[],
                confidence_score=0.9
            ),
            "rubric2": RubricEvaluation(
                rubric_name="rubric2",
                score=2.0,
                is_passed=True,
                feedback="",
                evidence=[],
                confidence_score=0.8
            )
        }

        confidence = EvaluationOrchestrator.calculate_confidence_score(rubric_scores)

        expected = (0.9 + 0.8) / 2
        assert confidence == expected

    def test_validate_evaluation_consistency(self):
        """Test evaluation consistency validation"""
        rubric_scores = {
            "factual_accuracy": RubricEvaluation(
                rubric_name="factual_accuracy",
                score=2.0,
                is_passed=True,
                feedback="",
                evidence=[],
                confidence_score=1.0
            ),
            "source_fidelity": RubricEvaluation(
                rubric_name="source_fidelity",
                score=0.5,
                is_passed=False,
                feedback="",
                evidence=[],
                confidence_score=1.0
            )
        }

        consistent, warnings = EvaluationOrchestrator.validate_evaluation_consistency(rubric_scores)

        assert consistent == False
        assert len(warnings) > 0
        assert any("High factual accuracy with low source fidelity" in w for w in warnings)


class TestEvaluationAdapter:
    """Test Evaluation adapter"""

    def test_request_to_domain(self):
        """Test converting request to domain"""
        request = EvaluationRequest(
            analysis_id="test_analysis",
            agent_id="test_agent",
            analysis_content="Test content",
            source_documents=[],
            rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
        )

        evaluation = EvaluationAdapter.request_to_domain(request)

        assert evaluation.analysis_id == request.analysis_id
        assert evaluation.agent_id == request.agent_id
        assert "request" in evaluation.metadata

    def test_domain_to_result(self):
        """Test converting domain to result"""
        evaluation = Evaluation(
            evaluation_id="test_eval",
            analysis_id="test_analysis",
            agent_id="test_agent"
        )

        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="test_rubric",
            score=1.5,
            is_passed=True,
            feedback="Test feedback",
            evidence=["Evidence1"],
            confidence_score=0.9
        ))

        evaluation.complete_evaluation()

        result = EvaluationAdapter.domain_to_result(evaluation)

        assert result.evaluation_id == evaluation.evaluation_id
        assert result.agent_id == evaluation.agent_id
        assert result.overall_score == evaluation.overall_score
        assert "test_rubric" in result.rubric_scores

    def test_rubric_to_domain(self):
        """Test converting rubric score to domain"""
        score_data = {
            "score": 1.8,
            "passed": True,
            "feedback": "Good job",
            "evidence": ["Evidence1", "Evidence2"],
            "confidence": 0.95
        }

        rubric_eval = EvaluationAdapter.rubric_to_domain("test_rubric", score_data)

        assert rubric_eval.rubric_name == "test_rubric"
        assert rubric_eval.score == 1.8
        assert rubric_eval.is_passed == True
        assert rubric_eval.feedback == "Good job"
        assert rubric_eval.evidence == ["Evidence1", "Evidence2"]
        assert rubric_eval.confidence_score == 0.95

    def test_merge_evaluations(self):
        """Test merging evaluations"""
        eval1 = Evaluation(
            evaluation_id="eval1",
            analysis_id="analysis1",
            agent_id="agent1"
        )

        eval1.add_rubric_evaluation(RubricEvaluation(
            rubric_name="rubric1",
            score=2.0,
            is_passed=True,
            feedback="Excellent",
            evidence=["E1"],
            confidence_score=1.0
        ))

        eval2 = Evaluation(
            evaluation_id="eval2",
            analysis_id="analysis1",
            agent_id="agent1"
        )

        eval2.add_rubric_evaluation(RubricEvaluation(
            rubric_name="rubric1",
            score=1.0,
            is_passed=True,
            feedback="Good",
            evidence=["E2"],
            confidence_score=0.8
        ))

        merged = EvaluationAdapter.merge_evaluations(eval1, eval2, {"primary": 0.6, "secondary": 0.4})

        assert merged.evaluation_id.startswith("merged_")
        assert "rubric1" in merged.rubric_evaluations
        assert 1.0 <= merged.rubric_evaluations["rubric1"].score <= 2.0

    def test_normalize_scores(self):
        """Test score normalization"""
        evaluation = Evaluation(
            evaluation_id="test_eval",
            analysis_id="test_analysis",
            agent_id="test_agent"
        )

        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="rubric1",
            score=0.5,
            is_passed=False,
            feedback="Low",
            evidence=[],
            confidence_score=1.0
        ))

        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="rubric2",
            score=2.0,
            is_passed=True,
            feedback="High",
            evidence=[],
            confidence_score=1.0
        ))

        normalized = EvaluationAdapter.normalize_scores(evaluation, 0.0, 1.0)

        assert len(normalized.rubric_evaluations) == 2
        assert normalized.rubric_evaluations["rubric1"].score >= 0.0
        assert normalized.rubric_evaluations["rubric2"].score <= 1.0

    def test_create_summary_report(self):
        """Test creating summary report"""
        evaluation = Evaluation(
            evaluation_id="test_eval",
            analysis_id="test_analysis",
            agent_id="test_agent"
        )

        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="rubric1",
            score=1.8,
            is_passed=True,
            feedback="Excellent",
            evidence=[],
            confidence_score=1.0
        ))

        evaluation.add_rubric_evaluation(RubricEvaluation(
            rubric_name="rubric2",
            score=0.8,
            is_passed=False,
            feedback="Poor",
            evidence=[],
            confidence_score=1.0
        ))

        report = EvaluationAdapter.create_summary_report(evaluation)

        assert "summary" in report
        assert "strengths" in report
        assert "weaknesses" in report
        assert "key_recommendations" in report
        assert report["summary"]["overall_score"] == evaluation.overall_score


class TestEvaluateAnalysisUseCase:
    """Test Evaluate Analysis use case"""

    @pytest.fixture
    def use_case(self):
        """Create use case instance with mocked dependencies"""
        mcp_client = Mock()
        a2a_client = Mock()
        sec_data_provider = Mock()

        return EvaluateAnalysisUseCase(mcp_client, a2a_client, sec_data_provider)

    @pytest.mark.asyncio
    async def test_execute(self, use_case):
        """Test use case execution"""
        request = EvaluationRequest(
            analysis_id="test_analysis",
            agent_id="test_agent",
            analysis_content="Test content",
            source_documents=[],
            rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
        )

        # Mock dependencies
        use_case._fetch_source_documents = AsyncMock(return_value=[])
        use_case._parse_analysis_content = Mock()
        use_case._extract_expected_values = Mock(return_value={})
        use_case._to_result_contract = Mock()

        result = await use_case.execute(request)

        assert result is not None
        use_case._fetch_source_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_source_documents(self, use_case):
        """Test fetching source documents"""
        source_refs = [
            {"cik": "0000320193", "filing_type": "10-Q", "date": "2023-06-30"}
        ]

        use_case.sec_data_provider.fetch_document = AsyncMock(return_value={"test": "document"})

        documents = await use_case._fetch_source_documents(source_refs)

        assert len(documents) == 1
        use_case.sec_data_provider.fetch_document.assert_called_once()


class TestScoreRubricsUseCase:
    """Test Score Rubrics use case"""

    @pytest.fixture
    def use_case(self):
        """Create use case instance"""
        sec_data_provider = Mock()
        return ScoreRubricsUseCase(sec_data_provider)

    @pytest.mark.asyncio
    async def test_execute(self, use_case):
        """Test rubric scoring execution"""
        analysis_content = "Test analysis with revenue $81.8B"
        rubrics = ["factual_accuracy", "regulatory_compliance"]
        source_documents = []

        # Mock rubric evaluation
        with patch.object(use_case, '_score_rubric') as mock_score:
            mock_score.return_value = Mock(
                rubric="factual_accuracy",
                score=1.8,
                passed=True,
                feedback="Good",
                evidence=[],
                confidence=0.9
            )

            result = await use_case.execute(analysis_content, rubrics, source_documents)

            assert len(result) == 2
            mock_score.assert_called()

    @pytest.mark.asyncio
    async def test_calculate_composite_score(self, use_case):
        """Test composite score calculation"""
        rubric_scores = {
            "factual_accuracy": Mock(
                rubric="factual_accuracy",
                score=2.0,
                passed=True,
                feedback="",
                evidence=[],
                confidence=1.0
            ),
            "source_fidelity": Mock(
                rubric="source_fidelity",
                score=1.0,
                passed=True,
                feedback="",
                evidence=[],
                confidence=1.0
            )
        }

        weights = {
            "factual_accuracy": 0.7,
            "source_fidelity": 0.3
        }

        result = await use_case.calculate_composite_score(rubric_scores, weights)

        assert "composite_score" in result
        assert "passed" in result
        assert "passed_rubrics" in result
        assert "failed_rubrics" in result
        assert result["composite_score"] == 1.7  # (2.0*0.7 + 1.0*0.3) / 1.0


@pytest.mark.integration
class TestIntegration:
    """Integration tests for evaluation system"""

    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self):
        """Test full evaluation pipeline integration"""
        from domain.models.finance import FinancialAnalysis
        from domain.services.rubrics_service import RubricEvaluator as DomainRubricEvaluator

        # Create analysis
        analysis = FinancialAnalysis(
            analysis_id="integration_test",
            agent_id="integration_agent",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Apple's Q3 2023 revenue was $81.8 billion with 2% year-over-year growth.",
            metrics_used=["revenue", "growth_rate"],
            source_documents=[],
            conclusions=["Strong performance"],
            risks_identified=["Market risk"],
            assumptions=["Continued growth"]
        )

        # Evaluate rubrics
        factual_accuracy = DomainRubricEvaluator.evaluate_factual_accuracy(
            analysis, {"revenue": 81800000000, "growth_rate": 0.02}
        )

        regulatory_compliance = DomainRubricEvaluator.evaluate_regulatory_compliance(analysis)

        # Create evaluation
        evaluation = Evaluation(
            evaluation_id="integration_eval",
            analysis_id=analysis.analysis_id,
            agent_id=analysis.agent_id
        )

        evaluation.add_rubric_evaluation(factual_accuracy)
        evaluation.add_rubric_evaluation(regulatory_compliance)
        evaluation.complete_evaluation()

        # Convert to result
        result = EvaluationAdapter.domain_to_result(evaluation)

        # Assertions
        assert evaluation.status == EvaluationStatus.COMPLETED
        assert evaluation.is_passed == (evaluation.overall_score >= 1.5)
        assert result.overall_score == evaluation.overall_score
        assert len(result.rubric_scores) == 2

    @pytest.mark.asyncio
    async def test_batch_evaluation_integration(self):
        """Test batch evaluation integration"""
        from agents.judge_agent.judge_agent import JudgeAgent

        judge_agent = JudgeAgent(agent_id="integration_judge")

        # Create multiple requests
        requests = [
            EvaluationRequest(
                analysis_id=f"batch_test_{i}",
                agent_id=f"batch_agent_{i}",
                analysis_content=f"Batch test content {i}",
                source_documents=[],
                rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY, RubricCategory.REGULATORY_COMPLIANCE]
            )
            for i in range(5)
        ]

        # Mock rubric processing
        with patch.object(judge_agent, '_process_rubrics') as mock_process:
            mock_process.return_value = {
                "factual_accuracy": RubricEvaluation(
                    rubric_name="factual_accuracy",
                    score=1.5,
                    is_passed=True,
                    feedback="Batch test",
                    evidence=[],
                    confidence_score=1.0
                ),
                "regulatory_compliance": RubricEvaluation(
                    rubric_name="regulatory_compliance",
                    score=2.0,
                    is_passed=True,
                    feedback="Compliant",
                    evidence=[],
                    confidence_score=1.0
                )
            }

            results = await judge_agent.batch_evaluate(requests)

            assert len(results) == 5
            for result in results:
                assert result.overall_score >= 0
                assert result.overall_score <= 2


@pytest.mark.contract
class TestContractValidation:
    """Contract validation tests"""

    def test_evaluation_request_contract(self):
        """Test evaluation request contract validation"""
        # Valid request
        valid_request = EvaluationRequest(
            analysis_id="test_analysis",
            agent_id="test_agent",
            analysis_content="Test content",
            source_documents=[],
            rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
        )

        assert valid_request.analysis_id == "test_analysis"
        assert valid_request.agent_id == "test_agent"

        # Test validation
        with pytest.raises(ValueError):
            EvaluationRequest(
                analysis_id="",
                agent_id="",
                analysis_content="",
                source_documents=[{"invalid": "document"}]
            )

    def test_rubric_score_contract(self):
        """Test rubric score contract validation"""
        from contracts.evaluation_contracts import RubricScore

        # Valid score
        score = RubricScore(
            rubric=RubricCategory.FACTUAL_ACCURACY,
            score=1.8,
            passed=True,
            feedback="Good job",
            evidence=["Evidence1"],
            confidence=0.95
        )

        assert score.rubric == RubricCategory.FACTUAL_ACCURACY
        assert score.score == 1.8
        assert score.passed == True

        # Test bounds
        with pytest.raises(ValueError):
            RubricScore(
                rubric=RubricCategory.FACTUAL_ACCURACY,
                score=2.5,  # Out of bounds
                passed=True,
                feedback="",
                evidence=[],
                confidence=1.0
            )

    def test_evaluation_result_contract(self):
        """Test evaluation result contract validation"""
        from contracts.evaluation_contracts import EvaluationResult

        result = EvaluationResult(
            evaluation_id="test_eval",
            request_id="test_request",
            agent_id="test_agent",
            overall_score=1.5,
            passed=True,
            rubric_scores={},
            recommendations=[],
            warnings=[],
            metadata={}
        )

        assert result.evaluation_id == "test_eval"
        assert result.overall_score == 1.5
        assert result.passed == True

        # Test bounds
        with pytest.raises(ValueError):
            EvaluationResult(
                evaluation_id="test_eval",
                request_id="test_request",
                agent_id="test_agent",
                overall_score=2.5,  # Out of bounds
                passed=True,
                rubric_scores={},
                recommendations=[],
                warnings=[],
                metadata={}
            )