# tests/test_judge_agent.py
"""
Unit tests for Judge Agent
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.judge_agent.judge_agent import JudgeAgent
from agents.judge_agent.rubrics_evaluator import RubricEvaluator
from domain.models.evaluation import Evaluation, RubricEvaluation
from domain.models.finance import FinancialAnalysis, SECDocument, FilingType
from contracts.evaluation_contracts import EvaluationRequest, RubricCategory, ScoringScale
from domain.services.rubrics_service import RubricEvaluator as DomainRubricEvaluator


class TestJudgeAgent:
    """Test Judge Agent functionality"""

    @pytest.fixture
    def judge_agent(self):
        """Create judge agent instance"""
        return JudgeAgent(
            agent_id="test_judge_001",
            configuration=None
        )

    @pytest.fixture
    def sample_analysis(self):
        """Create sample financial analysis"""
        return FinancialAnalysis(
            analysis_id="test_analysis_001",
            agent_id="test_agent_001",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Apple's revenue in Q3 2023 was $81.8 billion, which represents a 2% year-over-year growth. The company maintains strong liquidity with a current ratio of 1.5.",
            metrics_used=["revenue", "current_ratio"],
            source_documents=[],
            conclusions=["Strong financial position"],
            risks_identified=["Market competition", "Supply chain disruptions"],
            assumptions=["Revenue growth continues at 2%"],
            confidence_score=0.9
        )

    @pytest.fixture
    def sample_evaluation_request(self):
        """Create sample evaluation request"""
        return EvaluationRequest(
            analysis_id="test_analysis_001",
            agent_id="test_agent_001",
            analysis_content="Apple's revenue was $81.8 billion with 2% growth.",
            source_documents=[
                {
                    "cik": "0000320193",
                    "filing_type": "10-Q",
                    "date": "2023-06-30",
                    "content": "Test SEC filing content"
                }
            ],
            rubrics_to_evaluate=[
                RubricCategory.FACTUAL_ACCURACY,
                RubricCategory.SOURCE_FIDELITY,
                RubricCategory.REGULATORY_COMPLIANCE
            ],
            scoring_scale=ScoringScale.ZERO_TO_TWO,
            minimum_threshold=1.0
        )

    @pytest.mark.asyncio
    async def test_judge_agent_initialization(self, judge_agent):
        """Test judge agent initialization"""
        assert judge_agent.agent_id == "test_judge_001"
        assert judge_agent.capabilities is not None
        assert judge_agent.metrics is not None
        assert judge_agent.configuration is not None
        assert len(judge_agent.capabilities.supported_rubrics) > 0
        assert judge_agent.is_active == True

    @pytest.mark.asyncio
    async def test_evaluate_method(self, judge_agent, sample_evaluation_request):
        """Test evaluate method"""
        with patch.object(judge_agent, '_process_rubrics') as mock_process:
            mock_process.return_value = {
                "factual_accuracy": RubricEvaluation(
                    rubric_name="factual_accuracy",
                    score=2.0,
                    is_passed=True,
                    feedback="All facts accurate",
                    evidence=["Revenue matches: $81.8B"],
                    confidence_score=1.0
                )
            }

            result = await judge_agent.evaluate(sample_evaluation_request)

            assert result is not None
            assert result.evaluation_id is not None
            assert result.agent_id == sample_evaluation_request.agent_id
            assert result.overall_score >= 0
            assert result.overall_score <= 2
            assert "factual_accuracy" in result.rubric_scores

    @pytest.mark.asyncio
    async def test_batch_evaluate(self, judge_agent):
        """Test batch evaluation"""
        requests = [
            EvaluationRequest(
                analysis_id=f"analysis_{i}",
                agent_id=f"agent_{i}",
                analysis_content=f"Test analysis {i}",
                source_documents=[],
                rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
            )
            for i in range(3)
        ]

        with patch.object(judge_agent, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = Mock(
                evaluation_id="test_eval",
                overall_score=1.5,
                passed=True
            )

            results = await judge_agent.batch_evaluate(requests)

            assert len(results) == 3
            assert mock_evaluate.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_status(self, judge_agent):
        """Test agent status reporting"""
        status = judge_agent.get_status()

        assert "agent_id" in status
        assert "status" in status
        assert "metrics" in status
        assert "queue_size" in status
        assert "capabilities" in status
        assert status["agent_id"] == judge_agent.agent_id

    @pytest.mark.asyncio
    async def test_agent_stop(self, judge_agent):
        """Test agent stop functionality"""
        await judge_agent.stop()
        assert judge_agent.is_active == False

    @pytest.mark.asyncio
    async def test_metrics_update(self, judge_agent):
        """Test metrics update after evaluation"""
        initial_metrics = judge_agent.metrics.evaluations_completed

        # Mock evaluation
        with patch.object(judge_agent, '_process_rubrics'):
            request = EvaluationRequest(
                analysis_id="test_metrics",
                agent_id="test_agent",
                analysis_content="Test",
                source_documents=[]
            )

            await judge_agent.evaluate(request)

            assert judge_agent.metrics.evaluations_completed == initial_metrics + 1


class TestRubricEvaluator:
    """Test Rubric Evaluator functionality"""

    @pytest.fixture
    def rubric_evaluator(self):
        """Create rubric evaluator instance"""
        return RubricEvaluator()

    @pytest.fixture
    def sample_sec_document(self):
        """Create sample SEC document"""
        return SECDocument(
            document_id="test_doc_001",
            company_cik="0000320193",
            company_name="Apple Inc.",
            filing_type=FilingType.FORM_10Q,
            filing_date=datetime(2023, 6, 30),
            period_end=datetime(2023, 6, 30),
            document_url="http://example.com",
            content={"revenue": 81800000000},
            raw_text="Revenue: $81,800,000,000",
            items={
                "Item 1A": "Risk factors include market competition and supply chain disruptions."
            }
        )

    def test_factual_accuracy_evaluation(self):
        """Test factual accuracy evaluation"""
        analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Revenue was $81.8 billion.",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        expected_values = {"revenue": 81800000000}

        result = DomainRubricEvaluator.evaluate_factual_accuracy(analysis, expected_values)

        assert result.rubric_name == "factual_accuracy"
        assert 0 <= result.score <= 2
        assert isinstance(result.is_passed, bool)
        assert isinstance(result.feedback, str)

    def test_source_fidelity_evaluation(self, sample_sec_document):
        """Test source fidelity evaluation"""
        analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="According to the 10-Q filing, revenue was $81.8 billion.",
            metrics_used=[],
            source_documents=[sample_sec_document],
            conclusions=[],
            risks_identified=[]
        )

        result = DomainRubricEvaluator.evaluate_source_fidelity(analysis, [sample_sec_document])

        assert result.rubric_name == "source_fidelity"
        assert 0 <= result.score <= 2

    def test_regulatory_compliance_evaluation(self):
        """Test regulatory compliance evaluation"""
        # Test compliant analysis
        compliant_analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="The filing indicates improved liquidity metrics.",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        compliant_result = DomainRubricEvaluator.evaluate_regulatory_compliance(compliant_analysis)
        assert compliant_result.score >= 1.0

        # Test non-compliant analysis
        non_compliant_analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="You should buy this stock immediately!",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        non_compliant_result = DomainRubricEvaluator.evaluate_regulatory_compliance(non_compliant_analysis)
        assert non_compliant_result.score < compliant_result.score

    def test_completeness_evaluation(self):
        """Test completeness evaluation"""
        analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Analysis with metrics, conclusions, and risks.",
            metrics_used=["revenue"],
            source_documents=[],
            conclusions=["Strong position"],
            risks_identified=["Market risk"],
            assumptions=["Growth continues"]
        )

        result = RubricEvaluator.evaluate_completeness(analysis)

        assert result.rubric_name == "completeness"
        assert 0 <= result.score <= 2
        assert "Includes" in result.evidence[0] or "Missing" in result.feedback

    def test_clarity_evaluation(self):
        """Test clarity evaluation"""
        # Clear analysis
        clear_analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="First, revenue increased. Second, margins improved. In conclusion, the company is performing well.",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        result = RubricEvaluator.evaluate_clarity(clear_analysis)
        assert result.score >= 1.0

        # Jargon-heavy analysis without explanation
        jargon_analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="The EBITDA amortization accrual derivative hedging strategy.",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        jargon_result = RubricEvaluator.evaluate_clarity(jargon_analysis)
        assert jargon_result.score < result.score

    @pytest.mark.parametrize("content,expected_score_range", [
        ("This may happen. It could occur. Possibly will.", (1.5, 2.0)),  # Good uncertainty handling
        ("This will definitely happen. It is certain.", (0.0, 1.0)),  # Overconfident
        ("Assuming growth continues. Presuming stable markets.", (1.0, 2.0)),  # Explicit assumptions
    ])
    def test_uncertainty_handling_evaluation(self, content, expected_score_range):
        """Test uncertainty handling evaluation with various inputs"""
        analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content=content,
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        result = RubricEvaluator.evaluate_uncertainty(analysis)

        assert result.rubric_name == "uncertainty_handling"
        assert expected_score_range[0] <= result.score <= expected_score_range[1]

    def test_consistency_evaluation(self):
        """Test consistency evaluation"""
        # Consistent analysis
        consistent_analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Revenue increased. Profit margins improved.",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        consistent_result = RubricEvaluator.evaluate_consistency(consistent_analysis)

        # Inconsistent analysis
        inconsistent_analysis = FinancialAnalysis(
            analysis_id="test",
            agent_id="test",
            company_ticker="AAPL",
            analysis_date=datetime.utcnow(),
            content="Revenue increased. Revenue decreased.",
            metrics_used=[],
            source_documents=[],
            conclusions=[],
            risks_identified=[]
        )

        inconsistent_result = RubricEvaluator.evaluate_consistency(inconsistent_analysis)

        assert consistent_result.score >= inconsistent_result.score

    @pytest.mark.asyncio
    async def test_comprehensive_rubric_evaluation(self, sample_analysis, sample_sec_document):
        """Test comprehensive evaluation of all rubrics"""
        result = RubricEvaluator.evaluate_all_rubrics(
            sample_analysis,
            [sample_sec_document],
            {"revenue": 81800000000}
        )

        assert len(result) == 12  # All 12 rubrics

        for rubric_name, evaluation in result.items():
            assert evaluation.rubric_name == rubric_name
            assert 0 <= evaluation.score <= 2
            assert isinstance(evaluation.is_passed, bool)
            assert isinstance(evaluation.feedback, str)
            assert isinstance(evaluation.evidence, list)


@pytest.mark.performance
class TestPerformance:
    """Performance tests for Judge Agent"""

    @pytest.mark.asyncio
    async def test_evaluation_performance(self):
        """Test evaluation performance under load"""
        judge_agent = JudgeAgent(agent_id="perf_test")

        # Create multiple evaluation requests
        requests = [
            EvaluationRequest(
                analysis_id=f"perf_analysis_{i}",
                agent_id=f"perf_agent_{i}",
                analysis_content=f"Performance test analysis {i} with revenue of ${i} billion.",
                source_documents=[],
                rubrics_to_evaluate=[
                    RubricCategory.FACTUAL_ACCURACY,
                    RubricCategory.REGULATORY_COMPLIANCE,
                    RubricCategory.FINANCIAL_REASONING
                ]
            )
            for i in range(10)
        ]

        import time
        start_time = time.time()

        # Process evaluations
        with patch.object(judge_agent, '_process_rubrics'):
            results = await asyncio.gather(*[
                judge_agent.evaluate(req) for req in requests
            ])

        end_time = time.time()
        total_time = end_time - start_time

        assert len(results) == 10
        assert total_time < 5.0  # Should complete within 5 seconds

        print(f"Processed 10 evaluations in {total_time:.2f} seconds")
        print(f"Average time per evaluation: {total_time / 10:.2f} seconds")

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during batch evaluation"""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        judge_agent = JudgeAgent(agent_id="memory_test")

        # Process multiple evaluations
        with patch.object(judge_agent, '_process_rubrics'):
            for i in range(100):
                request = EvaluationRequest(
                    analysis_id=f"mem_analysis_{i}",
                    agent_id=f"mem_agent_{i}",
                    analysis_content=f"Memory test {i}",
                    source_documents=[],
                    rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
                )

                await judge_agent.evaluate(request)

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert memory_increase < 50  # Should not increase by more than 50MB
        print(f"Memory increase: {memory_increase:.2f} MB")