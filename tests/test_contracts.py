# tests/test_contracts.py
"""
Contract validation tests
"""
import pytest
from datetime import datetime
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from contracts.evaluation_contracts import (
    EvaluationRequest, EvaluationResult, RubricScore, A2AMessage,
    RubricCategory, ScoringScale
)
from contracts.finance_contracts import (
    SECFilingRequest, CompanyFinancials, FinancialMetricData,
    MarketDataRequest, RiskAssessment
)
from contracts.judge_contracts import (
    JudgeCapabilities, JudgeMetrics, JudgeConfiguration
)


class TestEvaluationContracts:
    """Test evaluation contract validation"""

    def test_evaluation_request_validation(self):
        """Test EvaluationRequest validation"""
        # Valid request
        request = EvaluationRequest(
            analysis_id="test_analysis",
            agent_id="test_agent",
            analysis_content="Test content",
            source_documents=[
                {
                    "document_type": "10-K",
                    "content": "Filing content",
                    "cik": "0000320193",
                    "date": "2023-12-31"
                }
            ],
            rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY],
            scoring_scale=ScoringScale.ZERO_TO_TWO
        )

        assert request.analysis_id == "test_analysis"
        assert request.agent_id == "test_agent"
        assert len(request.source_documents) == 1

        # Test invalid source documents
        with pytest.raises(ValidationError):
            EvaluationRequest(
                analysis_id="test",
                agent_id="test",
                analysis_content="",
                source_documents=[{"invalid": "structure"}]
            )

        # Test minimum threshold bounds
        with pytest.raises(ValidationError):
            EvaluationRequest(
                analysis_id="test",
                agent_id="test",
                analysis_content="",
                source_documents=[],
                minimum_threshold=3.0  # Out of bounds
            )

    def test_rubric_score_validation(self):
        """Test RubricScore validation"""
        # Valid score
        score = RubricScore(
            rubric=RubricCategory.FACTUAL_ACCURACY,
            score=1.8,
            passed=True,
            feedback="Accurate",
            evidence=["Evidence1"],
            confidence=0.95
        )

        assert score.rubric == RubricCategory.FACTUAL_ACCURACY
        assert score.score == 1.8
        assert score.confidence == 0.95

        # Test score bounds
        with pytest.raises(ValidationError):
            RubricScore(
                rubric=RubricCategory.FACTUAL_ACCURACY,
                score=2.5,  # Out of bounds
                passed=True,
                feedback="",
                evidence=[],
                confidence=1.0
            )

        # Test confidence bounds
        with pytest.raises(ValidationError):
            RubricScore(
                rubric=RubricCategory.FACTUAL_ACCURACY,
                score=1.5,
                passed=True,
                feedback="",
                evidence=[],
                confidence=1.5  # Out of bounds
            )

    def test_evaluation_result_validation(self):
        """Test EvaluationResult validation"""
        # Valid result
        result = EvaluationResult(
            evaluation_id="test_eval",
            request_id="test_request",
            agent_id="test_agent",
            overall_score=1.5,
            passed=True,
            rubric_scores={
                RubricCategory.FACTUAL_ACCURACY: RubricScore(
                    rubric=RubricCategory.FACTUAL_ACCURACY,
                    score=1.8,
                    passed=True,
                    feedback="Good",
                    evidence=[],
                    confidence=1.0
                )
            }
        )

        assert result.evaluation_id == "test_eval"
        assert result.overall_score == 1.5
        assert len(result.rubric_scores) == 1

        # Test overall score bounds
        with pytest.raises(ValidationError):
            EvaluationResult(
                evaluation_id="test",
                request_id="test",
                agent_id="test",
                overall_score=3.0,  # Out of bounds
                passed=True,
                rubric_scores={}
            )

    def test_a2a_message_validation(self):
        """Test A2AMessage validation"""
        message = A2AMessage(
            message_id="msg_001",
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type="evaluation_request",
            content={"analysis": "test"},
            correlation_id="corr_001",
            priority=5
        )

        assert message.message_id == "msg_001"
        assert message.sender_id == "agent_1"
        assert message.message_type == "evaluation_request"
        assert message.priority == 5

        # Test priority bounds
        with pytest.raises(ValidationError):
            A2AMessage(
                message_id="msg_002",
                sender_id="agent_1",
                receiver_id="agent_2",
                message_type="evaluation_request",
                content={},
                priority=11  # Out of bounds
            )

        # Test message type validation
        with pytest.raises(ValidationError):
            A2AMessage(
                message_id="msg_003",
                sender_id="agent_1",
                receiver_id="agent_2",
                message_type="invalid_type",  # Not in Literal
                content={}
            )


class TestFinanceContracts:
    """Test finance contract validation"""

    def test_sec_filing_request_validation(self):
        """Test SECFilingRequest validation"""
        # Valid request
        request = SECFilingRequest(
            company_cik="0000320193",
            filing_type="10-K",
            period_end=datetime(2023, 12, 31)
        )

        assert request.company_cik == "0000000320193"  # Zero-padded
        assert request.filing_type == "10-K"
        assert request.include_attachments == False

        # Test CIK validation
        with pytest.raises(ValidationError):
            SECFilingRequest(
                company_cik="invalid_cik",  # Not numeric
                filing_type="10-K"
            )

    def test_financial_metric_data_validation(self):
        """Test FinancialMetricData validation"""
        metric = FinancialMetricData(
            metric_name="revenue",
            value=81800000000.0,
            unit="USD",
            period=datetime(2023, 6, 30),
            source_document="10-Q",
            footnote="As reported",
            is_estimated=False,
            confidence=0.95
        )

        assert metric.metric_name == "revenue"
        assert metric.value == 81800000000.0
        assert metric.unit == "USD"
        assert metric.confidence == 0.95

        # Test confidence bounds
        with pytest.raises(ValidationError):
            FinancialMetricData(
                metric_name="revenue",
                value=100.0,
                unit="USD",
                period=datetime.utcnow(),
                source_document="test",
                confidence=1.5  # Out of bounds
            )

    def test_company_financials_validation(self):
        """Test CompanyFinancials validation"""
        financials = CompanyFinancials(
            company_cik="0000320193",
            company_name="Apple Inc.",
            ticker="AAPL",
            fiscal_year_end="09-30",
            filings=[],
            metrics={},
            risk_factors=[],
            management_discussion="MD&A content",
            recent_events=[]
        )

        assert financials.company_cik == "0000320193"
        assert financials.company_name == "Apple Inc."
        assert financials.ticker == "AAPL"

        # Test get_latest_metric method
        financials.metrics["revenue"] = [
            FinancialMetricData(
                metric_name="revenue",
                value=100.0,
                unit="USD",
                period=datetime(2023, 6, 30),
                source_document="10-Q"
            ),
            FinancialMetricData(
                metric_name="revenue",
                value=90.0,
                unit="USD",
                period=datetime(2023, 3, 31),
                source_document="10-Q"
            )
        ]

        latest = financials.get_latest_metric("revenue")
        assert latest is not None
        assert latest.value == 100.0
        assert latest.period == datetime(2023, 6, 30)

    def test_market_data_request_validation(self):
        """Test MarketDataRequest validation"""
        # Valid request
        request = MarketDataRequest(
            ticker="AAPL",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            interval="1d",
            metrics=["close", "volume"]
        )

        assert request.ticker == "AAPL"
        assert request.interval == "1d"
        assert "close" in request.metrics

        # Test date validation
        with pytest.raises(ValidationError):
            MarketDataRequest(
                ticker="AAPL",
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1),  # End before start
                interval="1d",
                metrics=[]
            )

        # Test interval pattern
        with pytest.raises(ValidationError):
            MarketDataRequest(
                ticker="AAPL",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                interval="invalid",  # Not in pattern
                metrics=[]
            )

    def test_risk_assessment_validation(self):
        """Test RiskAssessment validation"""
        assessment = RiskAssessment(
            assessment_id="risk_001",
            company_cik="0000320193",
            risk_categories={"market": 0.7, "credit": 0.5},
            overall_risk_score=0.6,
            mitigations=["Hedging", "Diversification"],
            monitoring_indicators=["Debt ratio", "Liquidity"],
            next_review_date=datetime(2024, 6, 30)
        )

        assert assessment.assessment_id == "risk_001"
        assert assessment.overall_risk_score == 0.6
        assert len(assessment.mitigations) == 2

        # Test risk score bounds
        with pytest.raises(ValidationError):
            RiskAssessment(
                assessment_id="risk_002",
                company_cik="0000320193",
                risk_categories={},
                overall_risk_score=1.5,  # Out of bounds
                mitigations=[],
                monitoring_indicators=[]
            )


class TestJudgeContracts:
    """Test judge contract validation"""

    def test_judge_capabilities_validation(self):
        """Test JudgeCapabilities validation"""
        capabilities = JudgeCapabilities(
            agent_id="judge_001",
            supported_rubrics=[
                "factual_accuracy",
                "source_fidelity",
                "regulatory_compliance"
            ],
            max_concurrent_evaluations=10,
            evaluation_timeout=30,
            requires_grounding=True,
            compliance_level="strict"
        )

        assert capabilities.agent_id == "judge_001"
        assert len(capabilities.supported_rubrics) == 3
        assert capabilities.max_concurrent_evaluations == 10
        assert capabilities.evaluation_timeout == 30

    def test_judge_metrics_validation(self):
        """Test JudgeMetrics validation"""
        metrics = JudgeMetrics(
            evaluations_completed=100,
            average_score=1.75,
            false_positives=2,
            false_negatives=3,
            avg_processing_time=1.5,
            uptime=99.9
        )

        assert metrics.evaluations_completed == 100
        assert metrics.average_score == 1.75
        assert metrics.false_positives == 2
        assert metrics.uptime == 99.9

    def test_judge_configuration_validation(self):
        """Test JudgeConfiguration validation"""
        config = JudgeConfiguration(
            scoring_weights={
                "factual_accuracy": 0.25,
                "source_fidelity": 0.20,
                "regulatory_compliance": 0.15
            },
            pass_threshold=1.5,
            strict_mode=True,
            log_detailed=True,
            auto_calibrate=False
        )

        assert config.pass_threshold == 1.5
        assert config.strict_mode == True
        assert config.scoring_weights["factual_accuracy"] == 0.25

        # Test weight sum validation (if implemented)
        # Note: Pydantic doesn't automatically validate weight sums

        # Test pass threshold bounds
        with pytest.raises(ValidationError):
            JudgeConfiguration(
                scoring_weights={},
                pass_threshold=3.0,  # Out of bounds
                strict_mode=False,
                log_detailed=False,
                auto_calibrate=False
            )


class TestContractSerialization:
    """Test contract serialization/deserialization"""

    def test_evaluation_request_serialization(self):
        """Test EvaluationRequest serialization"""
        request = EvaluationRequest(
            analysis_id="test_analysis",
            agent_id="test_agent",
            analysis_content="Test content",
            source_documents=[],
            rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
        )

        # Serialize to dict
        data = request.dict()

        assert data["analysis_id"] == "test_analysis"
        assert data["agent_id"] == "test_agent"
        assert "analysis_content" in data

        # Deserialize from dict
        new_request = EvaluationRequest(**data)

        assert new_request.analysis_id == request.analysis_id
        assert new_request.agent_id == request.agent_id

    def test_evaluation_result_serialization(self):
        """Test EvaluationResult serialization"""
        result = EvaluationResult(
            evaluation_id="test_eval",
            request_id="test_request",
            agent_id="test_agent",
            overall_score=1.5,
            passed=True,
            rubric_scores={
                RubricCategory.FACTUAL_ACCURACY: RubricScore(
                    rubric=RubricCategory.FACTUAL_ACCURACY,
                    score=1.8,
                    passed=True,
                    feedback="Good",
                    evidence=[],
                    confidence=1.0
                )
            }
        )

        # Serialize to JSON
        json_str = result.json()

        assert "test_eval" in json_str
        assert "factual_accuracy" in json_str

        # Deserialize from JSON
        from pydantic import parse_raw_as
        new_result = parse_raw_as(EvaluationResult, json_str)

        assert new_result.evaluation_id == result.evaluation_id
        assert new_result.overall_score == result.overall_score

    def test_contract_with_datetime(self):
        """Test contracts with datetime fields"""
        from datetime import timezone

        # Create with timezone-aware datetime
        dt = datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        assessment = RiskAssessment(
            assessment_id="test",
            company_cik="0000320193",
            risk_categories={},
            overall_risk_score=0.5,
            mitigations=[],
            monitoring_indicators=[],
            next_review_date=dt
        )

        # Serialize
        data = assessment.dict()

        # Should contain ISO format datetime
        assert "2023-12-31T23:59:59" in str(data["next_review_date"])

    def test_contract_with_enums(self):
        """Test contracts with enum fields"""
        request = EvaluationRequest(
            analysis_id="test",
            agent_id="test",
            analysis_content="",
            source_documents=[],
            rubrics_to_evaluate=[
                RubricCategory.FACTUAL_ACCURACY,
                RubricCategory.SOURCE_FIDELITY
            ],
            scoring_scale=ScoringScale.ZERO_TO_TWO
        )

        # Enums should be converted to strings in dict
        data = request.dict()

        assert "factual_accuracy" in data["rubrics_to_evaluate"]
        assert data["scoring_scale"] == "0-2"

        # Should handle both enum and string input
        request2 = EvaluationRequest(
            analysis_id="test2",
            agent_id="test2",
            analysis_content="",
            source_documents=[],
            rubrics_to_evaluate=["factual_accuracy", "source_fidelity"],  # Strings
            scoring_scale="0-2"  # String
        )

        assert RubricCategory.FACTUAL_ACCURACY in request2.rubrics_to_evaluate
        assert request2.scoring_scale == ScoringScale.ZERO_TO_TWO


@pytest.mark.integration
class TestContractIntegration:
    """Integration tests for contracts"""

    def test_contract_compatibility(self):
        """Test that contracts work together"""
        # Create a complete workflow with contracts
        filing_request = SECFilingRequest(
            company_cik="0000320193",
            filing_type="10-K"
        )

        evaluation_request = EvaluationRequest(
            analysis_id="analysis_001",
            agent_id="agent_001",
            analysis_content="Based on SEC filing...",
            source_documents=[
                {
                    "document_type": filing_request.filing_type,
                    "content": "Filing content",
                    "cik": filing_request.company_cik,
                    "date": "2023-12-31"
                }
            ],
            rubrics_to_evaluate=[RubricCategory.FACTUAL_ACCURACY]
        )

        rubric_score = RubricScore(
            rubric=RubricCategory.FACTUAL_ACCURACY,
            score=1.8,
            passed=True,
            feedback="Accurate",
            evidence=["Matches SEC filing"],
            confidence=0.95
        )

        evaluation_result = EvaluationResult(
            evaluation_id="eval_001",
            request_id=evaluation_request.analysis_id,
            agent_id=evaluation_request.agent_id,
            overall_score=1.8,
            passed=True,
            rubric_scores={RubricCategory.FACTUAL_ACCURACY: rubric_score}
        )

        # Create A2A message with result
        a2a_message = A2AMessage(
            message_id="msg_001",
            sender_id="judge_agent",
            receiver_id=evaluation_request.agent_id,
            message_type="evaluation_response",
            content={"result": evaluation_result.dict()},
            correlation_id="req_001"
        )

        # Verify the chain works
        assert filing_request.company_cik == "0000000320193"
        assert evaluation_request.source_documents[0]["cik"] == filing_request.company_cik
        assert evaluation_result.request_id == evaluation_request.analysis_id
        assert a2a_message.content["result"]["evaluation_id"] == evaluation_result.evaluation_id

    def test_contract_validation_in_production_scenario(self):
        """Test contract validation in a production-like scenario"""
        # Simulate production data
        production_data = {
            "analysis_id": "prod_analysis_001",
            "agent_id": "prod_agent_001",
            "analysis_content": "Revenue was $81.8B with 2% growth. Debt increased but financial health remains strong.",
            "source_documents": [
                {
                    "document_type": "10-K",
                    "content": "SEC filing JSON data",
                    "cik": "0000320193",
                    "date": "2023-12-31",
                    "accession_number": "0000320193-23-000106"
                }
            ],
            "rubrics_to_evaluate": [
                "factual_accuracy",
                "source_fidelity",
                "regulatory_compliance",
                "financial_reasoning",
                "materiality_relevance",
                "risk_awareness"
            ],
            "scoring_scale": "0-2",
            "minimum_threshold": 1.5,
            "context": {
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "analysis_type": "quarterly"
            }
        }

        # Should validate successfully
        request = EvaluationRequest(**production_data)

        assert request.analysis_id == "prod_analysis_001"
        assert len(request.rubrics_to_evaluate) == 6
        assert request.scoring_scale == ScoringScale.ZERO_TO_TWO

        # Simulate evaluation result
        result_data = {
            "evaluation_id": "prod_eval_001",
            "request_id": request.analysis_id,
            "agent_id": request.agent_id,
            "overall_score": 1.75,
            "passed": True,
            "rubric_scores": {
                "factual_accuracy": {
                    "rubric": "factual_accuracy",
                    "score": 2.0,
                    "passed": True,
                    "feedback": "All facts accurate",
                    "evidence": ["Revenue matches: $81.8B"],
                    "confidence": 1.0
                },
                "financial_reasoning": {
                    "rubric": "financial_reasoning",
                    "score": 1.5,
                    "passed": True,
                    "feedback": "Sound reasoning",
                    "evidence": ["Proper ratio interpretation"],
                    "confidence": 0.9
                }
            },
            "recommendations": [
                "Include more specific risk analysis",
                "Add forward-looking statement disclaimer"
            ],
            "warnings": [
                "Check debt increase justification"
            ],
            "metadata": {
                "processing_time_ms": 245,
                "judge_version": "1.0.0"
            }
        }

        result = EvaluationResult(**result_data)

        assert result.overall_score == 1.75
        assert result.passed == True
        assert len(result.recommendations) == 2
        assert "factual_accuracy" in result.rubric_scores


class TestContractErrorCases:
    """Test contract error cases and edge cases"""

    def test_missing_required_fields(self):
        """Test missing required fields"""
        # EvaluationRequest missing required fields
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRequest(
                analysis_id="test",
                # Missing agent_id
                analysis_content=""
            )

        assert "agent_id" in str(exc_info.value)

    def test_invalid_field_types(self):
        """Test invalid field types"""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRequest(
                analysis_id=123,  # Should be string
                agent_id="test",
                analysis_content=""
            )

        assert "string" in str(exc_info.value).lower()

    def test_empty_strings_validation(self):
        """Test empty string validation"""
        # Empty analysis_id should fail
        with pytest.raises(ValidationError):
            EvaluationRequest(
                analysis_id="",  # Empty string
                agent_id="test",
                analysis_content=""
            )

    def test_list_bounds_validation(self):
        """Test list bounds validation"""
        # Empty rubrics list should be OK (defaults to all)
        request = EvaluationRequest(
            analysis_id="test",
            agent_id="test",
            analysis_content="",
            rubrics_to_evaluate=[]  # Empty list
        )

        assert request.rubrics_to_evaluate == []

    def test_nested_object_validation(self):
        """Test nested object validation"""
        # Invalid nested object in source_documents
        with pytest.raises(ValidationError):
            EvaluationRequest(
                analysis_id="test",
                agent_id="test",
                analysis_content="",
                source_documents=[
                    {
                        "document_type": "10-K",
                        # Missing required 'content' field
                        "cik": "0000320193"
                    }
                ]
            )

    def test_custom_validators(self):
        """Test custom validators"""
        # SECFilingRequest CIK validator
        with pytest.raises(ValidationError):
            SECFilingRequest(
                company_cik="123-456-789",  # Invalid format
                filing_type="10-K"
            )

        # MarketDataRequest date validator
        with pytest.raises(ValidationError):
            MarketDataRequest(
                ticker="AAPL",
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1),  # End before start
                interval="1d"
            )