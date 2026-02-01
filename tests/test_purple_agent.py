"""Tests for purple agent endpoint."""

import pytest

from finance_green_agent.a2a_schemas import AnswerRequest, AnswerResponse


def test_answer_request_validation():
    """Test AnswerRequest model validation."""
    request = AnswerRequest(question="What is Apple's revenue?")
    assert request.question == "What is Apple's revenue?"
    assert request.session_id is None
    assert request.config == {}


def test_answer_request_with_session():
    """Test AnswerRequest with session_id."""
    request = AnswerRequest(
        question="What is Apple's revenue?",
        session_id="test-session-123",
        config={"max_turns": 10},
    )
    assert request.session_id == "test-session-123"
    assert request.config == {"max_turns": 10}


def test_answer_response_success():
    """Test AnswerResponse for successful answer."""
    response = AnswerResponse(
        answer="Apple's revenue was $100B.",
        sources=[{"id": "sec-1", "name": "10-K Filing"}],
        metadata={"turns": 3},
    )
    assert response.answer == "Apple's revenue was $100B."
    assert len(response.sources) == 1
    assert response.error is None


def test_answer_response_error():
    """Test AnswerResponse for error case."""
    response = AnswerResponse(
        answer="",
        error="Model not available",
        metadata={},
    )
    assert response.answer == ""
    assert response.error == "Model not available"


def test_answer_response_serialization():
    """Test AnswerResponse serialization with camelCase."""
    response = AnswerResponse(
        answer="Test answer",
        sources=[],
        metadata={"session_id": "test"},
    )
    payload = response.model_dump(by_alias=True, exclude_none=True)
    assert "answer" in payload
    assert "sources" in payload
    assert "metadata" in payload
