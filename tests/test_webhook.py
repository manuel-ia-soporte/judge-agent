"""Tests for webhook functionality."""

import pytest

from finance_green_agent.a2a_schemas import (
    CreateWebhookRequest,
    CreateWebhookResponse,
    WebhookConfig,
    WebhookEvent,
)


def test_webhook_config_creation():
    """Test WebhookConfig model creation with defaults."""
    config = WebhookConfig(url="https://example.com/webhook")
    assert config.url == "https://example.com/webhook"
    assert config.token is None
    assert "task.completed" in config.events
    assert "task.failed" in config.events
    assert config.id is not None  # auto-generated


def test_webhook_config_with_token():
    """Test WebhookConfig with authentication token."""
    config = WebhookConfig(
        url="https://example.com/webhook",
        token="secret-token-123",
        events=["task.completed"],
    )
    assert config.token == "secret-token-123"
    assert config.events == ["task.completed"]


def test_create_webhook_request_validation():
    """Test CreateWebhookRequest validation."""
    request = CreateWebhookRequest(url="https://example.com/webhook")
    assert request.url == "https://example.com/webhook"
    assert request.token is None


def test_create_webhook_request_with_all_fields():
    """Test CreateWebhookRequest with all fields."""
    request = CreateWebhookRequest(
        url="https://example.com/webhook",
        token="my-token",
        events=["task.completed", "evaluation.complete"],
    )
    assert request.token == "my-token"
    assert len(request.events) == 2


def test_create_webhook_response():
    """Test CreateWebhookResponse contains config."""
    config = WebhookConfig(url="https://example.com/webhook")
    response = CreateWebhookResponse(config=config)
    assert response.config.url == "https://example.com/webhook"


def test_webhook_event_creation():
    """Test WebhookEvent model creation."""
    event = WebhookEvent(
        event_type="task.completed",
        task_id="task-123",
        context_id="context-456",
        timestamp="2025-01-31T12:00:00Z",
        data={"winner": "participant", "score": 0.95},
    )
    assert event.event_type == "task.completed"
    assert event.task_id == "task-123"
    assert event.data["winner"] == "participant"


def test_webhook_event_serialization():
    """Test WebhookEvent serialization with camelCase aliases."""
    event = WebhookEvent(
        event_type="evaluation.complete",
        task_id="task-123",
        data={"results": []},
    )
    payload = event.model_dump(by_alias=True, exclude_none=True)
    assert payload["eventType"] == "evaluation.complete"
    assert payload["taskId"] == "task-123"


def test_webhook_config_serialization():
    """Test WebhookConfig serialization."""
    config = WebhookConfig(
        url="https://example.com/webhook",
        events=["task.completed"],
    )
    payload = config.model_dump(by_alias=True, exclude_none=True)
    assert "url" in payload
    assert "events" in payload
    assert "id" in payload
