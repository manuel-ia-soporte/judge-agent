from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class A2ABaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=_to_camel,
        extra="ignore",
    )


class Role(str, Enum):
    unspecified = "ROLE_UNSPECIFIED"
    user = "ROLE_USER"
    agent = "ROLE_AGENT"


class TaskState(str, Enum):
    unspecified = "TASK_STATE_UNSPECIFIED"
    submitted = "TASK_STATE_SUBMITTED"
    working = "TASK_STATE_WORKING"
    completed = "TASK_STATE_COMPLETED"
    failed = "TASK_STATE_FAILED"
    cancelled = "TASK_STATE_CANCELLED"
    input_required = "TASK_STATE_INPUT_REQUIRED"
    rejected = "TASK_STATE_REJECTED"
    auth_required = "TASK_STATE_AUTH_REQUIRED"


class PushNotificationAuthenticationInfo(A2ABaseModel):
    schemes: list[str] = Field(default_factory=list)
    credentials: str | None = None


class PushNotificationConfig(A2ABaseModel):
    id: str | None = None
    url: str | None = None
    token: str | None = None
    authentication: PushNotificationAuthenticationInfo | None = None


class SendMessageConfiguration(A2ABaseModel):
    accepted_output_modes: list[str] = Field(default_factory=list)
    push_notification: PushNotificationConfig | None = None
    history_length: int | None = None
    blocking: bool | None = None


class FilePart(A2ABaseModel):
    file_with_uri: str | None = None
    file_with_bytes: str | None = None
    mime_type: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _check_file_variant(self) -> "FilePart":
        if bool(self.file_with_uri) == bool(self.file_with_bytes):
            raise ValueError(
                "File part must set exactly one of file_with_uri or file_with_bytes"
            )
        return self


class DataPart(A2ABaseModel):
    data: dict[str, Any]


class Part(A2ABaseModel):
    text: str | None = None
    file: FilePart | None = None
    data: DataPart | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_part_variant(self) -> "Part":
        variants = [self.text is not None, self.file is not None, self.data is not None]
        if sum(variants) != 1:
            raise ValueError("Part must include exactly one of text, file, or data")
        return self


class Message(A2ABaseModel):
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    context_id: str | None = None
    task_id: str | None = None
    role: Role
    content: list[Part]
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class TaskStatus(A2ABaseModel):
    state: TaskState
    message: Message | None = None
    timestamp: dict[str, Any] | None = None


class Artifact(A2ABaseModel):
    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    metadata: dict[str, Any] | None = None
    extensions: list[str] | None = None


class Task(A2ABaseModel):
    id: str
    context_id: str | None = None
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class TaskStatusUpdateEvent(A2ABaseModel):
    task_id: str
    context_id: str | None = None
    status: TaskStatus
    final: bool | None = None
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(A2ABaseModel):
    task_id: str
    context_id: str | None = None
    artifact: Artifact
    append: bool | None = None
    last_chunk: bool | None = None
    metadata: dict[str, Any] | None = None


class SendMessageRequest(A2ABaseModel):
    message: Message
    configuration: SendMessageConfiguration | None = None
    metadata: dict[str, Any] | None = None


class SendMessageResponse(A2ABaseModel):
    task: Task | None = None
    message: Message | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> "SendMessageResponse":
        if (self.task is None) == (self.message is None):
            raise ValueError("SendMessageResponse must contain exactly one of task or message")
        return self


class StreamResponse(A2ABaseModel):
    task: Task | None = None
    message: Message | None = None
    status_update: TaskStatusUpdateEvent | None = None
    artifact_update: TaskArtifactUpdateEvent | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> "StreamResponse":
        variants = [
            self.task is not None,
            self.message is not None,
            self.status_update is not None,
            self.artifact_update is not None,
        ]
        if sum(variants) != 1:
            raise ValueError("StreamResponse must contain exactly one payload variant")
        return self


class AgentProvider(A2ABaseModel):
    url: str
    organization: str


class AgentExtension(A2ABaseModel):
    uri: str
    description: str | None = None
    required: bool | None = None
    params: dict[str, Any] | None = None


class AgentCapabilities(A2ABaseModel):
    streaming: bool = False
    push_notifications: bool = False
    extensions: list[AgentExtension] = Field(default_factory=list)


class AgentSkill(A2ABaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    input_modes: list[str] | None = None
    output_modes: list[str] | None = None
    security: list[dict[str, list[str]]] | None = None


class AgentInterface(A2ABaseModel):
    url: str
    transport: str


class AgentCard(A2ABaseModel):
    protocol_version: str | None = None
    name: str
    description: str
    url: str
    preferred_transport: str | None = None
    additional_interfaces: list[AgentInterface] = Field(default_factory=list)
    provider: AgentProvider | None = None
    version: str
    documentation_url: str | None = None
    capabilities: AgentCapabilities
    security_schemes: dict[str, Any] | None = None
    security: list[dict[str, list[str]]] | None = None
    default_input_modes: list[str]
    default_output_modes: list[str]
    skills: list[AgentSkill] = Field(default_factory=list)
    supports_authenticated_extended_card: bool = False
    signatures: list[dict[str, Any]] | None = None
    icon_url: str | None = None


def new_text_part(text: str) -> Part:
    return Part(text=text)


def new_data_part(data: dict[str, Any]) -> Part:
    return Part(data=DataPart(data=data))


def new_message(
    *,
    role: Role,
    parts: list[Part],
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    return Message(role=role, content=parts, context_id=context_id, task_id=task_id)


def new_artifact(
    *,
    name: str,
    parts: list[Part],
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    return Artifact(name=name, description=description, parts=parts, metadata=metadata)


# Webhook / Push Notification schemas for leaderboard integration

class WebhookConfig(A2ABaseModel):
    """Configuration for webhook callbacks."""
    id: str = Field(default_factory=lambda: uuid4().hex)
    url: str
    token: str | None = None
    events: list[str] = Field(default_factory=lambda: ["task.completed", "task.failed"])
    authentication: PushNotificationAuthenticationInfo | None = None


class WebhookEvent(A2ABaseModel):
    """Event payload sent to webhook endpoints."""
    event_type: str
    task_id: str
    context_id: str | None = None
    timestamp: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class CreateWebhookRequest(A2ABaseModel):
    """Request to create a webhook configuration."""
    url: str
    token: str | None = None
    events: list[str] = Field(default_factory=lambda: ["task.completed", "task.failed"])
    authentication: PushNotificationAuthenticationInfo | None = None


class CreateWebhookResponse(A2ABaseModel):
    """Response after creating a webhook configuration."""
    config: WebhookConfig


class AnswerRequest(A2ABaseModel):
    """Request for the purple agent to answer a question."""
    question: str
    session_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(A2ABaseModel):
    """Response from the purple agent with the answer."""
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
