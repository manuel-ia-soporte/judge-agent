import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .a2a_schemas import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    AnswerRequest,
    AnswerResponse,
    Artifact,
    CreateWebhookRequest,
    CreateWebhookResponse,
    Message,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    StreamResponse,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    WebhookConfig,
    WebhookEvent,
    new_artifact,
    new_data_part,
    new_message,
    new_text_part,
)
from .green_eval import run_assessment
from .leaderboard_client import LeaderboardClient, get_leaderboard_client
from .purple_agent import run_purple_agent
from .task_store import InMemoryTaskStore


app = FastAPI(title="finance-green-agent")
task_store = InMemoryTaskStore()

# In-memory webhook storage (task_id -> list of WebhookConfig)
webhook_store: dict[str, list[WebhookConfig]] = {}


def _agent_url() -> str:
    return os.environ.get("FINANCE_GREEN_URL", "http://127.0.0.1:9009").rstrip("/")


def _build_agent_card() -> AgentCard:
    base_url = _agent_url()
    return AgentCard(
        protocol_version="1.0",
        name=os.environ.get("FINANCE_GREEN_NAME", "finance-green-agent"),
        description="Offline-default finance green agent that evaluates participant agents.",
        url=base_url,
        preferred_transport="HTTP+JSON",
        additional_interfaces=[AgentInterface(url=base_url, transport="JSONRPC")],
        version=os.environ.get("FINANCE_GREEN_VERSION", "1.0.0"),
        provider=AgentProvider(url=base_url, organization="finance-green-agent"),
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="finance-green-eval",
                name="Finance Agent Assessment",
                description="Evaluates finance agents against public.csv with rubric scoring.",
                tags=["finance", "evaluation", "a2a"],
                examples=["Run an offline AgentBeats evaluation."],
                input_modes=["text"],
                output_modes=["text"],
            ),
            AgentSkill(
                id="finance-purple-answer",
                name="SEC/EDGAR Question Answering",
                description="Answers SEC/EDGAR finance questions using offline cached data.",
                tags=["finance", "sec", "edgar", "qa"],
                examples=["What was Apple's revenue in Q4 2024?"],
                input_modes=["text"],
                output_modes=["text"],
            ),
        ],
        supports_authenticated_extended_card=False,
    )


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True, exclude_none=True)
    return model


def _json_response(model: Any) -> JSONResponse:
    return JSONResponse(content=_dump_model(model))


def _extract_message_text(message) -> str:
    parts = []
    for part in message.content:
        if part.text is not None:
            parts.append(part.text)
        elif part.data is not None:
            try:
                parts.append(json.dumps(part.data.data, ensure_ascii=False))
            except TypeError:
                parts.append(str(part.data.data))
    return "\n".join(part for part in parts if part).strip()


def _summary_text(result: dict[str, Any]) -> str:
    lines = [
        "Evaluation complete.",
        f"Winner: {result.get('winner')}",
        f"Questions: {result.get('max_questions')}",
    ]
    participants = result.get("participants", {})
    for role, data in participants.items():
        summary = data.get("summary", {})
        lines.append(
            f"{role}: avg={summary.get('average_score')}, passed={summary.get('passed')}/{summary.get('total')}, errors={summary.get('errors')}"
        )
    return "\n".join(lines)


def _role_to_jsonrpc(role: Role) -> str:
    if role == Role.agent:
        return "agent"
    if role == Role.user:
        return "user"
    return "user"


def _part_to_jsonrpc(part: Part) -> dict[str, Any]:
    if part.text is not None:
        payload = {"kind": "text", "text": part.text}
    elif part.data is not None:
        payload = {"kind": "data", "data": part.data.data}
    elif part.file is not None:
        file_payload: dict[str, Any] = {}
        if part.file.file_with_uri:
            file_payload["uri"] = part.file.file_with_uri
        if part.file.file_with_bytes:
            file_payload["bytes"] = part.file.file_with_bytes
        if part.file.mime_type:
            file_payload["mimeType"] = part.file.mime_type
        if part.file.name:
            file_payload["name"] = part.file.name
        payload = {"kind": "file", "file": file_payload}
    else:
        payload = {"kind": "text", "text": ""}
    if part.metadata:
        payload["metadata"] = part.metadata
    return payload


def _message_to_jsonrpc(message: Message) -> dict[str, Any]:
    payload = {
        "kind": "message",
        "messageId": message.message_id,
        "role": _role_to_jsonrpc(message.role),
        "parts": [_part_to_jsonrpc(part) for part in message.content],
    }
    if message.context_id:
        payload["contextId"] = message.context_id
    if message.task_id:
        payload["taskId"] = message.task_id
    if message.metadata:
        payload["metadata"] = message.metadata
    if message.extensions:
        payload["extensions"] = message.extensions
    return payload


def _artifact_to_jsonrpc(artifact: Artifact) -> dict[str, Any]:
    payload = {
        "artifactId": artifact.artifact_id,
        "parts": [_part_to_jsonrpc(part) for part in artifact.parts],
    }
    if artifact.name:
        payload["name"] = artifact.name
    if artifact.description:
        payload["description"] = artifact.description
    if artifact.metadata:
        payload["metadata"] = artifact.metadata
    if artifact.extensions:
        payload["extensions"] = artifact.extensions
    return payload


def _state_to_jsonrpc(state: TaskState) -> str:
    return {
        TaskState.submitted: "submitted",
        TaskState.working: "working",
        TaskState.completed: "completed",
        TaskState.failed: "failed",
        TaskState.cancelled: "canceled",
        TaskState.input_required: "input-required",
        TaskState.auth_required: "auth-required",
        TaskState.rejected: "rejected",
        TaskState.unspecified: "unknown",
    }.get(state, "unknown")


def _task_to_jsonrpc(task) -> dict[str, Any]:
    status_payload = {"state": _state_to_jsonrpc(task.status.state)}
    if task.status.message:
        status_payload["message"] = _message_to_jsonrpc(task.status.message)

    payload = {
        "kind": "task",
        "id": task.id,
        "status": status_payload,
    }
    if task.context_id:
        payload["contextId"] = task.context_id
    if task.artifacts:
        payload["artifacts"] = [_artifact_to_jsonrpc(a) for a in task.artifacts]
    if task.history:
        payload["history"] = [_message_to_jsonrpc(m) for m in task.history]
    if task.metadata:
        payload["metadata"] = task.metadata
    return payload


def _jsonrpc_response(result: dict[str, Any], request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _extract_jsonrpc_message_text(message: dict[str, Any]) -> str:
    parts = []
    for part in message.get("parts", []) or []:
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text":
            parts.append(part.get("text", ""))
        elif part.get("kind") == "data":
            data_payload = part.get("data")
            try:
                parts.append(json.dumps(data_payload, ensure_ascii=False))
            except TypeError:
                parts.append(str(data_payload))
    return "\n".join(part for part in parts if part).strip()


async def _handle_jsonrpc_send(
    params: dict[str, Any], request_id: Any
) -> Any:
    message_payload = params.get("message", {}) if isinstance(params, dict) else {}
    context_id = message_payload.get("contextId") or f"context-{request_id or uuid4().hex}"
    request_text = _extract_jsonrpc_message_text(message_payload)
    incoming = Message(
        message_id=message_payload.get("messageId") or uuid4().hex,
        role=Role.user,
        content=[new_text_part(request_text or "")],
        context_id=context_id,
        task_id=message_payload.get("taskId"),
    )
    task = task_store.create_task(context_id=context_id, history=[incoming])
    working_message = new_message(
        role=Role.agent,
        parts=[new_text_part("Starting assessment.")],
        context_id=task.context_id,
        task_id=task.id,
    )
    task_store.update_status(task.id, TaskState.working, working_message)

    if not request_text:
        error_message = new_message(
            role=Role.agent,
            parts=[new_text_part("Missing EvalRequest payload.")],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.rejected, error_message)
        return task

    try:
        result, config = await run_assessment(request_text)
    except ValueError as exc:
        error_message = new_message(
            role=Role.agent,
            parts=[new_text_part(str(exc))],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.rejected, error_message)
        return task
    except Exception as exc:  # noqa: BLE001 - return failure to client
        error_message = new_message(
            role=Role.agent,
            parts=[new_text_part(f"Evaluation failed: {exc}")],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.failed, error_message)
        return task

    summary_text = _summary_text(result)
    artifact = new_artifact(
        name="EvaluationResult",
        parts=[new_text_part(summary_text), new_data_part(result)],
        metadata={"config": config.__dict__},
    )
    task_store.add_artifact(task.id, artifact)
    summary_message = new_message(
        role=Role.agent,
        parts=[new_text_part(summary_text)],
        context_id=task.context_id,
        task_id=task.id,
    )
    task_store.update_status(task.id, TaskState.completed, summary_message)
    return task


def _encode_sse(event: StreamResponse) -> str:
    payload = event.model_dump(by_alias=True, exclude_none=True)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/.well-known/agent-card.json")
async def agent_card() -> JSONResponse:
    return _json_response(_build_agent_card())


@app.get("/v1/card")
async def agent_card_extended() -> JSONResponse:
    return _json_response(_build_agent_card())


@app.get("/manifest")
async def manifest_alias() -> JSONResponse:
    return _json_response(_build_agent_card())


@app.post("/", response_model=None)
async def jsonrpc_endpoint(request: Request):
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        )
    method = payload.get("method")
    request_id = payload.get("id")
    params = payload.get("params", {})

    if method == "message/send":
        task = await _handle_jsonrpc_send(params, request_id)
        return JSONResponse(
            content=_jsonrpc_response(_task_to_jsonrpc(task), request_id)
        )

    if method == "message/stream":
        async def event_generator() -> AsyncGenerator[str, None]:
            task = await _handle_jsonrpc_send(params, request_id)
            response = _jsonrpc_response(_task_to_jsonrpc(task), request_id)
            yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return JSONResponse(
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found"},
        }
    )


@app.post("/v1/message:send")
async def message_send(payload: dict[str, Any]) -> JSONResponse:
    try:
        request = SendMessageRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    incoming = request.message
    if not incoming.context_id:
        incoming.context_id = f"context-{incoming.message_id}"

    task = task_store.create_task(context_id=incoming.context_id, history=[incoming])
    working_message = new_message(
        role=Role.agent,
        parts=[new_text_part("Starting assessment.")],
        context_id=task.context_id,
        task_id=task.id,
    )
    task_store.update_status(task.id, TaskState.working, working_message)

    request_text = _extract_message_text(incoming)
    if not request_text:
        error_message = new_message(
            role=Role.agent,
            parts=[new_text_part("Missing EvalRequest payload.")],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.rejected, error_message)
        return _json_response(SendMessageResponse(task=task))

    try:
        result, config = await run_assessment(request_text)
    except ValueError as exc:
        error_message = new_message(
            role=Role.agent,
            parts=[new_text_part(str(exc))],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.rejected, error_message)
        return _json_response(SendMessageResponse(task=task))
    except Exception as exc:  # noqa: BLE001 - return failure to client
        error_message = new_message(
            role=Role.agent,
            parts=[new_text_part(f"Evaluation failed: {exc}")],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.failed, error_message)
        return _json_response(SendMessageResponse(task=task))

    summary_text = _summary_text(result)
    summary_message = new_message(
        role=Role.agent,
        parts=[new_text_part(summary_text)],
        context_id=task.context_id,
        task_id=task.id,
    )
    artifact = new_artifact(
        name="EvaluationResult",
        parts=[new_text_part(summary_text), new_data_part(result)],
        metadata={"config": config.__dict__},
    )
    task_store.add_artifact(task.id, artifact)
    task_store.update_status(task.id, TaskState.completed, summary_message)

    return _json_response(SendMessageResponse(task=task))


@app.post("/v1/message:stream")
async def message_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    try:
        message_request = SendMessageRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    incoming = message_request.message
    if not incoming.context_id:
        incoming.context_id = f"context-{incoming.message_id}"

    task = task_store.create_task(context_id=incoming.context_id, history=[incoming])

    async def event_generator() -> AsyncGenerator[str, None]:
        yield _encode_sse(StreamResponse(task=task))

        working_message = new_message(
            role=Role.agent,
            parts=[new_text_part("Starting assessment.")],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.working, working_message)
        status_update = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=task.status,
            final=False,
        )
        yield _encode_sse(StreamResponse(status_update=status_update))

        request_text = _extract_message_text(incoming)
        if not request_text:
            error_message = new_message(
                role=Role.agent,
                parts=[new_text_part("Missing EvalRequest payload.")],
                context_id=task.context_id,
                task_id=task.id,
            )
            task_store.update_status(task.id, TaskState.rejected, error_message)
            yield _encode_sse(
                StreamResponse(
                    status_update=TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    )
                )
            )
            return

        try:
            result, config = await run_assessment(request_text)
        except ValueError as exc:
            error_message = new_message(
                role=Role.agent,
                parts=[new_text_part(str(exc))],
                context_id=task.context_id,
                task_id=task.id,
            )
            task_store.update_status(task.id, TaskState.rejected, error_message)
            yield _encode_sse(
                StreamResponse(
                    status_update=TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    )
                )
            )
            return
        except Exception as exc:  # noqa: BLE001 - return failure to client
            error_message = new_message(
                role=Role.agent,
                parts=[new_text_part(f"Evaluation failed: {exc}")],
                context_id=task.context_id,
                task_id=task.id,
            )
            task_store.update_status(task.id, TaskState.failed, error_message)
            yield _encode_sse(
                StreamResponse(
                    status_update=TaskStatusUpdateEvent(
                        task_id=task.id,
                        context_id=task.context_id,
                        status=task.status,
                        final=True,
                    )
                )
            )
            return

        summary_text = _summary_text(result)
        artifact = new_artifact(
            name="EvaluationResult",
            parts=[new_text_part(summary_text), new_data_part(result)],
            metadata={"config": config.__dict__},
        )
        task_store.add_artifact(task.id, artifact)
        yield _encode_sse(
            StreamResponse(
                artifact_update=TaskArtifactUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    artifact=artifact,
                    append=False,
                    last_chunk=True,
                )
            )
        )

        summary_message = new_message(
            role=Role.agent,
            parts=[new_text_part(summary_text)],
            context_id=task.context_id,
            task_id=task.id,
        )
        task_store.update_status(task.id, TaskState.completed, summary_message)
        yield _encode_sse(
            StreamResponse(
                status_update=TaskStatusUpdateEvent(
                    task_id=task.id,
                    context_id=task.context_id,
                    status=task.status,
                    final=True,
                )
            )
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/message")
async def message_alias(payload: dict[str, Any]) -> JSONResponse:
    return await message_send(payload)


@app.get("/v1/tasks/{task_id}")
async def get_task(task_id: str, historyLength: int | None = None) -> JSONResponse:
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if historyLength is not None and historyLength >= 0:
        task_copy = task.model_copy(deep=True)
        task_copy.history = task_copy.history[-historyLength:]
        return _json_response(task_copy)

    return _json_response(task)


@app.post("/v1/tasks/{task_id}:cancel")
async def cancel_task(task_id: str) -> JSONResponse:
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    cancel_message = new_message(
        role=Role.agent,
        parts=[new_text_part("Task cancelled by client.")],
        context_id=task.context_id,
        task_id=task.id,
    )
    task_store.update_status(task.id, TaskState.cancelled, cancel_message)
    return _json_response(task)


@app.get("/v1/tasks/{task_id}:subscribe")
async def subscribe_task(task_id: str) -> StreamingResponse:
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        yield _encode_sse(StreamResponse(task=task))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/tasks/{task_id}/pushNotificationConfigs")
async def create_push_config(task_id: str, payload: dict[str, Any]) -> JSONResponse:
    """Create a webhook/push notification configuration for a task."""
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        request = CreateWebhookRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    config = WebhookConfig(
        url=request.url,
        token=request.token,
        events=request.events,
        authentication=request.authentication,
    )

    if task_id not in webhook_store:
        webhook_store[task_id] = []
    webhook_store[task_id].append(config)

    return _json_response(CreateWebhookResponse(config=config))


@app.get("/v1/tasks/{task_id}/pushNotificationConfigs")
async def list_push_configs(task_id: str) -> JSONResponse:
    """List all webhook configurations for a task."""
    configs = webhook_store.get(task_id, [])
    return JSONResponse(
        content={
            "configs": [_dump_model(c) for c in configs],
            "nextPageToken": "",
        }
    )


@app.get("/v1/tasks/{task_id}/pushNotificationConfigs/{config_id}")
async def get_push_config(task_id: str, config_id: str) -> JSONResponse:
    """Get a specific webhook configuration."""
    configs = webhook_store.get(task_id, [])
    for config in configs:
        if config.id == config_id:
            return _json_response(config)
    raise HTTPException(status_code=404, detail="Push notification config not found")


@app.delete("/v1/tasks/{task_id}/pushNotificationConfigs/{config_id}")
async def delete_push_config(task_id: str, config_id: str) -> JSONResponse:
    """Delete a webhook configuration."""
    if task_id in webhook_store:
        webhook_store[task_id] = [
            c for c in webhook_store[task_id] if c.id != config_id
        ]
    return JSONResponse(content={})


# =============================================================================
# Purple Agent Endpoints - Answer SEC/EDGAR questions
# =============================================================================


@app.post("/v1/answer")
async def answer_question(payload: dict[str, Any]) -> JSONResponse:
    """
    Purple agent endpoint: Answer a SEC/EDGAR question using offline tools.
    
    Request body:
    {
        "question": "What was Apple's revenue in Q4 2024?",
        "session_id": "optional-session-id",
        "config": {}
    }
    """
    try:
        request = AnswerRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    result = await run_purple_agent(request.question, request.session_id)

    if result.get("error"):
        return _json_response(
            AnswerResponse(
                answer="",
                error=result["error"],
                metadata=result.get("metadata", {}),
            )
        )

    # Extract sources from the answer if present
    answer_text = result.get("answer", "")
    sources = []
    try:
        import re
        match = re.search(r'(\{"sources".*\})', answer_text, re.DOTALL)
        if match:
            import json as json_mod
            sources_data = json_mod.loads(match.group(1))
            sources = sources_data.get("sources", [])
    except Exception:
        pass

    return _json_response(
        AnswerResponse(
            answer=answer_text,
            sources=sources,
            metadata=result.get("metadata", {}),
        )
    )


# =============================================================================
# Webhook / Leaderboard Integration Endpoints
# =============================================================================


async def _trigger_webhooks(task_id: str, event_type: str, data: dict[str, Any]) -> None:
    """Trigger all registered webhooks for a task."""
    configs = webhook_store.get(task_id, [])
    task = task_store.get_task(task_id)
    
    for config in configs:
        if event_type not in config.events:
            continue
        
        event = WebhookEvent(
            event_type=event_type,
            task_id=task_id,
            context_id=task.context_id if task else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        
        client = LeaderboardClient(
            webhook_url=config.url,
            webhook_token=config.token,
        )
        
        try:
            await client.notify_evaluation_complete(
                task_id=task_id,
                evaluation_result=data,
                callback_url=config.url,
            )
        except Exception:
            pass  # Log but don't fail


@app.post("/v1/webhook/leaderboard")
async def send_to_leaderboard(payload: dict[str, Any]) -> JSONResponse:
    """
    Manually trigger sending evaluation results to the leaderboard.
    
    Request body:
    {
        "task_id": "task-id-with-results",
        "event_type": "evaluation-complete"
    }
    """
    task_id = payload.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Extract evaluation result from task artifacts
    evaluation_result = {}
    for artifact in task.artifacts:
        for part in artifact.parts:
            if part.data is not None:
                evaluation_result = part.data.data
                break

    if not evaluation_result:
        raise HTTPException(status_code=400, detail="No evaluation result found in task")

    client = get_leaderboard_client()
    result = await client.send_results(
        evaluation_result=evaluation_result,
        event_type=payload.get("event_type", "evaluation-complete"),
    )

    return JSONResponse(content=result)


@app.post("/v1/webhook/test")
async def test_webhook(payload: dict[str, Any]) -> JSONResponse:
    """Test webhook connectivity by sending a test event."""
    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    client = LeaderboardClient(
        webhook_url=url,
        webhook_token=payload.get("token"),
    )

    result = await client.notify_evaluation_complete(
        task_id="test-task-id",
        evaluation_result={"test": True, "message": "Webhook test from finance-green-agent"},
        callback_url=url,
    )

    return JSONResponse(content=result)


def main():
    parser = argparse.ArgumentParser(description="Run the Finance Green Agent server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument(
        "--card-url",
        dest="card_url",
        default=None,
        help="External URL to advertise in the agent card (e.g. http://localhost:9009)",
    )
    args = parser.parse_args()

    if args.card_url:
        os.environ["FINANCE_GREEN_URL"] = str(args.card_url).rstrip("/")
    elif "FINANCE_GREEN_URL" not in os.environ:
        if args.host in {"0.0.0.0", "::"}:
            os.environ["FINANCE_GREEN_URL"] = f"http://localhost:{args.port}"
        else:
            os.environ["FINANCE_GREEN_URL"] = f"http://{args.host}:{args.port}"

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
