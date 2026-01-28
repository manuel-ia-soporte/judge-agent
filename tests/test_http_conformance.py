from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx


def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    required_fields = frozenset(
        [
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "defaultInputModes",
            "defaultOutputModes",
            "skills",
        ]
    )

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    url = card_data.get("url")
    if isinstance(url, str) and not (url.startswith("http://") or url.startswith("https://")):
        errors.append("Field 'url' must be an absolute URL starting with http:// or https://.")

    capabilities = card_data.get("capabilities")
    if capabilities is not None and not isinstance(capabilities, dict):
        errors.append("Field 'capabilities' must be an object.")

    for field in ["defaultInputModes", "defaultOutputModes"]:
        value = card_data.get(field)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"Field '{field}' must be an array of strings.")

    skills = card_data.get("skills")
    if skills is not None and not isinstance(skills, list):
        errors.append("Field 'skills' must be an array of AgentSkill objects.")

    return errors


def _assessment_request_payload() -> dict[str, Any]:
    request = {
        "participants": {"participant": "http://localhost:9999"},
        "config": {"allowNetwork": False, "maxQuestions": 1, "seed": 42},
    }
    return {
        "message": {
            "messageId": uuid4().hex,
            "contextId": uuid4().hex,
            "role": "ROLE_USER",
            "content": [{"text": json.dumps(request)}],
        },
        "configuration": {"acceptedOutputModes": ["text"]},
    }


def test_agent_card_structure(agent_url):
    response = httpx.get(f"{agent_url}/.well-known/agent-card.json", timeout=5)
    assert response.status_code == 200
    card_data = response.json()
    errors = validate_agent_card(card_data)
    assert not errors, "\n".join(errors)


def test_message_send_returns_task(agent_url):
    card_data = httpx.get(f"{agent_url}/.well-known/agent-card.json", timeout=5).json()
    base_url = str(card_data.get("url") or agent_url).rstrip("/")
    payload = _assessment_request_payload()

    response = httpx.post(f"{base_url}/v1/message:send", json=payload, timeout=20)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert "task" in data
    task = data["task"]
    assert isinstance(task, dict)
    assert "id" in task
    assert "status" in task and isinstance(task["status"], dict)
    assert "state" in task["status"]
    assert task["status"]["state"] == "TASK_STATE_REJECTED"


def test_message_stream_emits_events(agent_url):
    card_data = httpx.get(f"{agent_url}/.well-known/agent-card.json", timeout=5).json()
    base_url = str(card_data.get("url") or agent_url).rstrip("/")
    payload = _assessment_request_payload()

    events: list[dict[str, Any]] = []
    with httpx.Client(timeout=20) as client:
        with client.stream(
            "POST",
            f"{base_url}/v1/message:stream",
            json=payload,
            headers={"accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[len("data: ") :])
                if isinstance(data, dict):
                    events.append(data)
                if len(events) >= 3:
                    break

    assert events
    assert any(
        "task" in event or "statusUpdate" in event or "artifactUpdate" in event
        for event in events
    )
