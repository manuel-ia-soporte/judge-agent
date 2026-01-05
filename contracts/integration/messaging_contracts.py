# contracts/integration/messaging_contracts.py
from typing import Any, Dict, Literal
from pydantic import BaseModel


MessageType = Literal[
    "evaluation_request",
    "evaluation_response",
    "agent_query",
]


class A2AMessage(BaseModel):
    message_type: MessageType
    sender: str
    payload: Dict[str, Any]
