# infrastructure/a2a/message_broker.py
from typing import Dict

from contracts.integration.messaging_contracts import A2AMessage


class MessageBroker:
    def __init__(self) -> None:
        self._subscribers: Dict[str, list] = {}

    def subscribe(self, message_type: str, handler) -> None:
        self._subscribers.setdefault(message_type, []).append(handler)

    async def publish(self, message: A2AMessage) -> None:
        for handler in self._subscribers.get(message.message_type, []):
            await handler(message)
