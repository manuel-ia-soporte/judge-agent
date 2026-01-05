# application/interfaces/a2a_interface.py
from typing import Protocol
from contracts.integration.messaging_contracts import A2AMessage


class A2AClientProtocol(Protocol):
    async def send(self, message: A2AMessage) -> None:
        ...

    async def receive(self) -> A2AMessage:
        ...
