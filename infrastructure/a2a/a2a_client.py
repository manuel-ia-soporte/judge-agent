# infrastructure/a2a/a2a_client.py
import asyncio
from application.interfaces.a2a_interface import A2AClientProtocol
from contracts.integration.messaging_contracts import A2AMessage


class A2AClient(A2AClientProtocol):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[A2AMessage] = asyncio.Queue()

    async def send(self, message: A2AMessage) -> None:
        await self._queue.put(message)

    async def receive(self) -> A2AMessage:
        return await self._queue.get()
