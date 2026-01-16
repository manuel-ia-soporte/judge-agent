# infrastructure/a2a/a2a_client.py
import asyncio
import json
import uuid
from typing import Any, Callable, Dict, Optional
from datetime import datetime

from contracts.evaluation_contracts import A2AMessage


class A2AClient:
    """WebSocket-based A2A protocol client."""

    def __init__(self, url: str, client_id: str):
        self.url = url
        self.client_id = client_id
        self.websocket = None
        self.connected = False
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._listener_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to the A2A server."""
        try:
            import websockets
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            self._listener_task = asyncio.create_task(self._listen())
        except Exception as e:
            self.connected = False
            raise

    async def close(self) -> None:
        """Close the connection."""
        self.connected = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self.websocket:
            await self.websocket.close()

    async def _listen(self) -> None:
        """Listen for incoming messages."""
        try:
            while self.connected and self.websocket:
                data = await self.websocket.recv()
                message = json.loads(data)
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.connected = False

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle an incoming message."""
        correlation_id = message.get("correlation_id")

        # Check if this is a response to a pending request
        if correlation_id and correlation_id in self.pending_responses:
            future = self.pending_responses.pop(correlation_id)
            if not future.done():
                future.set_result(message)
            return

        # Check for registered handlers
        message_type = message.get("message_type")
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            await handler(message)

    async def send_message(
        self,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        expect_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Send a message to another agent."""
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected")

        correlation_id = str(uuid.uuid4()) if expect_response else None

        message = {
            "message_id": str(uuid.uuid4()),
            "sender_id": self.client_id,
            "receiver_id": receiver_id,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
        }

        await self.websocket.send(json.dumps(message))

        if expect_response and correlation_id:
            future: asyncio.Future = asyncio.Future()
            self.pending_responses[correlation_id] = future
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                self.pending_responses.pop(correlation_id, None)
                raise

        return None

    async def broadcast(
        self,
        message_type: str,
        content: Dict[str, Any]
    ) -> None:
        """Broadcast a message to all agents."""
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected")

        message = {
            "message_id": str(uuid.uuid4()),
            "sender_id": self.client_id,
            "receiver_id": "*",  # Broadcast indicator
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.websocket.send(json.dumps(message))

    def register_handler(
        self,
        message_type: str,
        handler: Callable
    ) -> None:
        """Register a handler for a message type."""
        self.message_handlers[message_type] = handler

    def unregister_handler(self, message_type: str) -> None:
        """Unregister a handler."""
        self.message_handlers.pop(message_type, None)
