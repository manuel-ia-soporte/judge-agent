# infrastructure/a2a/a2a_client.py
import asyncio
import json
import uuid
from typing import Any, Callable, Dict, Optional, List, Set
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
        self._known_agents: Set[str] = set()

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
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)

    async def send_message(
        self,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        expect_response: bool = True,
        timeout: float = 30.0,
        correlation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a message to another agent."""
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected")

        corr_id = correlation_id or (str(uuid.uuid4()) if expect_response else None)

        message = {
            "message_id": str(uuid.uuid4()),
            "sender_id": self.client_id,
            "receiver_id": receiver_id,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": corr_id,
        }

        await self.websocket.send(json.dumps(message))

        if expect_response and corr_id:
            future: asyncio.Future
            existing_future = self.pending_responses.get(corr_id)
            if existing_future is not None:
                future = existing_future
            else:
                future = asyncio.Future()
                self.pending_responses[corr_id] = future
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
                self.pending_responses.pop(corr_id, None)
                return result
            except asyncio.TimeoutError:
                self.pending_responses.pop(corr_id, None)
                raise TimeoutError("Response timed out")

        return None

    async def broadcast(
        self,
        message_type: str,
        content: Dict[str, Any],
        agent_filter: Optional[Callable[[str], bool]] = None,
    ) -> List[Any]:
        """Broadcast a message to known agents using unicast sends."""
        responses = []
        for agent_id in self._get_available_agents():
            if agent_id == self.client_id:
                continue
            if agent_filter and not agent_filter(agent_id):
                continue
            response = await self.send_message(
                receiver_id=agent_id,
                message_type=message_type,
                content=content,
                expect_response=False,
            )
            responses.append(response)
        return responses

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

    def _get_available_agents(self) -> List[str]:
        return sorted(self._known_agents)

    def update_known_agents(self, agents: List[str]) -> None:
        self._known_agents = set(agents)
