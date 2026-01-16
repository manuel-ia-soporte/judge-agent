# infrastructure/a2a/a2a_server.py
import asyncio
import json
from typing import Any, Dict, Optional
from datetime import datetime

from infrastructure.a2a.message_broker import MessageBroker
from contracts.evaluation_contracts import A2AMessage


class A2AServer:
    """A2A protocol server for agent communication."""

    def __init__(self, broker: MessageBroker = None):
        self.message_broker = broker or MessageBroker()
        self.active_connections: Dict[str, Any] = {}  # agent_id -> websocket

    async def websocket_endpoint(self, websocket: Any, agent_id: str) -> None:
        """Handle a WebSocket connection from an agent."""
        await websocket.accept()
        self.active_connections[agent_id] = websocket
        await self.message_broker.register_agent(agent_id, websocket)

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await self._process_message(agent_id, message)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            await self.message_broker.unregister_agent(agent_id)
            self.active_connections.pop(agent_id, None)

    async def _process_message(self, sender_id: str, message: Dict[str, Any]) -> None:
        """Process an incoming message."""
        a2a_message = A2AMessage(
            message_id=message.get("message_id", ""),
            sender_id=sender_id,
            receiver_id=message.get("receiver_id", ""),
            message_type=message.get("message_type", ""),
            content=message.get("content", {}),
            timestamp=datetime.utcnow(),
            correlation_id=message.get("correlation_id"),
        )

        receiver_id = message.get("receiver_id")

        if receiver_id == "*":
            # Broadcast
            await self.message_broker.broadcast(a2a_message)
        else:
            # Direct message
            try:
                await self.message_broker.route_message(a2a_message)
            except ValueError:
                # Unknown recipient, ignore
                pass

    async def _send_to_websocket(self, agent_id: str, message: Any) -> None:
        """Send a message to an agent via WebSocket."""
        websocket = self.active_connections.get(agent_id)
        if websocket:
            if isinstance(message, A2AMessage):
                data = message.model_dump_json()
            else:
                data = json.dumps(message)
            await websocket.send_text(data)

    async def send_to_agent(self, agent_id: str, message: Any) -> bool:
        """Send a message to a specific agent."""
        websocket = self.active_connections.get(agent_id)
        if not websocket:
            return False

        if hasattr(message, "model_dump_json"):
            payload = message.model_dump_json()
        else:
            payload = json.dumps(message)
        await websocket.send_text(payload)
        return True

    def get_connected_agents(self) -> list:
        """Get list of connected agent IDs."""
        return list(self.active_connections.keys())

    def is_agent_connected(self, agent_id: str) -> bool:
        """Check if an agent is connected."""
        return agent_id in self.active_connections
