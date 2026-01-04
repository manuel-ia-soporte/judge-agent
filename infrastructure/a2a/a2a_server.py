# infrastructure/a2a/a2a_server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
from typing import Dict, Any
import logging
from contracts.evaluation_contracts import A2AMessage
from infrastructure.a2a.message_broker import MessageBroker


class A2AServer:
    """A2A Protocol Server for agent communication"""

    def __init__(self):
        self.app = FastAPI(title="A2A Server")
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_broker = MessageBroker()
        self._setup_routes()

    def _setup_routes(self):
        """Setup WebSocket routes"""

        @self.app.websocket("/a2a/ws/{agent_id}")
        async def websocket_endpoint(websocket: WebSocket, agent_id: str):
            await websocket.accept()
            self.active_connections[agent_id] = websocket

            try:
                # Register agent
                await self.message_broker.register_agent(agent_id, websocket)

                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Process the incoming message
                    await self._process_message(agent_id, message)

            except WebSocketDisconnect:
                logging.info(f"Agent {agent_id} disconnected")
                await self.message_broker.unregister_agent(agent_id)
                self.active_connections.pop(agent_id, None)

        @self.app.post("/a2a/send")
        async def send_message(message: A2AMessage):
            """HTTP endpoint for sending messages"""
            await self.message_broker.route_message(message)
            return {"status": "sent", "message_id": message.message_id}

        @self.app.get("/a2a/agents")
        async def list_agents():
            """List registered agents"""
            return {
                "agents": list(self.active_connections.keys()),
                "count": len(self.active_connections)
            }

    async def _process_message(self, sender_id: str, message: Dict[str, Any]):
        """Process the incoming A2A message"""

        # Validate message structure
        if "message_type" not in message or "receiver_id" not in message:
            logging.error(f"Invalid message from {sender_id}")
            return

        # Create A2A message contract
        a2a_message = A2AMessage(
            message_id=message.get("message_id", f"msg_{hash(str(message))}"),
            sender_id=sender_id,
            receiver_id=message["receiver_id"],
            message_type=message["message_type"],
            content=message.get("content", {}),
            correlation_id=message.get("correlation_id")
        )

        # Route through message broker
        await self.message_broker.route_message(a2a_message)

    async def send_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send the message to specific agent"""
        if agent_id in self.active_connections:
            websocket = self.active_connections[agent_id]
            await websocket.send_text(json.dumps(message))
        else:
            logging.warning(f"Agent {agent_id} not connected")

    def get_app(self):
        """Get FastAPI app"""
        return self.app