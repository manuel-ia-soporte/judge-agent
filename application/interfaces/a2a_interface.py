# application/interfaces/a2a_interface.py
from typing import Dict, Any, List, Optional
import websockets
import json
import asyncio
import uuid
import logging
from datetime import datetime
from contracts.evaluation_contracts import A2AMessage


class A2AClient:
    """Client for A2A protocol communication"""

    def __init__(self, websocket_url: str, agent_id: str):
        self.websocket_url = websocket_url
        self.agent_id = agent_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.message_callbacks: Dict[str, callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.is_connected = False
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Connect to A2A server"""
        try:
            self.websocket = await websockets.connect(
                f"{self.websocket_url}/a2a/ws/{self.agent_id}"
            )
            self.is_connected = True
            self.logger.info(f"A2A client connected for agent {self.agent_id}")

            # Start message listener
            asyncio.create_task(self._listen_messages())

        except Exception as e:
            self.logger.error(f"Failed to connect to A2A server: {e}")
            raise

    async def send_message(
            self,
            receiver_id: str,
            message_type: str,
            content: Dict[str, Any],
            correlation_id: Optional[str] = None,
            timeout: int = 30
    ) -> Dict[str, Any]:
        """Send message and wait for response"""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to A2A server")

        message_id = str(uuid.uuid4())
        correlation_id = correlation_id or message_id

        # Create message
        message = A2AMessage(
            message_id=message_id,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id
        )

        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[correlation_id] = response_future

        try:
            # Send message
            await self.websocket.send(message.json())
            self.logger.debug(f"Sent message {message_id} to {receiver_id}")

            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to {message_id}")
            raise TimeoutError(f"No response from {receiver_id} within {timeout}s")

        finally:
            # Clean up
            self.pending_responses.pop(correlation_id, None)

    async def broadcast_message(
            self,
            message_type: str,
            content: Dict[str, Any],
            agent_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Broadcast message to multiple agents"""
        # This would implement broadcasting logic
        # For simplicity, we'll just send to judge agent
        return [await self.send_message("judge_agent", message_type, content)]

    def register_callback(self, message_type: str, callback: callable):
        """Register callback for specific message types"""
        self.message_callbacks[message_type] = callback

    async def _listen_messages(self):
        """Listen for incoming messages"""
        try:
            async for message_raw in self.websocket:
                message_data = json.loads(message_raw)

                # Check if this is a response to pending request
                correlation_id = message_data.get("correlation_id")
                if correlation_id in self.pending_responses:
                    self.pending_responses[correlation_id].set_result(message_data)
                    continue

                # Handle via callback if registered
                message_type = message_data.get("message_type")
                if message_type in self.message_callbacks:
                    callback = self.message_callbacks[message_type]
                    asyncio.create_task(callback(message_data))
                else:
                    self.logger.warning(f"No callback for message type: {message_type}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("A2A connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error in message listener: {e}")
            self.is_connected = False

    async def close(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False


class A2AInterface:
    """High-level interface for A2A communication"""

    def __init__(self, a2a_url: str, agent_id: str):
        self.client = A2AClient(a2a_url, agent_id)

    async def request_evaluation(
            self,
            analysis_content: str,
            source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Request evaluation from judge agent"""
        return await self.client.send_message(
            receiver_id="judge_agent",
            message_type="evaluation_request",
            content={
                "analysis_content": analysis_content,
                "source_documents": source_documents,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def query_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Query agent capabilities"""
        return await self.client.send_message(
            receiver_id=agent_id,
            message_type="capability_query",
            content={"query": "capabilities"}
        )

    async def notify_evaluation_complete(
            self,
            evaluation_id: str,
            agent_id: str,
            result: Dict[str, Any]
    ):
        """Notify agent that evaluation is complete"""
        await self.client.send_message(
            receiver_id=agent_id,
            message_type="evaluation_complete",
            content={
                "evaluation_id": evaluation_id,
                "result": result
            }
        )

    async def register_for_updates(self, agent_types: List[str]):
        """Register to receive updates from specific agent types"""
        # Implementation would register with message broker
        pass