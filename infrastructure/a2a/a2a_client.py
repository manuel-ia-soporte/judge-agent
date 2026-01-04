# infrastructure/a2a/a2a_client.py
from typing import Dict, Any, List, Optional
import websockets
import json
import asyncio
import uuid
import logging
from datetime import datetime, UTC
from contracts.evaluation_contracts import A2AMessage


class A2AClient:
    """A2A Protocol Client for agent communication"""

    def __init__(self, server_url: str, agent_id: str):
        self.server_url = server_url
        self.agent_id = agent_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.message_handlers: Dict[str, callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Connect to A2A server"""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.websocket = await websockets.connect(
                    f"{self.server_url}/a2a/ws/{self.agent_id}",
                    ping_interval=20,
                    ping_timeout=10
                )
                self.connected = True
                self.reconnect_attempts = 0
                self.logger.info(f"A2A client connected: {self.agent_id}")

                # Start message handler
                asyncio.create_task(self._handle_messages())
                break

            except Exception as e:
                self.reconnect_attempts += 1
                self.logger.warning(
                    f"Connection attempt {self.reconnect_attempts} failed: {e}"
                )

                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)
                else:
                    self.logger.error("Max reconnection attempts reached")
                    raise

    async def send_message(
            self,
            receiver_id: str,
            message_type: str,
            content: Dict[str, Any],
            expect_response: bool = True,
            timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Send the message to another agent"""
        if not self.connected or not self.websocket:
            raise ConnectionError("Not connected to A2A server")

        message_id = str(uuid.uuid4())
        correlation_id = message_id if expect_response else None

        message = A2AMessage(
            message_id=message_id,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC)
        )

        try:
            # Register for response if expected
            if expect_response and correlation_id:
                response_future = asyncio.Future()
                self.pending_responses[correlation_id] = response_future

            # Send the message
            await self.websocket.send(message.model_dump_json())
            self.logger.debug(f"Sent message {message_id} to {receiver_id}")

            # Wait for a response if expected
            if expect_response and correlation_id:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to {message_id}")
            if correlation_id in self.pending_responses:
                del self.pending_responses[correlation_id]
            raise TimeoutError(f"No response within {timeout}s")

        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            if correlation_id in self.pending_responses:
                del self.pending_responses[correlation_id]
            raise

        return None

    async def broadcast(
            self,
            message_type: str,
            content: Dict[str, Any],
            agent_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Broadcast a message to multiple agents"""
        # First get the list of agents
        agents = await self._get_available_agents()

        if agent_filter:
            agents = [agent for agent in agents if agent in agent_filter]

        # Send it to all agents
        responses = []
        for agent_id in agents:
            if agent_id != self.agent_id:  # Don't send it to self
                try:
                    response = await self.send_message(
                        agent_id, message_type, content, expect_response=False
                    )
                    if response:
                        responses.append(response)
                except Exception as e:
                    self.logger.warning(f"Failed to broadcast to {agent_id}: {e}")

        return responses

    def register_handler(self, message_type: str, handler: callable):
        """Register handler for the specific message type"""
        self.message_handlers[message_type] = handler

    async def _handle_messages(self):
        """Handle incoming messages"""
        try:
            async for message_raw in self.websocket:
                try:
                    message_data = json.loads(message_raw)

                    # Check if this is a response to pending request
                    correlation_id = message_data.get("correlation_id")
                    if correlation_id and correlation_id in self.pending_responses:
                        self.pending_responses[correlation_id].set_result(message_data)
                        del self.pending_responses[correlation_id]
                        continue

                    # Handle via registered handler
                    message_type = message_data.get("message_type")
                    if message_type in self.message_handlers:
                        handler = self.message_handlers[message_type]
                        asyncio.create_task(handler(message_data))
                    else:
                        self.logger.warning(f"No handler for message type: {message_type}")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON message: {e}")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("A2A connection closed")
            self.connected = False
            await self._attempt_reconnect()
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            self.connected = False
            await self._attempt_reconnect()

    async def _attempt_reconnect(self):
        """Attempt to reconnect to A2A server"""
        self.logger.info("Attempting to reconnect...")
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")

    @staticmethod
    async def _get_available_agents() -> List[str]:
        """Get the list of available agents"""
        try:
            # This would query the A2A server for registered agents
            # For now, return the empty list
            return []
        except Exception as e:
            print(f"Failed to get available agents: {e}")
            return []

    async def close(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self.logger.info("A2A client closed")


class A2AClientFactory:
    """Factory for creating A2A clients"""

    @staticmethod
    def create_judge_client(server_url: str) -> A2AClient:
        """Create judge agent A2A client"""
        return A2AClient(server_url, "judge_agent")

    @staticmethod
    def create_finance_agent_client(server_url: str, agent_id: str) -> A2AClient:
        """Create finance agent A2A client"""
        return A2AClient(server_url, f"finance_agent_{agent_id}")

    @staticmethod
    def create_sec_client(server_url: str) -> A2AClient:
        """Create SEC agent A2A client"""
        return A2AClient(server_url, "sec_agent")