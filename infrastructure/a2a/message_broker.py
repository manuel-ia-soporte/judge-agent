# infrastructure/a2a/message_broker.py
from typing import Dict, List, Any, Optional, Set
import asyncio
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from contracts.evaluation_contracts import A2AMessage


class MessageQueue:
    """Message queue for A2A messaging"""

    def __init__(self, max_size: int = 10000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.message_ids: Set[str] = set()
        self.message_ttl = timedelta(hours=1)
        self.message_timestamps: Dict[str, datetime] = {}

    async def put(self, message: A2AMessage):
        """Put message in queue"""
        if message.message_id not in self.message_ids:
            await self.queue.put(message)
            self.message_ids.add(message.message_id)
            self.message_timestamps[message.message_id] = datetime.utcnow()

    async def get(self) -> A2AMessage:
        """Get message from queue"""
        message = await self.queue.get()
        return message

    def task_done(self):
        """Mark task as done"""
        self.queue.task_done()

    def cleanup_expired(self):
        """Clean up expired messages"""
        now = datetime.utcnow()
        expired = [
            msg_id for msg_id, timestamp in self.message_timestamps.items()
            if now - timestamp > self.message_ttl
        ]

        for msg_id in expired:
            self.message_ids.discard(msg_id)
            self.message_timestamps.pop(msg_id, None)


class MessageBroker:
    """A2A Message Broker for agent communication"""

    def __init__(self):
        self.agent_queues: Dict[str, MessageQueue] = {}
        self.agent_connections: Dict[str, Any] = {}
        self.agent_subscriptions: Dict[str, List[str]] = defaultdict(list)
        self.message_log: List[Dict[str, Any]] = []
        self.max_log_size = 10000
        self.logger = logging.getLogger(__name__)

        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())

    async def register_agent(self, agent_id: str, connection: Any):
        """Register agent with message broker"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = MessageQueue()

        self.agent_connections[agent_id] = connection
        self.logger.info(f"Agent registered: {agent_id}")

    async def unregister_agent(self, agent_id: str):
        """Unregister agent from message broker"""
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]

        if agent_id in self.agent_connections:
            del self.agent_connections[agent_id]

        # Remove from subscriptions
        for subscribers in self.agent_subscriptions.values():
            if agent_id in subscribers:
                subscribers.remove(agent_id)

        self.logger.info(f"Agent unregistered: {agent_id}")

    async def route_message(self, message: A2AMessage) -> bool:
        """Route message to destination agent"""
        try:
            # Log message
            self._log_message(message)

            # Check if agent exists
            if message.receiver_id not in self.agent_queues:
                self.logger.warning(f"Unknown agent: {message.receiver_id}")
                return False

            # Add to agent's queue
            await self.agent_queues[message.receiver_id].put(message)

            # Notify via WebSocket if connected
            await self._notify_agent(message.receiver_id)

            # Check for broadcast
            if message.receiver_id == "broadcast":
                await self._broadcast_message(message)

            self.logger.debug(f"Message routed: {message.message_id} -> {message.receiver_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to route message: {e}")
            return False

    async def subscribe_agent(self, agent_id: str, topic: str):
        """Subscribe agent to topic"""
        if agent_id not in self.agent_subscriptions[topic]:
            self.agent_subscriptions[topic].append(agent_id)
            self.logger.info(f"Agent {agent_id} subscribed to {topic}")

    async def unsubscribe_agent(self, agent_id: str, topic: str):
        """Unsubscribe agent from topic"""
        if topic in self.agent_subscriptions and agent_id in self.agent_subscriptions[topic]:
            self.agent_subscriptions[topic].remove(agent_id)
            self.logger.info(f"Agent {agent_id} unsubscribed from {topic}")

    async def get_messages(self, agent_id: str, limit: int = 10) -> List[A2AMessage]:
        """Get messages for agent"""
        if agent_id not in self.agent_queues:
            return []

        messages = []
        queue = self.agent_queues[agent_id]

        try:
            for _ in range(min(limit, queue.queue.qsize())):
                message = await queue.get()
                messages.append(message)
                queue.task_done()
        except asyncio.QueueEmpty:
            pass

        return messages

    async def _notify_agent(self, agent_id: str):
        """Notify agent of new messages via WebSocket"""
        if agent_id in self.agent_connections:
            try:
                connection = self.agent_connections[agent_id]
                await connection.send_text(json.dumps({
                    "type": "notification",
                    "message": "new_messages_available",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception as e:
                self.logger.warning(f"Failed to notify agent {agent_id}: {e}")

    async def _broadcast_message(self, message: A2AMessage):
        """Broadcast message to all agents"""
        for agent_id in list(self.agent_queues.keys()):
            if agent_id != message.sender_id:
                broadcast_msg = A2AMessage(
                    message_id=f"broadcast_{message.message_id}",
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    message_type="broadcast",
                    content=message.content,
                    correlation_id=message.correlation_id
                )
                await self.agent_queues[agent_id].put(broadcast_msg)

    def _log_message(self, message: A2AMessage):
        """Log message for auditing"""
        log_entry = {
            "message_id": message.message_id,
            "sender": message.sender_id,
            "receiver": message.receiver_id,
            "type": message.message_type,
            "timestamp": message.timestamp.isoformat(),
            "size": len(json.dumps(message.dict()))
        }

        self.message_log.append(log_entry)

        # Trim log if too large
        if len(self.message_log) > self.max_log_size:
            self.message_log = self.message_log[-self.max_log_size // 2:]

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired messages"""
        while True:
            await asyncio.sleep(3600)  # Run every hour

            for queue in self.agent_queues.values():
                queue.cleanup_expired()

            # Clean old log entries
            cutoff = datetime.utcnow() - timedelta(days=7)
            self.message_log = [
                entry for entry in self.message_log
                if datetime.fromisoformat(entry["timestamp"]) > cutoff
            ]

            self.logger.debug("Message broker cleanup completed")

    def get_stats(self) -> Dict[str, Any]:
        """Get message broker statistics"""
        stats = {
            "total_agents": len(self.agent_queues),
            "connected_agents": len(self.agent_connections),
            "total_messages_logged": len(self.message_log),
            "queue_sizes": {
                agent_id: queue.queue.qsize()
                for agent_id, queue in self.agent_queues.items()
            },
            "subscriptions": {
                topic: len(subscribers)
                for topic, subscribers in self.agent_subscriptions.items()
            }
        }

        # Calculate message rates
        if self.message_log:
            recent_messages = [
                msg for msg in self.message_log[-100:]
                if datetime.fromisoformat(msg["timestamp"]) > datetime.utcnow() - timedelta(minutes=5)
            ]
            stats["message_rate_5min"] = len(recent_messages) / 5  # messages per minute

        return stats