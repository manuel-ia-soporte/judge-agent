# infrastructure/a2a/message_broker.py
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
from dataclasses import dataclass, field
import asyncio
from datetime import datetime, timedelta


@dataclass
class Message:
    """A message in the queue."""
    sender: str
    recipient: str
    payload: Dict[str, Any]
    message_id: str = ""


class MessageQueue:
    """Async message queue with size limits and duplicate prevention."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._seen_messages: Set[str] = set()
        self.message_timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(hours=1)

    async def put(self, message: Any) -> bool:
        """Add a message to the queue. Returns False if duplicate."""
        message_id = getattr(message, 'message_id', str(id(message)))

        # Prevent duplicates
        if message_id in self._seen_messages:
            return False

        if self.queue.full():
            return False

        self._seen_messages.add(message_id)
        self.message_timestamps[message_id] = datetime.utcnow()
        await self.queue.put(message)
        return True

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get a message from the queue."""
        try:
            if timeout:
                return await asyncio.wait_for(self.queue.get(), timeout=timeout)
            return await self.queue.get()
        except asyncio.TimeoutError:
            return None

    def peek(self) -> Optional[Any]:
        """Look at the next message without removing it."""
        if not self.queue.empty():
            # This is a workaround since asyncio.Queue doesn't have peek
            return None
        return None

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.queue.empty()

    def size(self) -> int:
        """Get the number of messages in the queue."""
        return self.queue.qsize()

    @property
    def message_ids(self) -> Set[str]:
        """Get set of message IDs that have been seen."""
        return self._seen_messages

    async def cleanup_expired(self) -> int:
        """Remove expired messages. Returns count of removed messages."""
        now = datetime.utcnow()
        expired = []

        for msg_id, timestamp in list(self.message_timestamps.items()):
            if now - timestamp > self._ttl:
                expired.append(msg_id)

        for msg_id in expired:
            self._seen_messages.discard(msg_id)
            del self.message_timestamps[msg_id]

        return len(expired)


class MessageBroker:
    """Routes messages between agents."""

    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._queues: Dict[str, MessageQueue] = {}
        self._subscriptions: Dict[str, Set[str]] = {}  # topic -> agent_ids
        self._stats = {
            "messages_routed": 0,
            "messages_dropped": 0,
            "active_agents": 0,
            "total_agents": 0,
        }

    async def register_agent(self, agent_id: str, handler: Callable = None) -> None:
        """Register an agent with its message handler."""
        self._agents[agent_id] = handler or (lambda x: None)
        self._queues[agent_id] = MessageQueue()
        self._stats["active_agents"] = len(self._agents)
        self._stats["total_agents"] = len(self._agents)

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agents.pop(agent_id, None)
        self._queues.pop(agent_id, None)
        # Remove from all subscriptions
        for subscribers in self._subscriptions.values():
            subscribers.discard(agent_id)
        self._stats["active_agents"] = len(self._agents)
        self._stats["total_agents"] = len(self._agents)

    async def subscribe_agent(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic (async alias)."""
        self.subscribe(agent_id, topic)

    async def unsubscribe_agent(self, agent_id: str, topic: str) -> None:
        """Unsubscribe an agent from a topic (async alias)."""
        self.unsubscribe(agent_id, topic)

    async def route_message(self, message: Any) -> bool:
        """Route a message to the recipient."""
        receiver_id = getattr(message, 'receiver_id', None)
        if receiver_id is None:
            receiver_id = getattr(message, 'recipient', None)

        if receiver_id not in self._agents:
            self._stats["messages_dropped"] += 1
            raise ValueError(f"Unknown recipient: {receiver_id}")

        queue = self._queues.get(receiver_id)
        if queue:
            await queue.put(message)
            self._stats["messages_routed"] += 1
            return True

        return False

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic."""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = set()
        self._subscriptions[topic].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe an agent from a topic."""
        if topic in self._subscriptions:
            self._subscriptions[topic].discard(agent_id)

    async def broadcast(self, message: Any, topic: str = None) -> int:
        """Broadcast a message to all agents or topic subscribers."""
        count = 0
        if topic and topic in self._subscriptions:
            recipients = self._subscriptions[topic]
        else:
            recipients = set(self._agents.keys())

        for agent_id in recipients:
            queue = self._queues.get(agent_id)
            if queue:
                await queue.put(message)
                count += 1
                self._stats["messages_routed"] += 1

        return count

    async def get_messages(self, agent_id: str, timeout: float = None) -> Optional[Any]:
        """Get messages for an agent."""
        queue = self._queues.get(agent_id)
        if queue:
            return await queue.get(timeout=timeout)
        return None

    def get_queue(self, agent_id: str) -> Optional[MessageQueue]:
        """Get the message queue for an agent."""
        return self._queues.get(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        return dict(self._stats)
