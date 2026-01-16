# infrastructure/a2a/message_broker.py
from typing import Any, Dict, List, Optional, Set
import asyncio
import json
from datetime import datetime, timedelta


class MessageQueue:
    """Async message queue with size limits and duplicate prevention."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._seen_messages: Set[str] = set()
        self.message_timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(hours=1)

    async def put(self, message: Any) -> bool:
        message_id = getattr(message, "message_id", str(id(message)))
        if message_id in self._seen_messages or self.queue.full():
            return False

        self._seen_messages.add(message_id)
        self.message_timestamps[message_id] = datetime.utcnow()
        await self.queue.put(message)
        return True

    async def get(self, timeout: Optional[float] = None) -> Any:
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.queue.get(), timeout=timeout)
            return await self.queue.get()
        except asyncio.TimeoutError:
            return None

    def size(self) -> int:
        return self.queue.qsize()

    @property
    def message_ids(self) -> Set[str]:
        return set(self._seen_messages)

    def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        expired = [
            msg_id
            for msg_id, timestamp in self.message_timestamps.items()
            if now - timestamp > self._ttl
        ]
        for msg_id in expired:
            self._seen_messages.discard(msg_id)
            self.message_timestamps.pop(msg_id, None)
        return len(expired)


class MessageBroker:
    """Routes messages between agents."""

    def __init__(self):
        self.agent_queues: Dict[str, MessageQueue] = {}
        self.agent_connections: Dict[str, Any] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = {}
        self.message_handlers: Dict[str, Any] = {}
        self._stats = {
            "messages_routed": 0,
            "messages_dropped": 0,
            "total_messages_logged": 0,
        }

    async def register_agent(self, agent_id: str, connection: Any = None) -> None:
        self.agent_queues[agent_id] = MessageQueue()
        if connection is not None:
            self.agent_connections[agent_id] = connection

    async def unregister_agent(self, agent_id: str) -> None:
        self.agent_queues.pop(agent_id, None)
        self.agent_connections.pop(agent_id, None)
        for subscribers in self.agent_subscriptions.values():
            subscribers.discard(agent_id)

    async def subscribe_agent(self, agent_id: str, topic: str) -> None:
        self.agent_subscriptions.setdefault(topic, set()).add(agent_id)

    async def unsubscribe_agent(self, agent_id: str, topic: str) -> None:
        if topic in self.agent_subscriptions:
            self.agent_subscriptions[topic].discard(agent_id)

    async def route_message(self, message: Any) -> bool:
        receiver_id = getattr(message, "receiver_id", None)
        message_type = getattr(message, "message_type", "")

        if receiver_id in {"broadcast", "*"} or message_type in {"broadcast", "announcement"}:
            await self.broadcast(message)
            return True

        queue = self.agent_queues.get(receiver_id)
        if not queue:
            self._stats["messages_dropped"] += 1
            return False

        stored = await queue.put(message)
        if not stored:
            self._stats["messages_dropped"] += 1
            return False

        self._stats["messages_routed"] += 1
        self._stats["total_messages_logged"] += 1

        connection = self.agent_connections.get(receiver_id)
        if connection:
            payload = message.model_dump_json() if hasattr(message, "model_dump_json") else json.dumps(
                {
                    "message_id": getattr(message, "message_id", ""),
                    "sender_id": getattr(message, "sender_id", ""),
                    "receiver_id": receiver_id,
                    "message_type": message_type,
                    "content": getattr(message, "content", {}),
                }
            )
            await connection.send_text(payload)
        return True

    async def broadcast(self, message: Any, topic: Optional[str] = None) -> int:
        recipients: Set[str]
        if topic and topic in self.agent_subscriptions:
            recipients = set(self.agent_subscriptions[topic])
        else:
            recipients = set(self.agent_queues.keys())
        sender = getattr(message, "sender_id", None)
        delivered = 0
        for agent_id in recipients:
            if agent_id == sender:
                continue
            queue = self.agent_queues.get(agent_id)
            if queue and await queue.put(message):
                delivered += 1
                self._stats["messages_routed"] += 1
                self._stats["total_messages_logged"] += 1
        return delivered

    async def get_messages(
        self, agent_id: str, limit: Optional[int] = None, timeout: Optional[float] = None
    ) -> List[Any]:
        queue = self.agent_queues.get(agent_id)
        if not queue:
            return []

        messages: List[Any] = []
        if queue.queue.qsize() == 0:
            message = await queue.get(timeout=timeout)
            if message is None:
                return []
            messages.append(message)

        target = limit or queue.queue.qsize() + len(messages)
        while len(messages) < target and not queue.queue.empty():
            try:
                messages.append(queue.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def get_queue(self, agent_id: str) -> Optional[MessageQueue]:
        return self.agent_queues.get(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self.agent_queues),
            "connected_agents": len(self.agent_connections),
            "total_messages_logged": self._stats["total_messages_logged"],
            "queue_sizes": {agent: queue.size() for agent, queue in self.agent_queues.items()},
            "subscriptions": {topic: list(subs) for topic, subs in self.agent_subscriptions.items()},
        }
