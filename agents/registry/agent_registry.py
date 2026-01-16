# agents/registry/agent_registry.py
from typing import Dict, Any, Optional, Protocol, List


class AgentRepository(Protocol):
    """Protocol for agent storage."""

    def save(self, agent_id: str, agent: Any) -> None:
        ...

    def find(self, agent_id: str) -> Optional[Any]:
        ...

    def find_all(self) -> Dict[str, Any]:
        ...

    def delete(self, agent_id: str) -> None:
        ...


class InMemoryAgentRepository:
    """In-memory implementation of AgentRepository with async support."""

    def __init__(self) -> None:
        self._storage: Dict[str, Any] = {}

    # Sync methods for backward compatibility
    def save_sync(self, agent_id: str, agent: Any) -> None:
        self._storage[agent_id] = agent

    def find_sync(self, agent_id: str) -> Optional[Any]:
        return self._storage.get(agent_id)

    def find_all(self) -> Dict[str, Any]:
        return dict(self._storage)

    def delete_sync(self, agent_id: str) -> None:
        self._storage.pop(agent_id, None)

    # Async methods for tests
    async def save(self, agent: Any) -> bool:
        """Save an agent."""
        agent_id = getattr(agent, 'agent_id', str(id(agent)))
        self._storage[agent_id] = agent
        return True

    async def find_by_id(self, agent_id: str) -> Optional[Any]:
        """Find an agent by ID."""
        return self._storage.get(agent_id)

    async def find_by_capability(self, capability: Any) -> List[Any]:
        """Find agents by capability."""
        result = []
        for agent in self._storage.values():
            caps = getattr(agent, 'capabilities', None)
            if caps:
                capabilities_list = getattr(caps, 'capabilities', [])
                if capability in capabilities_list:
                    result.append(agent)
        return result

    async def find_by_status(self, status: Any) -> List[Any]:
        """Find agents by status."""
        return [
            agent for agent in self._storage.values()
            if getattr(agent, 'status', None) == status
        ]

    async def update_status(self, agent_id: str, status: Any) -> bool:
        """Update agent status."""
        agent = self._storage.get(agent_id)
        if agent:
            agent.status = status
            return True
        return False

    async def delete(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id in self._storage:
            del self._storage[agent_id]
            return True
        return False

    async def count(self) -> int:
        """Count total agents."""
        return len(self._storage)

    async def find_inactive_agents(self, threshold_seconds: int = 300, timeout_minutes: int = None) -> List[Any]:
        """Find agents that haven't sent heartbeat recently."""
        from datetime import datetime, timedelta
        if timeout_minutes is not None:
            threshold_seconds = timeout_minutes * 60
        threshold = datetime.utcnow() - timedelta(seconds=threshold_seconds)
        result = []
        for agent in self._storage.values():
            last_hb = getattr(agent, 'last_heartbeat', None)
            if isinstance(last_hb, datetime) and last_hb.tzinfo is not None:
                last_hb = last_hb.replace(tzinfo=None)
            if last_hb is None or last_hb < threshold:
                result.append(agent)
        return result

    async def get_capability_stats(self) -> Dict[str, int]:
        """Get statistics on agent capabilities."""
        stats: Dict[str, int] = {}
        for agent in self._storage.values():
            caps = getattr(agent, 'capabilities', None)
            if caps:
                capabilities_list = getattr(caps, 'capabilities', [])
                for cap in capabilities_list:
                    cap_name = cap.value if hasattr(cap, 'value') else str(cap)
                    stats[cap_name] = stats.get(cap_name, 0) + 1
        return stats


class AgentRegistry:
    """Registry for managing agents."""

    def __init__(self, repository: Optional[AgentRepository] = None) -> None:
        self._repository = repository or InMemoryAgentRepository()
        self._agents: Dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        self._agents[name] = agent
        sync_save = getattr(self._repository, "save_sync", None)
        if callable(sync_save):
            sync_save(name, agent)
        else:
            self._repository.save(name, agent)

    def get(self, name: str) -> Any:
        agent = self._agents.get(name)
        if agent is None:
            sync_find = getattr(self._repository, "find_sync", None)
            if callable(sync_find):
                agent = sync_find(name)
            else:
                agent = self._repository.find(name)
        return agent

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)
        sync_delete = getattr(self._repository, "delete_sync", None)
        if callable(sync_delete):
            sync_delete(name)
        else:
            self._repository.delete(name)

    def list_agents(self) -> Dict[str, Any]:
        return self._agents