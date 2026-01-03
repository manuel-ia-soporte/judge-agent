# domain/repositories/agent_repository.py
from typing import List, Optional, Dict, Any, Set
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from domain.models.agent import Agent, AgentStatus, AgentCapability
import asyncio
import logging


class AgentRepository(ABC):
    """Repository interface for Agent domain entity"""

    @abstractmethod
    async def save(self, agent: Agent) -> bool:
        """Save agent to repository"""
        pass

    @abstractmethod
    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        pass

    @abstractmethod
    async def find_by_capability(self, capability: AgentCapability) -> List[Agent]:
        """Find agents by capability"""
        pass

    @abstractmethod
    async def find_by_status(self, status: AgentStatus) -> List[Agent]:
        """Find agents by status"""
        pass

    @abstractmethod
    async def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        pass

    @abstractmethod
    async def delete(self, agent_id: str) -> bool:
        """Delete agent from repository"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Count agents in repository"""
        pass


class InMemoryAgentRepository(AgentRepository):
    """In-memory implementation of AgentRepository"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.capability_index: Dict[AgentCapability, Set[str]] = {}
        self.status_index: Dict[AgentStatus, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        self.lock = asyncio.Lock()

    async def save(self, agent: Agent) -> bool:
        """Save agent to repository"""
        async with self.lock:
            try:
                # Remove from old indexes
                if agent.agent_id in self.agents:
                    await self._remove_from_indexes(agent.agent_id)

                # Add to storage
                self.agents[agent.agent_id] = agent

                # Add to indexes
                await self._add_to_indexes(agent)

                self.logger.debug(f"Agent saved: {agent.agent_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to save agent {agent.agent_id}: {e}")
                return False

    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        async with self.lock:
            return self.agents.get(agent_id)

    async def find_by_capability(self, capability: AgentCapability) -> List[Agent]:
        """Find agents by capability"""
        async with self.lock:
            agent_ids = self.capability_index.get(capability, set())
            agents = [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
            return agents

    async def find_by_status(self, status: AgentStatus) -> List[Agent]:
        """Find agents by status"""
        async with self.lock:
            agent_ids = self.status_index.get(status, set())
            agents = [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
            return agents

    async def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        async with self.lock:
            if agent_id not in self.agents:
                return False

            agent = self.agents[agent_id]

            # Remove from old status index
            old_status = agent.status
            if old_status in self.status_index:
                self.status_index[old_status].discard(agent_id)

            # Update status
            agent.status = status

            # Add to new status index
            if status not in self.status_index:
                self.status_index[status] = set()
            self.status_index[status].add(agent_id)

            # Update last heartbeat
            agent.last_heartbeat = datetime.utcnow()

            self.logger.debug(f"Agent status updated: {agent_id} -> {status}")
            return True

    async def delete(self, agent_id: str) -> bool:
        """Delete agent from repository"""
        async with self.lock:
            if agent_id not in self.agents:
                return False

            # Remove from indexes
            await self._remove_from_indexes(agent_id)

            # Remove from storage
            del self.agents[agent_id]

            self.logger.debug(f"Agent deleted: {agent_id}")
            return True

    async def count(self) -> int:
        """Count agents in repository"""
        async with self.lock:
            return len(self.agents)

    async def find_inactive_agents(self, timeout_minutes: int = 5) -> List[Agent]:
        """Find agents that haven't sent heartbeat in timeout period"""
        async with self.lock:
            cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)
            inactive_agents = []

            for agent in self.agents.values():
                if agent.last_heartbeat and agent.last_heartbeat < cutoff:
                    inactive_agents.append(agent)

            return inactive_agents

    async def get_capability_stats(self) -> Dict[AgentCapability, int]:
        """Get statistics on agent capabilities"""
        async with self.lock:
            stats = {}
            for capability, agent_ids in self.capability_index.items():
                stats[capability] = len(agent_ids)
            return stats

    async def get_status_stats(self) -> Dict[AgentStatus, int]:
        """Get statistics on agent statuses"""
        async with self.lock:
            stats = {}
            for status, agent_ids in self.status_index.items():
                stats[status] = len(agent_ids)
            return stats

    async def _add_to_indexes(self, agent: Agent):
        """Add agent to all indexes"""
        # Capability index
        for capability in agent.capabilities.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(agent.agent_id)

        # Status index
        if agent.status not in self.status_index:
            self.status_index[agent.status] = set()
        self.status_index[agent.status].add(agent.agent_id)

    async def _remove_from_indexes(self, agent_id: str):
        """Remove agent from all indexes"""
        # Capability index
        for capability, agent_ids in self.capability_index.items():
            agent_ids.discard(agent_id)
            if not agent_ids:
                del self.capability_index[capability]

        # Status index
        for status, agent_ids in self.status_index.items():
            agent_ids.discard(agent_id)
            if not agent_ids:
                del self.status_index[status]


class CachedAgentRepository(AgentRepository):
    """Agent repository with caching"""

    def __init__(self, primary_repo: AgentRepository, cache_ttl: int = 300):
        self.primary_repo = primary_repo
        self.cache: Dict[str, tuple[Agent, datetime]] = {}
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(__name__)
        self.lock = asyncio.Lock()

    async def save(self, agent: Agent) -> bool:
        """Save agent to repository and update cache"""
        async with self.lock:
            # Save to primary repository
            success = await self.primary_repo.save(agent)

            if success:
                # Update cache
                self.cache[agent.agent_id] = (agent, datetime.utcnow())

            return success

    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID with cache lookup"""
        async with self.lock:
            # Check cache first
            cached = self.cache.get(agent_id)
            if cached:
                agent, timestamp = cached
                if datetime.utcnow() - timestamp < timedelta(seconds=self.cache_ttl):
                    return agent

            # Not in cache or expired, query primary
            agent = await self.primary_repo.find_by_id(agent_id)

            if agent:
                # Update cache
                self.cache[agent_id] = (agent, datetime.utcnow())

            return agent

    async def find_by_capability(self, capability: AgentCapability) -> List[Agent]:
        """Find agents by capability (no caching for this query)"""
        return await self.primary_repo.find_by_capability(capability)

    async def find_by_status(self, status: AgentStatus) -> List[Agent]:
        """Find agents by status (no caching for this query)"""
        return await self.primary_repo.find_by_status(status)

    async def update_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status and invalidate cache"""
        async with self.lock:
            # Update in primary repository
            success = await self.primary_repo.update_status(agent_id, status)

            if success:
                # Invalidate cache
                if agent_id in self.cache:
                    del self.cache[agent_id]

            return success

    async def delete(self, agent_id: str) -> bool:
        """Delete agent and invalidate cache"""
        async with self.lock:
            # Delete from primary repository
            success = await self.primary_repo.delete(agent_id)

            if success:
                # Invalidate cache
                if agent_id in self.cache:
                    del self.cache[agent_id]

            return success

    async def count(self) -> int:
        """Count agents (no caching for count)"""
        return await self.primary_repo.count()

    async def invalidate_cache(self, agent_id: str = None):
        """Invalidate cache entries"""
        async with self.lock:
            if agent_id:
                if agent_id in self.cache:
                    del self.cache[agent_id]
            else:
                self.cache.clear()

    async def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        async with self.lock:
            now = datetime.utcnow()
            expired = [
                agent_id for agent_id, (_, timestamp) in self.cache.items()
                if now - timestamp > timedelta(seconds=self.cache_ttl)
            ]

            for agent_id in expired:
                del self.cache[agent_id]

            if expired:
                self.logger.debug(f"Cleaned up {len(expired)} expired cache entries")


# Factory for creating repository instances
class AgentRepositoryFactory:
    """Factory for creating agent repositories"""

    @staticmethod
    def create_in_memory_repo() -> AgentRepository:
        """Create in-memory repository"""
        return InMemoryAgentRepository()

    @staticmethod
    def create_cached_repo(cache_ttl: int = 300) -> AgentRepository:
        """Create cached repository"""
        primary = InMemoryAgentRepository()
        return CachedAgentRepository(primary, cache_ttl)

    @staticmethod
    async def create_repo_with_cleanup(repo_type: str = "cached") -> AgentRepository:
        """Create repository with background cleanup task"""
        if repo_type == "cached":
            repo = AgentRepositoryFactory.create_cached_repo()

            # Start cleanup task
            async def cleanup_task():
                while True:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    if isinstance(repo, CachedAgentRepository):
                        await repo.cleanup_expired_cache()

            asyncio.create_task(cleanup_task())
            return repo

        return AgentRepositoryFactory.create_in_memory_repo()