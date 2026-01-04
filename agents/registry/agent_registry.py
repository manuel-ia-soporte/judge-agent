# agents/registry/agent_registry.py
from typing import Dict, Set
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from domain.models.agent import Agent, AgentCapability


class AgentRegistry:
    """Registry for managing agents in the system"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_capabilities: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self.agent_heartbeats: Dict[str, datetime] = {}
        self.agent_tasks: Dict[str, int] = defaultdict(int)
        self.max_inactive_time = timedelta(minutes=5)
        self.logger = logging.getLogger(__name__)

        # Start cleanup task
        asyncio.create_task(self._cleanup_inactive_agents())

    async def register_agent(self, agent: Agent) -> bool:
        """Register an agent in the registry"""
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent {agent.agent_id} already registered")
            return False

        self.agents[agent.agent_id] = agent
        self.agent_heartbeats[agent.agent_id] = datetime.now()

        # Index capabilities
        for capability in agent.capabilities.capabilities:
            self.agent_capabilities[capability].add(agent.agent_id)

        self.logger.info(f"Agent registered: {agent.agent_id} ({agent.agent_name})")
        return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry"""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        # Remove from