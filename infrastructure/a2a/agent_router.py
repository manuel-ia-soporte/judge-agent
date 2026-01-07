# infrastructure/a2a/agent_router.py
from typing import Dict, Any


class AgentRouter:
    """
    Routes capability invocations across multiple agents.
    """

    def __init__(self, agents: Dict[str, Any]):
        self._agents = agents

    def find_agent_for_capability(self, capability: str):
        for agent in self._agents.values():
            if capability in agent.list_capabilities():
                return agent
        raise LookupError(f"No agent supports capability '{capability}'")
