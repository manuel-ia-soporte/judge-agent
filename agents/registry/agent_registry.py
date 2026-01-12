# agents/registry/agent_registry.py
from typing import Dict, Any


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> Any:
        return self._agents[name]

    def list_agents(self) -> Dict[str, Any]:  # ← ADD THIS
        return self._agents