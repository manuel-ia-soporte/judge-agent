from dataclasses import dataclass
from typing import Callable, Dict, Any, Type


@dataclass(frozen=True)
class CapabilitySchema:
    """
    Describes the input/output contract of a capability.
    """
    input_type: Type
    output_type: Type
    description: str


@dataclass(frozen=True)
class AgentCapability:
    """
    Self-describing agent capability.
    """
    name: str
    schema: CapabilitySchema
    handler: Callable[..., Any]
    required_permission: str


class CapabilityRegistry:
    """
    Runtime registry of discoverable agent capabilities.
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, AgentCapability] = {}

    def register(self, capability: AgentCapability) -> None:
        if capability.name in self._capabilities:
            raise ValueError(f"Capability '{capability.name}' already registered")
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> AgentCapability:
        return self._capabilities[name]

    def list(self) -> Dict[str, AgentCapability]:
        return dict(self._capabilities)
