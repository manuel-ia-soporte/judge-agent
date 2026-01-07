from typing import Protocol


class CapabilityPolicy(Protocol):
    """
    Contract for capability permission checks.
    """

    def is_allowed(self, permission: str, subject: str) -> bool:
        ...
