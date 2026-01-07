# infrastructure/mcp/finance_mcp_server.py
from typing import Any

from agents.finance_agent.core.finance_agent import FinanceAgent
from application.policies.capability_policy import CapabilityPolicy
from infrastructure.audit.capability_audit_logger import CapabilityAuditLogger
from infrastructure.metrics.capability_metrics import CapabilityMetrics
import time


class FinanceMCPServer:
    """
    MCP adapter with dynamic capability invocation,
    auditing, and policy enforcement.
    """

    def __init__(
        self,
        agent: FinanceAgent,
        policy: CapabilityPolicy,
        subject: str,
        audit_logger: CapabilityAuditLogger,
    ):
        self._agent = agent
        self._policy = policy
        self._subject = subject
        self._audit = audit_logger
        self._metrics = CapabilityMetrics()

    async def invoke(self, capability_name: str, payload: Any) -> Any:
        capability = self._agent.get_capability(capability_name)

        if not self._policy.is_allowed(
            capability.required_permission, self._subject
        ):
            self._audit.log(
                "FinanceAgent", capability_name, self._subject, payload, False
            )
            raise PermissionError("Permission denied")

        start = time.time()
        try:
            result = (
                await capability.handler(payload)
                if hasattr(capability.handler, "__await__")
                else capability.handler(payload)
            )

            self._metrics.record(capability_name, time.time() - start, True)
            self._audit.log(
                "FinanceAgent", capability_name, self._subject, payload, True
            )
            return result

        except Exception:
            self._audit.log(
                "FinanceAgent", capability_name, self._subject, payload, False
            )
            self._metrics.record(capability_name, time.time() - start, False)
            raise
