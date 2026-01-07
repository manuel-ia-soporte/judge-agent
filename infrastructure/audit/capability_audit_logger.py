# infrastructure/audit/capability_audit_logger.py
from datetime import datetime
from typing import Any


class CapabilityAuditLogger:
    """
    Records capability invocations for traceability and compliance.
    """

    def log(
        self,
        agent_name: str,
        capability: str,
        subject: str,
        payload: Any,
        success: bool,
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "capability": capability,
            "subject": subject,
            "success": success,
        }

        # Replace it with DB / SIEM / Event Bus
        print(f"[AUDIT] {record}")
