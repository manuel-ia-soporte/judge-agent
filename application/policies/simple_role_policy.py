class SimpleRolePolicy:
    """
    Very simple role-based capability policy.
    """

    def __init__(self, role_permissions: dict[str, set[str]]):
        self._role_permissions = role_permissions

    def is_allowed(self, permission: str, subject: str) -> bool:
        return permission in self._role_permissions.get(subject, set())
