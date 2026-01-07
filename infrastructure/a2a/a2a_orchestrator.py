# infrastructure/a2a/a2a_orchestrator.py
class A2AOrchestrator:
    """
    Agent-to-Agent orchestrator.
    """

    def __init__(self, router):
        self._router = router

    async def invoke(self, capability: str, payload):
        agent = self._router.find_agent_for_capability(capability)
        cap = agent.get_capability(capability)
        return await cap.handler(payload)
