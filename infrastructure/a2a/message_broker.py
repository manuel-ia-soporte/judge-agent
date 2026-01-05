# infrastructure/a2a/message_broker.py
class MessageBroker:
    def register_agent(self, agent_id: str, handler) -> None:
        raise NotImplementedError

    def unregister_agent(self, agent_id: str) -> None:
        raise NotImplementedError

    def route_message(self, sender: str, recipient: str, payload: dict) -> None:
        raise NotImplementedError
