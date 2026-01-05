# infrastructure/a2a/a2a_server.py
from infrastructure.a2a.message_broker import MessageBroker


class A2AServer:
    def __init__(self, broker: MessageBroker) -> None:
        self._broker = broker

    def register(self, agent_id: str, handler) -> None:
        self._broker.register_agent(agent_id, handler)

    def unregister(self, agent_id: str) -> None:
        self._broker.unregister_agent(agent_id)

    def send(self, sender: str, recipient: str, payload: dict) -> None:
        self._broker.route_message(sender, recipient, payload)
