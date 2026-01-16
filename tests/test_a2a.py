# tests/test_a2a.py
"""
A2A Protocol Tests
"""
import pytest
import asyncio
import json
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from infrastructure.a2a.a2a_server import A2AServer
from infrastructure.a2a.a2a_client import A2AClient
from infrastructure.a2a.message_broker import MessageBroker, MessageQueue
from contracts.evaluation_contracts import A2AMessage
from agents.registry.agent_registry import AgentRegistry, InMemoryAgentRepository


class TestMessageQueue:
    """Test MessageQueue functionality"""

    @pytest.fixture
    def message_queue(self):
        """Create message queue instance"""
        return MessageQueue(max_size=5)

    @pytest.fixture
    def sample_message(self):
        """Create sample A2A message"""
        return A2AMessage(
            message_id="test_msg_001",
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type="test_message",
            content={"test": "data"},
            timestamp=datetime.utcnow()
        )

    @pytest.mark.asyncio
    async def test_put_and_get(self, message_queue, sample_message):
        """Test putting and getting messages"""
        # Put message
        await message_queue.put(sample_message)

        # Get message
        retrieved = await message_queue.get()

        assert retrieved.message_id == sample_message.message_id
        assert retrieved.sender_id == sample_message.sender_id
        assert message_queue.queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_queue_size_limit(self, message_queue):
        """Test queue size limit"""
        # Fill queue to capacity
        for i in range(5):
            msg = A2AMessage(
                message_id=f"msg_{i}",
                sender_id="sender",
                receiver_id="receiver",
                message_type="test",
                content={}
            )
            await message_queue.put(msg)

        # Queue should be full
        assert message_queue.queue.qsize() == 5
        assert message_queue.queue.full() == True

    @pytest.mark.asyncio
    async def test_duplicate_message_prevention(self, message_queue, sample_message):
        """Test duplicate message prevention"""
        # Put same message twice
        await message_queue.put(sample_message)
        await message_queue.put(sample_message)  # Should be ignored

        # Should only have one message
        assert message_queue.queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, message_queue):
        """Test cleanup of expired messages"""
        # Add message with old timestamp
        old_message = A2AMessage(
            message_id="old_msg",
            sender_id="sender",
            receiver_id="receiver",
            message_type="test",
            content={},
            timestamp=datetime.utcnow()  # Will be marked as expired after manipulation
        )

        await message_queue.put(old_message)

        # Manually set old timestamp
        message_queue.message_timestamps["old_msg"] = datetime.utcnow().replace(year=2020)

        # Clean up expired
        message_queue.cleanup_expired()

        # Message should be removed
        assert "old_msg" not in message_queue.message_ids
        assert "old_msg" not in message_queue.message_timestamps


class TestMessageBroker:
    """Test MessageBroker functionality"""

    @pytest.fixture
    def message_broker(self):
        """Create message broker instance"""
        return MessageBroker()

    @pytest.fixture
    def mock_connection(self):
        """Create mock WebSocket connection"""
        connection = AsyncMock()
        connection.send_text = AsyncMock()
        return connection

    @pytest.mark.asyncio
    async def test_register_agent(self, message_broker, mock_connection):
        """Test agent registration"""
        await message_broker.register_agent("test_agent", mock_connection)

        assert "test_agent" in message_broker.agent_queues
        assert "test_agent" in message_broker.agent_connections
        assert message_broker.agent_connections["test_agent"] == mock_connection

    @pytest.mark.asyncio
    async def test_unregister_agent(self, message_broker, mock_connection):
        """Test agent unregistration"""
        await message_broker.register_agent("test_agent", mock_connection)
        await message_broker.unregister_agent("test_agent")

        assert "test_agent" not in message_broker.agent_queues
        assert "test_agent" not in message_broker.agent_connections

    @pytest.mark.asyncio
    async def test_route_message(self, message_broker, mock_connection):
        """Test message routing"""
        # Register agent
        await message_broker.register_agent("receiver_agent", mock_connection)

        # Create message
        message = A2AMessage(
            message_id="test_msg",
            sender_id="sender_agent",
            receiver_id="receiver_agent",
            message_type="test",
            content={"data": "test"}
        )

        # Route message
        success = await message_broker.route_message(message)

        assert success == True
        assert message_broker.agent_queues["receiver_agent"].queue.qsize() == 1

        # Check notification was sent
        mock_connection.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_message_to_unknown_agent(self, message_broker):
        """Test routing to unknown agent"""
        message = A2AMessage(
            message_id="test_msg",
            sender_id="sender",
            receiver_id="unknown_agent",
            message_type="test",
            content={}
        )

        success = await message_broker.route_message(message)

        assert success == False

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, message_broker):
        """Test agent subscription"""
        # Subscribe agent to topic
        await message_broker.subscribe_agent("test_agent", "finance_updates")

        assert "test_agent" in message_broker.agent_subscriptions["finance_updates"]

        # Unsubscribe
        await message_broker.unsubscribe_agent("test_agent", "finance_updates")

        assert "test_agent" not in message_broker.agent_subscriptions["finance_updates"]

    @pytest.mark.asyncio
    async def test_get_messages(self, message_broker, mock_connection):
        """Test getting messages for agent"""
        await message_broker.register_agent("test_agent", mock_connection)

        # Add messages
        for i in range(3):
            message = A2AMessage(
                message_id=f"msg_{i}",
                sender_id="sender",
                receiver_id="test_agent",
                message_type="test",
                content={"index": i}
            )
            await message_broker.route_message(message)

        # Get messages with limit
        messages = await message_broker.get_messages("test_agent", limit=2)

        assert len(messages) == 2
        assert messages[0].message_id == "msg_0"
        assert messages[1].message_id == "msg_1"

    @pytest.mark.asyncio
    async def test_broadcast_message(self, message_broker, mock_connection):
        """Test message broadcasting"""
        # Register multiple agents
        connections = {}
        for i in range(3):
            agent_id = f"agent_{i}"
            connection = AsyncMock()
            connections[agent_id] = connection
            await message_broker.register_agent(agent_id, connection)

        # Create broadcast message
        broadcast_message = A2AMessage(
            message_id="broadcast_msg",
            sender_id="broadcaster",
            receiver_id="broadcast",
            message_type="broadcast",
            content={"announcement": "test"}
        )

        # Route broadcast
        await message_broker.route_message(broadcast_message)

        # Check all agents received broadcast
        for agent_id, connection in connections.items():
            if agent_id != "broadcaster":  # Broadcaster shouldn't receive its own broadcast
                # Each agent should have a broadcast message
                queue = message_broker.agent_queues[agent_id]
                assert queue.queue.qsize() == 1

    def test_get_stats(self, message_broker):
        """Test getting broker statistics"""
        stats = message_broker.get_stats()

        assert "total_agents" in stats
        assert "connected_agents" in stats
        assert "total_messages_logged" in stats
        assert "queue_sizes" in stats
        assert "subscriptions" in stats


class TestA2AClient:
    """Test A2A Client functionality"""

    @pytest.fixture
    def a2a_client(self):
        """Create A2A client instance"""
        return A2AClient("ws://localhost:8001", "test_client")

    @pytest.mark.asyncio
    async def test_connect_and_close(self, a2a_client):
        """Test client connection and closure"""
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            mock_websocket.recv = AsyncMock(side_effect=asyncio.CancelledError)  # Stop listener

            # Connect
            await a2a_client.connect()

            assert a2a_client.connected == True
            assert a2a_client.websocket == mock_websocket

            # Close
            await a2a_client.close()

            assert a2a_client.connected == False
            mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message(self, a2a_client):
        """Test sending message"""
        with patch.object(a2a_client, 'websocket', new_callable=AsyncMock) as mock_websocket:
            a2a_client.connected = True
            mock_websocket.send = AsyncMock()

            # Send message
            response = await a2a_client.send_message(
                receiver_id="receiver_agent",
                message_type="test_message",
                content={"data": "test"},
                expect_response=False
            )

            assert response is None
            mock_websocket.send.assert_called_once()

            # Check message format
            sent_data = json.loads(mock_websocket.send.call_args[0][0])
            assert sent_data["sender_id"] == "test_client"
            assert sent_data["receiver_id"] == "receiver_agent"
            assert sent_data["message_type"] == "test_message"

    @pytest.mark.asyncio
    async def test_send_message_with_response(self, a2a_client):
        """Test sending message and waiting for response"""
        with patch.object(a2a_client, 'websocket', new_callable=AsyncMock) as mock_websocket:
            a2a_client.connected = True

            # Mock send
            mock_websocket.send = AsyncMock()

            # Mock receiving response in listener
            response_data = {
                "message_id": "response_001",
                "sender_id": "receiver_agent",
                "receiver_id": "test_client",
                "message_type": "response",
                "content": {"result": "success"},
                "correlation_id": "test_correlation"
            }

            # Create future for response
            import asyncio
            response_future = asyncio.Future()
            response_future.set_result(response_data)
            a2a_client.pending_responses["test_correlation"] = response_future

            # Send message
            response = await a2a_client.send_message(
                receiver_id="receiver_agent",
                message_type="test_message",
                content={"data": "test"},
                correlation_id="test_correlation",
                timeout=5
            )

            assert response == response_data
            assert "test_correlation" not in a2a_client.pending_responses

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, a2a_client):
        """Test message send timeout"""
        with patch.object(a2a_client, 'websocket', new_callable=AsyncMock) as mock_websocket:
            a2a_client.connected = True
            mock_websocket.send = AsyncMock()

            # Send message with short timeout
            with pytest.raises(TimeoutError):
                await a2a_client.send_message(
                    receiver_id="receiver_agent",
                    message_type="test_message",
                    content={},
                    timeout=0.1  # Very short timeout
                )

    @pytest.mark.asyncio
    async def test_broadcast(self, a2a_client):
        """Test broadcast functionality"""
        with patch.object(a2a_client, 'send_message') as mock_send:
            mock_send.return_value = {"status": "sent"}

            with patch.object(a2a_client, '_get_available_agents') as mock_get_agents:
                mock_get_agents.return_value = ["agent_1", "agent_2", "test_client"]

                # Broadcast
                responses = await a2a_client.broadcast(
                    message_type="announcement",
                    content={"msg": "hello"},
                    agent_filter=None
                )

                # Should send to 2 agents (excluding self)
                assert mock_send.call_count == 2
                assert len(responses) == 2

    def test_register_handler(self, a2a_client):
        """Test handler registration"""

        def test_handler(message):
            pass

        a2a_client.register_handler("test_message", test_handler)

        assert "test_message" in a2a_client.message_handlers
        assert a2a_client.message_handlers["test_message"] == test_handler


class TestA2AServer:
    """Test A2A Server functionality"""

    @pytest.fixture
    def a2a_server(self):
        """Create A2A server instance"""
        return A2AServer()

    @pytest.mark.asyncio
    async def test_websocket_connection(self, a2a_server):
        """Test WebSocket connection handling"""
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=asyncio.CancelledError)  # Simulate disconnect

        # Mock the message broker
        a2a_server.message_broker = AsyncMock()
        a2a_server.message_broker.register_agent = AsyncMock()
        a2a_server.message_broker.unregister_agent = AsyncMock()

        # Simulate WebSocket connection
        try:
            await a2a_server.websocket_endpoint(mock_websocket, "test_agent")
        except asyncio.CancelledError:
            pass

        # Check agent was registered and unregistered
        a2a_server.message_broker.register_agent.assert_called_once_with("test_agent", mock_websocket)
        a2a_server.message_broker.unregister_agent.assert_called_once_with("test_agent")

    @pytest.mark.asyncio
    async def test_process_message(self, a2a_server):
        """Test message processing"""
        # Mock message broker
        a2a_server.message_broker = AsyncMock()
        a2a_server.message_broker.route_message = AsyncMock(return_value=True)

        # Test message
        message = {
            "message_type": "test_message",
            "receiver_id": "receiver_agent",
            "content": {"data": "test"}
        }

        await a2a_server._process_message("sender_agent", message)

        # Check message was routed
        a2a_server.message_broker.route_message.assert_called_once()
        routed_message = a2a_server.message_broker.route_message.call_args[0][0]
        assert routed_message.sender_id == "sender_agent"
        assert routed_message.receiver_id == "receiver_agent"

    @pytest.mark.asyncio
    async def test_send_to_agent(self, a2a_server):
        """Test sending message to specific agent"""
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        a2a_server.active_connections["test_agent"] = mock_websocket

        # Send message
        await a2a_server.send_to_agent("test_agent", {"type": "test"})

        mock_websocket.send_text.assert_called_once_with(json.dumps({"type": "test"}))

    @pytest.mark.asyncio
    async def test_send_to_unknown_agent(self, a2a_server):
        """Test sending to unknown agent"""
        # Should log warning but not crash
        await a2a_server.send_to_agent("unknown_agent", {"type": "test"})

        # No assertion needed, just shouldn't crash


class TestAgentRegistry:
    """Test Agent Registry functionality"""

    @pytest.fixture
    def agent_repository(self):
        """Create agent repository instance"""
        return InMemoryAgentRepository()

    @pytest.fixture
    def sample_agent(self):
        """Create sample agent"""
        from domain.models.agent import Agent, AgentCapabilities, AgentStatus, AgentCapability

        capabilities = AgentCapabilities(
            capabilities=[AgentCapability.FINANCIAL_ANALYSIS, AgentCapability.RISK_ASSESSMENT]
        )

        return Agent(
            agent_id="test_agent_001",
            agent_name="Test Agent",
            agent_type="finance",
            capabilities=capabilities,
            status=AgentStatus.ACTIVE
        )

    @pytest.mark.asyncio
    async def test_save_and_find_agent(self, agent_repository, sample_agent):
        """Test saving and finding agent"""
        # Save agent
        success = await agent_repository.save(sample_agent)
        assert success == True

        # Find by ID
        found = await agent_repository.find_by_id(sample_agent.agent_id)
        assert found is not None
        assert found.agent_id == sample_agent.agent_id
        assert found.agent_name == sample_agent.agent_name

    @pytest.mark.asyncio
    async def test_find_by_capability(self, agent_repository, sample_agent):
        """Test finding agents by capability"""
        from domain.models.agent import AgentCapability

        # Save agent
        await agent_repository.save(sample_agent)

        # Find by capability
        agents = await agent_repository.find_by_capability(AgentCapability.FINANCIAL_ANALYSIS)

        assert len(agents) == 1
        assert agents[0].agent_id == sample_agent.agent_id

    @pytest.mark.asyncio
    async def test_update_status(self, agent_repository, sample_agent):
        """Test updating agent status"""
        from domain.models.agent import AgentStatus

        # Save agent
        await agent_repository.save(sample_agent)

        # Update status
        success = await agent_repository.update_status(sample_agent.agent_id, AgentStatus.BUSY)
        assert success == True

        # Check status updated
        agent = await agent_repository.find_by_id(sample_agent.agent_id)
        assert agent.status == AgentStatus.BUSY

        # Find by status
        busy_agents = await agent_repository.find_by_status(AgentStatus.BUSY)
        assert len(busy_agents) == 1
        assert busy_agents[0].agent_id == sample_agent.agent_id

    @pytest.mark.asyncio
    async def test_delete_agent(self, agent_repository, sample_agent):
        """Test deleting agent"""
        # Save agent
        await agent_repository.save(sample_agent)

        # Delete agent
        success = await agent_repository.delete(sample_agent.agent_id)
        assert success == True

        # Should not find deleted agent
        found = await agent_repository.find_by_id(sample_agent.agent_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_count_agents(self, agent_repository, sample_agent):
        """Test counting agents"""
        # Initial count
        count = await agent_repository.count()
        assert count == 0

        # Save agent
        await agent_repository.save(sample_agent)

        # New count
        count = await agent_repository.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_find_inactive_agents(self, agent_repository, sample_agent):
        """Test finding inactive agents"""
        from datetime import datetime, timedelta

        # Save agent with old heartbeat
        sample_agent.last_heartbeat = datetime.utcnow() - timedelta(minutes=10)
        await agent_repository.save(sample_agent)

        # Find inactive agents
        inactive = await agent_repository.find_inactive_agents(timeout_minutes=5)

        assert len(inactive) == 1
        assert inactive[0].agent_id == sample_agent.agent_id

    @pytest.mark.asyncio
    async def test_get_capability_stats(self, agent_repository, sample_agent):
        """Test getting capability statistics"""
        from domain.models.agent import AgentCapability

        # Save agent
        await agent_repository.save(sample_agent)

        # Get stats
        stats = await agent_repository.get_capability_stats()

        assert AgentCapability.FINANCIAL_ANALYSIS in stats
        assert stats[AgentCapability.FINANCIAL_ANALYSIS] == 1

        assert AgentCapability.RISK_ASSESSMENT in stats
        assert stats[AgentCapability.RISK_ASSESSMENT] == 1


@pytest.mark.integration
class TestA2AIntegration:
    """Integration tests for A2A protocol"""

    @pytest.mark.asyncio
    async def test_complete_message_flow(self):
        """Test complete message flow between client and server"""
        # Create server
        server = A2AServer()

        # Create mock clients
        client1_ws = AsyncMock()
        client2_ws = AsyncMock()

        client1_ws.send_text = AsyncMock()
        client2_ws.send_text = AsyncMock()

        # Register agents
        await server.message_broker.register_agent("agent_1", client1_ws)
        await server.message_broker.register_agent("agent_2", client2_ws)

        # Create message from agent_1 to agent_2
        message = A2AMessage(
            message_id="test_flow_msg",
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type="test_message",
            content={"data": "test"}
        )

        # Route message
        success = await server.message_broker.route_message(message)
        assert success == True

        # Check agent_2's queue
        messages = await server.message_broker.get_messages("agent_2")
        assert len(messages) == 1
        assert messages[0].message_id == "test_flow_msg"

        # Check notification was sent
        client2_ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_flow(self):
        """Test broadcast message flow"""
        server = A2AServer()

        # Create multiple mock clients
        clients = {}
        for i in range(3):
            agent_id = f"agent_{i}"
            ws = AsyncMock()
            ws.send_text = AsyncMock()
            clients[agent_id] = ws
            await server.message_broker.register_agent(agent_id, ws)

        # Create broadcast message
        broadcast_msg = A2AMessage(
            message_id="broadcast_test",
            sender_id="agent_0",
            receiver_id="broadcast",
            message_type="announcement",
            content={"message": "Hello all"}
        )

        # Route broadcast
        await server.message_broker.route_message(broadcast_msg)

        # Check all agents (except sender) got broadcast message
        for agent_id, ws in clients.items():
            if agent_id != "agent_0":  # Sender doesn't receive its own broadcast
                # Get messages for this agent
                messages = await server.message_broker.get_messages(agent_id)
                assert len(messages) == 1
                assert messages[0].message_id.startswith("broadcast_")

    @pytest.mark.asyncio
    async def test_subscription_flow(self):
        """Test subscription-based message flow"""
        server = A2AServer()

        # Create clients
        subscriber_ws = AsyncMock()
        publisher_ws = AsyncMock()

        await server.message_broker.register_agent("subscriber", subscriber_ws)
        await server.message_broker.register_agent("publisher", publisher_ws)

        # Subscribe to topic
        await server.message_broker.subscribe_agent("subscriber", "market_updates")

        # Create topic message
        topic_msg = A2AMessage(
            message_id="topic_msg",
            sender_id="publisher",
            receiver_id="market_updates",  # Send to topic
            message_type="market_update",
            content={"price": 150.0}
        )

        # This won't directly route to subscriber since we're using receiver_id as topic
        # In real implementation, you'd have a different mechanism for topic routing

        # For now, just test subscription registration
        assert "subscriber" in server.message_broker.agent_subscriptions["market_updates"]


@pytest.mark.performance
class TestA2APerformance:
    """Performance tests for A2A protocol"""

    @pytest.mark.asyncio
    async def test_message_routing_performance(self):
        """Test message routing performance"""
        broker = MessageBroker()

        # Register many agents
        num_agents = 100
        for i in range(num_agents):
            ws = AsyncMock()
            ws.send_text = AsyncMock()
            await broker.register_agent(f"agent_{i}", ws)

        import time

        # Measure routing performance
        num_messages = 1000
        start_time = time.perf_counter()

        for i in range(num_messages):
            sender = f"agent_{i % num_agents}"
            receiver = f"agent_{(i + 1) % num_agents}"

            message = A2AMessage(
                message_id=f"perf_msg_{i}",
                sender_id=sender,
                receiver_id=receiver,
                message_type="performance_test",
                content={"index": i}
            )

            await broker.route_message(message)

        end_time = time.perf_counter()

        total_time = end_time - start_time
        messages_per_second = num_messages / total_time

        print(f"\nA2A Message Routing Performance:")
        print(f"  Messages routed: {num_messages}")
        print(f"  Agents: {num_agents}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Messages per second: {messages_per_second:.2f}")

        assert messages_per_second > 100  # Should handle at least 100 msg/sec

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        """Test concurrent message processing"""
        broker = MessageBroker()

        # Register agents
        num_agents = 10
        agents = []
        for i in range(num_agents):
            ws = AsyncMock()
            ws.send_text = AsyncMock()
            agents.append((f"agent_{i}", ws))
            await broker.register_agent(f"agent_{i}", ws)

        # Create concurrent message sending tasks
        async def send_messages(agent_id, num_messages):
            for i in range(num_messages):
                receiver = f"agent_{(int(agent_id.split('_')[1]) + 1) % num_agents}"

                message = A2AMessage(
                    message_id=f"concurrent_{agent_id}_{i}",
                    sender_id=agent_id,
                    receiver_id=receiver,
                    message_type="concurrent_test",
                    content={"from": agent_id, "index": i}
                )

                await broker.route_message(message)

        # Run concurrent
        num_messages_per_agent = 10
        start_time = time.perf_counter()

        tasks = [send_messages(agent_id, num_messages_per_agent) for agent_id, _ in agents]
        await asyncio.gather(*tasks)

        end_time = time.perf_counter()

        total_messages = num_agents * num_messages_per_agent
        total_time = end_time - start_time

        print(f"\nConcurrent A2A Processing:")
        print(f"  Agents: {num_agents}")
        print(f"  Messages per agent: {num_messages_per_agent}")
        print(f"  Total messages: {total_messages}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Messages per second: {total_messages / total_time:.2f}")

        # Check all messages were processed
        total_queued = sum(q.queue.qsize() for q in broker.agent_queues.values())
        assert total_queued == total_messages