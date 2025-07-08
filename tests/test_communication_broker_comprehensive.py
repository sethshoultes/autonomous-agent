"""
Comprehensive tests for communication broker.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.communication.broker import MessageQueue, MessageHandler, MessageRouter, MessageBroker
from src.agents.base import AgentMessage
from src.agents.exceptions import CommunicationError


class TestMessageQueue:
    """Test MessageQueue functionality."""
    
    @pytest.fixture
    def queue(self):
        """Create a MessageQueue instance."""
        return MessageQueue(max_size=5, priority_queue=False)
    
    @pytest.fixture
    def priority_queue(self):
        """Create a priority MessageQueue instance."""
        return MessageQueue(max_size=5, priority_queue=True)
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return AgentMessage(
            id="test_msg_1",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "test"}
        )
    
    def test_message_queue_initialization(self, queue):
        """Test MessageQueue initialization."""
        assert queue.max_size == 5
        assert queue.priority_queue is False
        assert queue.size == 0
        assert queue.is_empty is True
        assert queue.is_full is False
        assert queue._closed is False
    
    def test_priority_queue_initialization(self, priority_queue):
        """Test priority MessageQueue initialization."""
        assert priority_queue.priority_queue is True
    
    def test_queue_default_initialization(self):
        """Test MessageQueue with default parameters."""
        queue = MessageQueue()
        assert queue.max_size == 1000
        assert queue.priority_queue is False
    
    @pytest.mark.asyncio
    async def test_put_and_get_message(self, queue, sample_message):
        """Test putting and getting a message."""
        await queue.put(sample_message)
        
        assert queue.size == 1
        assert not queue.is_empty
        
        retrieved_message = await queue.get()
        assert retrieved_message == sample_message
        assert queue.size == 0
        assert queue.is_empty
    
    @pytest.mark.asyncio
    async def test_put_multiple_messages(self, queue):
        """Test putting multiple messages."""
        messages = []
        for i in range(3):
            msg = AgentMessage(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="agent_2",
                message_type="test",
                payload={"data": f"test_{i}"}
            )
            messages.append(msg)
            await queue.put(msg)
        
        assert queue.size == 3
        
        # Retrieve messages in FIFO order
        for i in range(3):
            retrieved = await queue.get()
            assert retrieved.id == f"msg_{i}"
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, priority_queue):
        """Test priority queue message ordering."""
        # Create messages with different priorities
        high_priority = AgentMessage(
            id="high",
            sender="agent_1",
            recipient="agent_2",
            message_type="urgent",
            payload={"data": "urgent"},
            priority=1
        )
        
        low_priority = AgentMessage(
            id="low",
            sender="agent_1",
            recipient="agent_2",
            message_type="normal",
            payload={"data": "normal"},
            priority=5
        )
        
        medium_priority = AgentMessage(
            id="medium",
            sender="agent_1",
            recipient="agent_2",
            message_type="important",
            payload={"data": "important"},
            priority=3
        )
        
        # Put in order: low, high, medium
        await priority_queue.put(low_priority)
        await priority_queue.put(high_priority)
        await priority_queue.put(medium_priority)
        
        # Should retrieve in priority order: high, medium, low
        first = await priority_queue.get()
        assert first.id == "high"
        
        second = await priority_queue.get()
        assert second.id == "medium"
        
        third = await priority_queue.get()
        assert third.id == "low"
    
    @pytest.mark.asyncio
    async def test_queue_full_condition(self, queue):
        """Test queue full condition."""
        # Fill the queue to capacity
        for i in range(5):
            msg = AgentMessage(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="agent_2",
                message_type="test",
                payload={"data": f"test_{i}"}
            )
            await queue.put(msg)
        
        assert queue.is_full
        assert queue.size == 5
        
        # Trying to put another message should raise exception
        overflow_msg = AgentMessage(
            id="overflow",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "overflow"}
        )
        
        with pytest.raises(CommunicationError):
            await queue.put(overflow_msg)
    
    @pytest.mark.asyncio
    async def test_put_nowait_success(self, queue, sample_message):
        """Test put_nowait when queue has space."""
        queue.put_nowait(sample_message)
        assert queue.size == 1
        
        retrieved = await queue.get()
        assert retrieved == sample_message
    
    @pytest.mark.asyncio
    async def test_put_nowait_full_queue(self, queue):
        """Test put_nowait when queue is full."""
        # Fill the queue
        for i in range(5):
            msg = AgentMessage(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="agent_2",
                message_type="test",
                payload={"data": f"test_{i}"}
            )
            queue.put_nowait(msg)
        
        # Trying to put another message should raise exception
        overflow_msg = AgentMessage(
            id="overflow",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "overflow"}
        )
        
        with pytest.raises(CommunicationError):
            queue.put_nowait(overflow_msg)
    
    @pytest.mark.asyncio
    async def test_get_nowait_success(self, queue, sample_message):
        """Test get_nowait when queue has messages."""
        await queue.put(sample_message)
        
        retrieved = queue.get_nowait()
        assert retrieved == sample_message
        assert queue.is_empty
    
    @pytest.mark.asyncio
    async def test_get_nowait_empty_queue(self, queue):
        """Test get_nowait when queue is empty."""
        with pytest.raises(CommunicationError):
            queue.get_nowait()
    
    @pytest.mark.asyncio
    async def test_peek_message(self, queue, sample_message):
        """Test peeking at next message without removing it."""
        await queue.put(sample_message)
        
        peeked = queue.peek()
        assert peeked == sample_message
        assert queue.size == 1  # Message should still be in queue
        
        retrieved = await queue.get()
        assert retrieved == sample_message
    
    @pytest.mark.asyncio
    async def test_peek_empty_queue(self, queue):
        """Test peeking at empty queue."""
        peeked = queue.peek()
        assert peeked is None
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, queue):
        """Test clearing the queue."""
        # Add some messages
        for i in range(3):
            msg = AgentMessage(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="agent_2",
                message_type="test",
                payload={"data": f"test_{i}"}
            )
            await queue.put(msg)
        
        assert queue.size == 3
        
        queue.clear()
        assert queue.size == 0
        assert queue.is_empty
    
    @pytest.mark.asyncio
    async def test_close_queue(self, queue, sample_message):
        """Test closing the queue."""
        await queue.put(sample_message)
        
        queue.close()
        assert queue._closed is True
        
        # Should not be able to put more messages
        new_msg = AgentMessage(
            id="new_msg",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "new"}
        )
        
        with pytest.raises(CommunicationError):
            await queue.put(new_msg)
        
        # Should still be able to get existing messages
        retrieved = await queue.get()
        assert retrieved == sample_message
    
    @pytest.mark.asyncio
    async def test_queue_statistics(self, queue):
        """Test queue statistics."""
        stats = queue.get_statistics()
        
        expected_keys = ["size", "max_size", "is_empty", "is_full", "is_closed", "priority_queue"]
        for key in expected_keys:
            assert key in stats
        
        assert stats["size"] == 0
        assert stats["max_size"] == 5
        assert stats["is_empty"] is True
        assert stats["is_full"] is False
        assert stats["is_closed"] is False
        assert stats["priority_queue"] is False
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, queue):
        """Test concurrent access to queue."""
        async def producer():
            for i in range(10):
                msg = AgentMessage(
                    id=f"msg_{i}",
                    sender="producer",
                    recipient="consumer",
                    message_type="test",
                    payload={"data": f"test_{i}"}
                )
                await queue.put(msg)
                await asyncio.sleep(0.001)  # Small delay
        
        async def consumer():
            messages = []
            for _ in range(10):
                msg = await queue.get()
                messages.append(msg)
                await asyncio.sleep(0.001)  # Small delay
            return messages
        
        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())
        
        await asyncio.gather(producer_task)
        consumed_messages = await consumer_task
        
        assert len(consumed_messages) == 10
        assert queue.is_empty


class TestMessageHandler:
    """Test MessageHandler functionality."""
    
    @pytest.fixture
    def handler(self):
        """Create a MessageHandler instance."""
        return MessageHandler(max_concurrent=3, timeout=1.0)
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return AgentMessage(
            id="test_msg",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "test"}
        )
    
    def test_message_handler_initialization(self, handler):
        """Test MessageHandler initialization."""
        assert handler.max_concurrent == 3
        assert handler.timeout == 1.0
        assert handler.handlers == {}
        assert handler.active_tasks == set()
    
    def test_message_handler_default_initialization(self):
        """Test MessageHandler with default parameters."""
        handler = MessageHandler()
        assert handler.max_concurrent == 10
        assert handler.timeout == 30.0
    
    def test_register_handler(self, handler):
        """Test registering a message handler."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        handler.register_handler("test", test_handler)
        assert "test" in handler.handlers
        assert handler.handlers["test"] == test_handler
    
    def test_unregister_handler(self, handler):
        """Test unregistering a message handler."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        handler.register_handler("test", test_handler)
        assert "test" in handler.handlers
        
        success = handler.unregister_handler("test")
        assert success is True
        assert "test" not in handler.handlers
    
    def test_unregister_nonexistent_handler(self, handler):
        """Test unregistering non-existent handler."""
        success = handler.unregister_handler("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_handle_message_success(self, handler, sample_message):
        """Test successful message handling."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        handler.register_handler("test", test_handler)
        
        result = await handler.handle_message(sample_message)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_handle_message_no_handler(self, handler, sample_message):
        """Test handling message with no registered handler."""
        result = await handler.handle_message(sample_message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_handle_message_handler_exception(self, handler, sample_message):
        """Test handling message when handler raises exception."""
        async def failing_handler(message):
            raise Exception("Handler failed")
        
        handler.register_handler("test", failing_handler)
        
        result = await handler.handle_message(sample_message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_handle_message_timeout(self, handler, sample_message):
        """Test handling message with timeout."""
        async def slow_handler(message):
            await asyncio.sleep(2.0)  # Longer than timeout
            return "Too slow"
        
        handler.register_handler("test", slow_handler)
        
        result = await handler.handle_message(sample_message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, handler):
        """Test concurrent message handling with limits."""
        handled_messages = []
        
        async def slow_handler(message):
            await asyncio.sleep(0.1)
            handled_messages.append(message.id)
            return f"Handled: {message.id}"
        
        handler.register_handler("test", slow_handler)
        
        # Create multiple messages
        messages = []
        for i in range(5):
            msg = AgentMessage(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="agent_2",
                message_type="test",
                payload={"data": f"test_{i}"}
            )
            messages.append(msg)
        
        # Handle all messages concurrently
        tasks = [handler.handle_message(msg) for msg in messages]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)
        assert len(handled_messages) == 5
    
    def test_get_registered_handlers(self, handler):
        """Test getting list of registered handlers."""
        assert handler.get_registered_handlers() == []
        
        async def handler1(message):
            pass
        
        async def handler2(message):
            pass
        
        handler.register_handler("type1", handler1)
        handler.register_handler("type2", handler2)
        
        registered = handler.get_registered_handlers()
        assert set(registered) == {"type1", "type2"}
    
    def test_get_statistics(self, handler):
        """Test getting handler statistics."""
        stats = handler.get_statistics()
        
        expected_keys = ["max_concurrent", "timeout", "registered_handlers", "active_tasks"]
        for key in expected_keys:
            assert key in stats
        
        assert stats["max_concurrent"] == 3
        assert stats["timeout"] == 1.0
        assert stats["registered_handlers"] == 0
        assert stats["active_tasks"] == 0


class TestMessageRouter:
    """Test MessageRouter functionality."""
    
    @pytest.fixture
    def router(self):
        """Create a MessageRouter instance."""
        return MessageRouter()
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return AgentMessage(
            id="test_msg",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "test"}
        )
    
    def test_message_router_initialization(self, router):
        """Test MessageRouter initialization."""
        assert router.routes == {}
        assert router.default_handler is None
    
    def test_register_route(self, router):
        """Test registering a route."""
        mock_queue = Mock()
        router.register_route("agent_1", mock_queue)
        
        assert "agent_1" in router.routes
        assert router.routes["agent_1"] == mock_queue
    
    def test_unregister_route(self, router):
        """Test unregistering a route."""
        mock_queue = Mock()
        router.register_route("agent_1", mock_queue)
        
        success = router.unregister_route("agent_1")
        assert success is True
        assert "agent_1" not in router.routes
    
    def test_unregister_nonexistent_route(self, router):
        """Test unregistering non-existent route."""
        success = router.unregister_route("nonexistent")
        assert success is False
    
    def test_set_default_handler(self, router):
        """Test setting default handler."""
        mock_handler = Mock()
        router.set_default_handler(mock_handler)
        assert router.default_handler == mock_handler
    
    @pytest.mark.asyncio
    async def test_route_message_success(self, router, sample_message):
        """Test successful message routing."""
        mock_queue = Mock()
        mock_queue.put = AsyncMock()
        
        router.register_route("agent_2", mock_queue)  # recipient
        
        result = await router.route_message(sample_message)
        assert result is True
        mock_queue.put.assert_called_once_with(sample_message)
    
    @pytest.mark.asyncio
    async def test_route_message_no_route(self, router, sample_message):
        """Test routing message with no registered route."""
        result = await router.route_message(sample_message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_route_message_default_handler(self, router, sample_message):
        """Test routing message using default handler."""
        mock_handler = AsyncMock(return_value=True)
        router.set_default_handler(mock_handler)
        
        result = await router.route_message(sample_message)
        assert result is True
        mock_handler.assert_called_once_with(sample_message)
    
    @pytest.mark.asyncio
    async def test_route_message_queue_error(self, router, sample_message):
        """Test routing message when queue raises error."""
        mock_queue = Mock()
        mock_queue.put = AsyncMock(side_effect=Exception("Queue error"))
        
        router.register_route("agent_2", mock_queue)
        
        result = await router.route_message(sample_message)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_route_message_default_handler_error(self, router, sample_message):
        """Test routing message when default handler raises error."""
        mock_handler = AsyncMock(side_effect=Exception("Handler error"))
        router.set_default_handler(mock_handler)
        
        result = await router.route_message(sample_message)
        assert result is False
    
    def test_get_routes(self, router):
        """Test getting registered routes."""
        assert router.get_routes() == []
        
        mock_queue1 = Mock()
        mock_queue2 = Mock()
        
        router.register_route("agent_1", mock_queue1)
        router.register_route("agent_2", mock_queue2)
        
        routes = router.get_routes()
        assert set(routes) == {"agent_1", "agent_2"}
    
    def test_get_statistics(self, router):
        """Test getting router statistics."""
        stats = router.get_statistics()
        
        expected_keys = ["registered_routes", "has_default_handler"]
        for key in expected_keys:
            assert key in stats
        
        assert stats["registered_routes"] == 0
        assert stats["has_default_handler"] is False


class TestMessageBroker:
    """Test MessageBroker functionality."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()
    
    @pytest.fixture
    def broker(self, mock_logger):
        """Create a MessageBroker instance."""
        return MessageBroker(
            max_queue_size=10,
            max_concurrent_handlers=2,
            logger=mock_logger
        )
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return AgentMessage(
            id="test_msg",
            sender="agent_1",
            recipient="agent_2",
            message_type="test",
            payload={"data": "test"}
        )
    
    def test_message_broker_initialization(self, broker):
        """Test MessageBroker initialization."""
        assert broker.max_queue_size == 10
        assert broker.logger is not None
        assert broker.queues == {}
        assert broker.router is not None
        assert broker.handler is not None
        assert broker.running is False
    
    def test_message_broker_default_initialization(self):
        """Test MessageBroker with default parameters."""
        broker = MessageBroker()
        assert broker.max_queue_size == 1000
    
    @pytest.mark.asyncio
    async def test_start_broker(self, broker):
        """Test starting the broker."""
        await broker.start()
        assert broker.running is True
        
        # Clean up
        await broker.stop()
    
    @pytest.mark.asyncio
    async def test_stop_broker(self, broker):
        """Test stopping the broker."""
        await broker.start()
        await broker.stop()
        assert broker.running is False
    
    @pytest.mark.asyncio
    async def test_register_agent(self, broker):
        """Test registering an agent."""
        await broker.register_agent("agent_1")
        
        assert "agent_1" in broker.queues
        assert broker.queues["agent_1"] is not None
    
    @pytest.mark.asyncio
    async def test_register_agent_duplicate(self, broker):
        """Test registering duplicate agent."""
        await broker.register_agent("agent_1")
        first_queue = broker.queues["agent_1"]
        
        await broker.register_agent("agent_1")  # Register again
        
        # Should not create new queue
        assert broker.queues["agent_1"] is first_queue
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self, broker):
        """Test unregistering an agent."""
        await broker.register_agent("agent_1")
        assert "agent_1" in broker.queues
        
        success = await broker.unregister_agent("agent_1")
        assert success is True
        assert "agent_1" not in broker.queues
    
    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent(self, broker):
        """Test unregistering non-existent agent."""
        success = await broker.unregister_agent("nonexistent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_send_message(self, broker, sample_message):
        """Test sending a message."""
        await broker.register_agent("agent_2")  # recipient
        
        success = await broker.send_message(sample_message)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_send_message_no_recipient(self, broker, sample_message):
        """Test sending message to unregistered recipient."""
        success = await broker.send_message(sample_message)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_receive_message(self, broker, sample_message):
        """Test receiving a message."""
        await broker.register_agent("agent_2")
        await broker.send_message(sample_message)
        
        received = await broker.receive_message("agent_2")
        assert received == sample_message
    
    @pytest.mark.asyncio
    async def test_receive_message_timeout(self, broker):
        """Test receiving message with timeout."""
        await broker.register_agent("agent_1")
        
        received = await broker.receive_message("agent_1", timeout=0.1)
        assert received is None
    
    @pytest.mark.asyncio
    async def test_receive_message_unregistered_agent(self, broker):
        """Test receiving message for unregistered agent."""
        received = await broker.receive_message("nonexistent")
        assert received is None
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, broker):
        """Test broadcasting a message."""
        # Register multiple agents
        await broker.register_agent("agent_1")
        await broker.register_agent("agent_2")
        await broker.register_agent("agent_3")
        
        broadcast_msg = AgentMessage(
            id="broadcast",
            sender="system",
            recipient="all",
            message_type="announcement",
            payload={"message": "Hello everyone"}
        )
        
        recipients = ["agent_1", "agent_2", "agent_3"]
        success = await broker.broadcast_message(broadcast_msg, recipients)
        assert success is True
        
        # Check that all agents received the message
        for agent_id in recipients:
            received = await broker.receive_message(agent_id)
            assert received.id == "broadcast"
    
    @pytest.mark.asyncio
    async def test_broadcast_message_partial_failure(self, broker):
        """Test broadcasting when some recipients are unregistered."""
        await broker.register_agent("agent_1")
        # agent_2 is not registered
        
        broadcast_msg = AgentMessage(
            id="broadcast",
            sender="system",
            recipient="all",
            message_type="announcement",
            payload={"message": "Hello everyone"}
        )
        
        recipients = ["agent_1", "agent_2"]  # agent_2 doesn't exist
        success = await broker.broadcast_message(broadcast_msg, recipients)
        assert success is False  # Should fail because not all recipients exist
    
    def test_register_message_handler(self, broker):
        """Test registering a message handler."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        broker.register_message_handler("test", test_handler)
        registered = broker.handler.get_registered_handlers()
        assert "test" in registered
    
    def test_unregister_message_handler(self, broker):
        """Test unregistering a message handler."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        broker.register_message_handler("test", test_handler)
        success = broker.unregister_message_handler("test")
        assert success is True
        
        registered = broker.handler.get_registered_handlers()
        assert "test" not in registered
    
    def test_get_registered_agents(self, broker):
        """Test getting registered agents."""
        assert broker.get_registered_agents() == []
    
    @pytest.mark.asyncio
    async def test_get_registered_agents_with_agents(self, broker):
        """Test getting registered agents after registration."""
        await broker.register_agent("agent_1")
        await broker.register_agent("agent_2")
        
        agents = broker.get_registered_agents()
        assert set(agents) == {"agent_1", "agent_2"}
    
    def test_get_statistics(self, broker):
        """Test getting broker statistics."""
        stats = broker.get_statistics()
        
        expected_keys = [
            "registered_agents", "total_queues", "running",
            "handler_stats", "router_stats"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["registered_agents"] == 0
        assert stats["total_queues"] == 0
        assert stats["running"] is False


class TestMessageBrokerIntegration:
    """Test integration scenarios for message broker."""
    
    @pytest.mark.asyncio
    async def test_full_broker_lifecycle(self):
        """Test complete broker lifecycle."""
        broker = MessageBroker(max_queue_size=5, max_concurrent_handlers=2)
        
        try:
            # Start broker
            await broker.start()
            assert broker.running is True
            
            # Register agents
            await broker.register_agent("agent_1")
            await broker.register_agent("agent_2")
            
            # Send message
            message = AgentMessage(
                id="test_msg",
                sender="agent_1",
                recipient="agent_2",
                message_type="greeting",
                payload={"message": "Hello"}
            )
            
            success = await broker.send_message(message)
            assert success is True
            
            # Receive message
            received = await broker.receive_message("agent_2")
            assert received.id == "test_msg"
            
            # Unregister agents
            await broker.unregister_agent("agent_1")
            await broker.unregister_agent("agent_2")
            
            assert len(broker.get_registered_agents()) == 0
            
        finally:
            await broker.stop()
    
    @pytest.mark.asyncio
    async def test_message_handling_with_custom_handler(self):
        """Test message handling with custom handlers."""
        broker = MessageBroker()
        
        handled_messages = []
        
        async def custom_handler(message):
            handled_messages.append(message.id)
            return True
        
        try:
            await broker.start()
            
            # Register handler
            broker.register_message_handler("greeting", custom_handler)
            
            # Register agent
            await broker.register_agent("agent_1")
            
            # Send message
            message = AgentMessage(
                id="greeting_msg",
                sender="system",
                recipient="agent_1",
                message_type="greeting",
                payload={"message": "Hello"}
            )
            
            await broker.send_message(message)
            
            # Let the message processing run
            await asyncio.sleep(0.1)
            
            # Check that handler was called
            assert "greeting_msg" in handled_messages
            
        finally:
            await broker.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        """Test concurrent message processing."""
        broker = MessageBroker(max_concurrent_handlers=3)
        
        try:
            await broker.start()
            
            # Register agents
            for i in range(5):
                await broker.register_agent(f"agent_{i}")
            
            # Send multiple messages concurrently
            tasks = []
            for i in range(10):
                message = AgentMessage(
                    id=f"msg_{i}",
                    sender="system",
                    recipient=f"agent_{i % 5}",
                    message_type="test",
                    payload={"data": f"test_{i}"}
                )
                tasks.append(broker.send_message(message))
            
            # Wait for all messages to be sent
            results = await asyncio.gather(*tasks)
            assert all(results)
            
            # Verify messages can be received
            for i in range(5):
                agent_id = f"agent_{i}"
                messages_for_agent = []
                
                # Try to receive messages for this agent
                while True:
                    msg = await broker.receive_message(agent_id, timeout=0.1)
                    if msg is None:
                        break
                    messages_for_agent.append(msg)
                
                # Each agent should have received some messages
                assert len(messages_for_agent) > 0
                
        finally:
            await broker.stop()