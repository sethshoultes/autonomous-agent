"""
Tests for the communication protocol and message broker.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional, Callable
import json

from src.communication.broker import MessageBroker, MessageHandler, MessageQueue
from src.communication.protocol import (
    CommunicationProtocol, MessageEncoder, MessageDecoder,
    MessageValidator
)
from src.communication.broker import MessageRouter
from src.agents.base import AgentMessage
from src.agents.exceptions import CommunicationError, MessageValidationError, MessageRoutingError


class TestMessageQueue:
    """Test MessageQueue class."""
    
    @pytest.fixture
    def queue(self):
        """Create a message queue for testing."""
        return MessageQueue(max_size=10)
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self, queue):
        """Test queue initialization."""
        assert queue.max_size == 10
        assert queue.size == 0
        assert queue.is_empty is True
        assert queue.is_full is False
    
    @pytest.mark.asyncio
    async def test_put_message(self, queue):
        """Test putting a message in the queue."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        await queue.put(message)
        
        assert queue.size == 1
        assert queue.is_empty is False
    
    @pytest.mark.asyncio
    async def test_get_message(self, queue):
        """Test getting a message from the queue."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        await queue.put(message)
        retrieved_message = await queue.get()
        
        assert retrieved_message == message
        assert queue.size == 0
        assert queue.is_empty is True
    
    @pytest.mark.asyncio
    async def test_queue_full(self, queue):
        """Test queue when full."""
        # Fill the queue
        for i in range(10):
            message = AgentMessage(
                id=f"msg_{i}",
                sender="agent_1",
                recipient="agent_2",
                message_type="command",
                payload={"action": f"test_{i}"}
            )
            await queue.put(message)
        
        assert queue.is_full is True
        
        # Try to add one more message
        extra_message = AgentMessage(
            id="msg_extra",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "extra"}
        )
        
        with pytest.raises(CommunicationError):
            await queue.put(extra_message, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_queue_empty_timeout(self, queue):
        """Test getting from empty queue with timeout."""
        with pytest.raises(CommunicationError):
            await queue.get(timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_queue_priority(self):
        """Test priority queue functionality."""
        queue = MessageQueue(max_size=10, priority_queue=True)
        
        # Add messages with different priorities
        high_priority = AgentMessage(
            id="msg_high",
            sender="agent_1",
            recipient="agent_2",
            message_type="urgent",
            payload={"action": "urgent"},
            priority=1
        )
        
        low_priority = AgentMessage(
            id="msg_low",
            sender="agent_1",
            recipient="agent_2",
            message_type="info",
            payload={"action": "info"},
            priority=10
        )
        
        # Add low priority first
        await queue.put(low_priority)
        await queue.put(high_priority)
        
        # High priority should come out first
        first_message = await queue.get()
        assert first_message.id == "msg_high"
        
        second_message = await queue.get()
        assert second_message.id == "msg_low"


class TestMessageHandler:
    """Test MessageHandler class."""
    
    @pytest.fixture
    def handler(self):
        """Create a message handler for testing."""
        return MessageHandler()
    
    @pytest.mark.asyncio
    async def test_handler_initialization(self, handler):
        """Test handler initialization."""
        assert len(handler.subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_handler(self, handler):
        """Test subscribing to message types."""
        async def test_callback(message):
            return f"Processed: {message.id}"
        
        handler.subscribe("command", test_callback)
        
        assert "command" in handler.subscribers
        assert test_callback in handler.subscribers["command"]
    
    @pytest.mark.asyncio
    async def test_unsubscribe_handler(self, handler):
        """Test unsubscribing from message types."""
        async def test_callback(message):
            return f"Processed: {message.id}"
        
        handler.subscribe("command", test_callback)
        handler.unsubscribe("command", test_callback)
        
        assert "command" not in handler.subscribers or test_callback not in handler.subscribers["command"]
    
    @pytest.mark.asyncio
    async def test_handle_message(self, handler):
        """Test handling a message."""
        processed_messages = []
        
        async def test_callback(message):
            processed_messages.append(message.id)
            return f"Processed: {message.id}"
        
        handler.subscribe("command", test_callback)
        
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        results = await handler.handle_message(message)
        
        assert len(processed_messages) == 1
        assert processed_messages[0] == "msg_123"
        assert len(results) == 1
        assert results[0] == "Processed: msg_123"
    
    @pytest.mark.asyncio
    async def test_handle_message_no_subscribers(self, handler):
        """Test handling a message with no subscribers."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="unknown",
            payload={"action": "test"}
        )
        
        results = await handler.handle_message(message)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_handle_message_error(self, handler):
        """Test handling a message with error in callback."""
        async def error_callback(message):
            raise ValueError("Test error")
        
        handler.subscribe("command", error_callback)
        
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        # Should not raise error, but log it
        results = await handler.handle_message(message)
        
        assert len(results) == 0  # No successful results


class TestMessageEncoder:
    """Test MessageEncoder class."""
    
    @pytest.fixture
    def encoder(self):
        """Create a message encoder for testing."""
        return MessageEncoder()
    
    def test_encode_message(self, encoder):
        """Test encoding a message."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        encoded = encoder.encode(message)
        
        assert isinstance(encoded, bytes)
        
        # Decode to verify
        decoded_data = json.loads(encoded.decode('utf-8'))
        assert decoded_data["id"] == "msg_123"
        assert decoded_data["sender"] == "agent_1"
        assert decoded_data["recipient"] == "agent_2"
        assert decoded_data["message_type"] == "command"
        assert decoded_data["payload"] == {"action": "test"}
    
    def test_encode_invalid_message(self, encoder):
        """Test encoding an invalid message."""
        with pytest.raises(MessageValidationError):
            encoder.encode(None)
        
        with pytest.raises(MessageValidationError):
            encoder.encode("not a message")


class TestMessageDecoder:
    """Test MessageDecoder class."""
    
    @pytest.fixture
    def decoder(self):
        """Create a message decoder for testing."""
        return MessageDecoder()
    
    def test_decode_message(self, decoder):
        """Test decoding a message."""
        message_data = {
            "id": "msg_123",
            "sender": "agent_1",
            "recipient": "agent_2",
            "message_type": "command",
            "payload": {"action": "test"},
            "timestamp": 1234567890.0
        }
        
        encoded = json.dumps(message_data).encode('utf-8')
        decoded_message = decoder.decode(encoded)
        
        assert isinstance(decoded_message, AgentMessage)
        assert decoded_message.id == "msg_123"
        assert decoded_message.sender == "agent_1"
        assert decoded_message.recipient == "agent_2"
        assert decoded_message.message_type == "command"
        assert decoded_message.payload == {"action": "test"}
        assert decoded_message.timestamp == 1234567890.0
    
    def test_decode_invalid_data(self, decoder):
        """Test decoding invalid data."""
        with pytest.raises(MessageValidationError):
            decoder.decode(b"invalid json")
        
        with pytest.raises(MessageValidationError):
            decoder.decode(json.dumps({"invalid": "message"}).encode('utf-8'))


class TestMessageValidator:
    """Test MessageValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a message validator for testing."""
        return MessageValidator()
    
    def test_validate_message(self, validator):
        """Test validating a valid message."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        # Should not raise an exception
        validator.validate(message)
    
    def test_validate_invalid_message(self, validator):
        """Test validating an invalid message."""
        # Missing required fields
        with pytest.raises(MessageValidationError):
            validator.validate(None)
        
        # Invalid message type
        invalid_message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="",  # Empty message type
            payload={"action": "test"}
        )
        
        with pytest.raises(MessageValidationError):
            validator.validate(invalid_message)
    
    def test_validate_message_size(self, validator):
        """Test validating message size."""
        # Large payload
        large_payload = {"data": "x" * 1000000}  # 1MB of data
        
        large_message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload=large_payload
        )
        
        with pytest.raises(MessageValidationError):
            validator.validate(large_message, max_size=1024)  # 1KB limit


class TestMessageRouter:
    """Test MessageRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create a message router for testing."""
        return MessageRouter()
    
    @pytest.fixture
    def mock_agent_registry(self):
        """Mock agent registry for testing."""
        registry = Mock()
        registry.get_agent = Mock()
        registry.list_agents = Mock(return_value=[])
        return registry
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router.routes == {}
        assert router.default_route is None
    
    def test_add_route(self, router):
        """Test adding a route."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        router.add_route("test_recipient", test_handler)
        
        assert "test_recipient" in router.routes
        assert router.routes["test_recipient"] == test_handler
    
    def test_remove_route(self, router):
        """Test removing a route."""
        async def test_handler(message):
            return f"Handled: {message.id}"
        
        router.add_route("test_recipient", test_handler)
        router.remove_route("test_recipient")
        
        assert "test_recipient" not in router.routes
    
    def test_set_default_route(self, router):
        """Test setting a default route."""
        async def default_handler(message):
            return f"Default: {message.id}"
        
        router.set_default_route(default_handler)
        
        assert router.default_route == default_handler
    
    @pytest.mark.asyncio
    async def test_route_message(self, router):
        """Test routing a message."""
        processed_messages = []
        
        async def test_handler(message):
            processed_messages.append(message.id)
            return f"Handled: {message.id}"
        
        router.add_route("agent_2", test_handler)
        
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        result = await router.route_message(message)
        
        assert len(processed_messages) == 1
        assert processed_messages[0] == "msg_123"
        assert result == "Handled: msg_123"
    
    @pytest.mark.asyncio
    async def test_route_message_no_route(self, router):
        """Test routing a message with no specific route."""
        processed_messages = []
        
        async def default_handler(message):
            processed_messages.append(message.id)
            return f"Default: {message.id}"
        
        router.set_default_route(default_handler)
        
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="unknown_agent",
            message_type="command",
            payload={"action": "test"}
        )
        
        result = await router.route_message(message)
        
        assert len(processed_messages) == 1
        assert processed_messages[0] == "msg_123"
        assert result == "Default: msg_123"
    
    @pytest.mark.asyncio
    async def test_route_message_no_handler(self, router):
        """Test routing a message with no handler."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="unknown_agent",
            message_type="command",
            payload={"action": "test"}
        )
        
        with pytest.raises(MessageRoutingError):
            await router.route_message(message)
    
    @pytest.mark.asyncio
    async def test_route_broadcast_message(self, router):
        """Test routing a broadcast message."""
        processed_messages = []
        
        async def handler_1(message):
            processed_messages.append(f"handler_1: {message.id}")
            return f"Handler1: {message.id}"
        
        async def handler_2(message):
            processed_messages.append(f"handler_2: {message.id}")
            return f"Handler2: {message.id}"
        
        router.add_route("agent_1", handler_1)
        router.add_route("agent_2", handler_2)
        
        message = AgentMessage(
            id="msg_123",
            sender="manager",
            recipient="broadcast",
            message_type="announcement",
            payload={"message": "Hello all"}
        )
        
        results = await router.route_broadcast_message(message)
        
        assert len(processed_messages) == 2
        assert "handler_1: msg_123" in processed_messages
        assert "handler_2: msg_123" in processed_messages
        assert len(results) == 2


class TestMessageBroker:
    """Test MessageBroker class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def broker(self, mock_logger):
        """Create a message broker for testing."""
        return MessageBroker(logger=mock_logger)
    
    def test_broker_initialization(self, broker):
        """Test broker initialization."""
        assert broker.logger is not None
        assert isinstance(broker.message_queue, MessageQueue)
        assert isinstance(broker.message_handler, MessageHandler)
        assert isinstance(broker.message_router, MessageRouter)
        assert broker.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_broker(self, broker):
        """Test starting the broker."""
        await broker.start()
        
        assert broker.is_running is True
        
        # Stop the broker
        await broker.stop()
    
    @pytest.mark.asyncio
    async def test_stop_broker(self, broker):
        """Test stopping the broker."""
        await broker.start()
        await broker.stop()
        
        assert broker.is_running is False
    
    @pytest.mark.asyncio
    async def test_publish_message(self, broker):
        """Test publishing a message."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        await broker.start()
        await broker.publish(message)
        
        # Message should be in the queue
        assert broker.message_queue.size == 1
        
        await broker.stop()
    
    @pytest.mark.asyncio
    async def test_subscribe_to_messages(self, broker):
        """Test subscribing to messages."""
        processed_messages = []
        
        async def test_callback(message):
            processed_messages.append(message.id)
            return f"Processed: {message.id}"
        
        await broker.subscribe("command", test_callback)
        
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        await broker.start()
        await broker.publish(message)
        
        # Give some time for message processing
        await asyncio.sleep(0.1)
        
        assert len(processed_messages) == 1
        assert processed_messages[0] == "msg_123"
        
        await broker.stop()
    
    @pytest.mark.asyncio
    async def test_disconnect_broker(self, broker):
        """Test disconnecting from the broker."""
        await broker.start()
        await broker.disconnect()
        
        assert broker.is_running is False


class TestCommunicationProtocol:
    """Test CommunicationProtocol class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def protocol(self, mock_logger):
        """Create a communication protocol for testing."""
        return CommunicationProtocol(logger=mock_logger)
    
    def test_protocol_initialization(self, protocol):
        """Test protocol initialization."""
        assert protocol.logger is not None
        assert isinstance(protocol.encoder, MessageEncoder)
        assert isinstance(protocol.decoder, MessageDecoder)
        assert isinstance(protocol.validator, MessageValidator)
    
    def test_encode_decode_message(self, protocol):
        """Test encoding and decoding a message."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        # Encode the message
        encoded = protocol.encode_message(message)
        assert isinstance(encoded, bytes)
        
        # Decode the message
        decoded = protocol.decode_message(encoded)
        assert isinstance(decoded, AgentMessage)
        assert decoded.id == message.id
        assert decoded.sender == message.sender
        assert decoded.recipient == message.recipient
        assert decoded.message_type == message.message_type
        assert decoded.payload == message.payload
    
    def test_validate_message(self, protocol):
        """Test validating a message."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        # Should not raise an exception
        protocol.validate_message(message)
    
    def test_validate_invalid_message(self, protocol):
        """Test validating an invalid message."""
        with pytest.raises(MessageValidationError):
            protocol.validate_message(None)
    
    @pytest.mark.asyncio
    async def test_send_message_with_retry(self, protocol):
        """Test sending a message with retry logic."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        # Mock the actual send method to fail first, then succeed
        call_count = 0
        
        async def mock_send(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise CommunicationError("Connection failed")
            return True
        
        protocol._send = mock_send
        
        result = await protocol.send_message_with_retry(message, max_retries=3)
        
        assert result is True
        assert call_count == 2  # Failed once, succeeded on second try
    
    @pytest.mark.asyncio
    async def test_send_message_retry_exhausted(self, protocol):
        """Test sending a message when all retries are exhausted."""
        message = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "test"}
        )
        
        # Mock the actual send method to always fail
        async def mock_send(msg):
            raise CommunicationError("Connection failed")
        
        protocol._send = mock_send
        
        with pytest.raises(CommunicationError):
            await protocol.send_message_with_retry(message, max_retries=3)