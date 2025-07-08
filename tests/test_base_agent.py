"""
Tests for the base agent classes and interfaces.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from enum import Enum
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent, AgentState, AgentMessage, AgentInterface
from src.agents.exceptions import AgentError, AgentStateError, AgentCommunicationError


class TestAgentState:
    """Test AgentState enum."""
    
    def test_agent_state_values(self):
        """Test that AgentState has all required values."""
        assert AgentState.INACTIVE.value == "inactive"
        assert AgentState.STARTING.value == "starting"
        assert AgentState.ACTIVE.value == "active"
        assert AgentState.STOPPING.value == "stopping"
        assert AgentState.ERROR.value == "error"


class TestAgentMessage:
    """Test AgentMessage class."""
    
    def test_agent_message_creation(self):
        """Test creating an AgentMessage."""
        msg = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "process"},
            timestamp=1234567890.0
        )
        
        assert msg.id == "msg_123"
        assert msg.sender == "agent_1"
        assert msg.recipient == "agent_2"
        assert msg.message_type == "command"
        assert msg.payload == {"action": "process"}
        assert msg.timestamp == 1234567890.0
    
    def test_agent_message_to_dict(self):
        """Test converting AgentMessage to dict."""
        msg = AgentMessage(
            id="msg_123",
            sender="agent_1",
            recipient="agent_2",
            message_type="command",
            payload={"action": "process"},
            timestamp=1234567890.0
        )
        
        expected = {
            "id": "msg_123",
            "sender": "agent_1",
            "recipient": "agent_2",
            "message_type": "command",
            "payload": {"action": "process"},
            "timestamp": 1234567890.0
        }
        
        assert msg.to_dict() == expected
    
    def test_agent_message_from_dict(self):
        """Test creating AgentMessage from dict."""
        data = {
            "id": "msg_123",
            "sender": "agent_1",
            "recipient": "agent_2",
            "message_type": "command",
            "payload": {"action": "process"},
            "timestamp": 1234567890.0
        }
        
        msg = AgentMessage.from_dict(data)
        
        assert msg.id == "msg_123"
        assert msg.sender == "agent_1"
        assert msg.recipient == "agent_2"
        assert msg.message_type == "command"
        assert msg.payload == {"action": "process"}
        assert msg.timestamp == 1234567890.0


class TestAgentInterface:
    """Test AgentInterface abstract base class."""
    
    def test_agent_interface_is_abstract(self):
        """Test that AgentInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentInterface()
    
    def test_agent_interface_methods_are_abstract(self):
        """Test that all interface methods are abstract."""
        # This will be validated when we implement a concrete class
        pass


class TestBaseAgent:
    """Test BaseAgent abstract base class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "agent_id": "test_agent",
            "heartbeat_interval": 30,
            "max_retries": 3
        }
    
    @pytest.fixture
    def mock_message_broker(self):
        """Mock message broker for testing."""
        broker = Mock()
        broker.publish = AsyncMock()
        broker.subscribe = AsyncMock()
        broker.disconnect = AsyncMock()
        return broker
    
    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent("test_agent", {}, Mock(), Mock())
    
    @pytest.mark.asyncio
    async def test_concrete_agent_initialization(self, mock_logger, mock_config, mock_message_broker):
        """Test initialization of a concrete agent."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.config == mock_config
        assert agent.logger == mock_logger
        assert agent.message_broker == mock_message_broker
        assert agent.state == AgentState.INACTIVE
        assert agent.metrics == {
            "messages_processed": 0,
            "tasks_completed": 0,
            "errors": 0,
            "uptime": 0
        }
    
    @pytest.mark.asyncio
    async def test_agent_start_lifecycle(self, mock_logger, mock_config, mock_message_broker):
        """Test agent start lifecycle."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        # Test starting from inactive state
        await agent.start()
        assert agent.state == AgentState.ACTIVE
        mock_logger.info.assert_called()
        
        # Test starting when already active
        with pytest.raises(AgentStateError):
            await agent.start()
    
    @pytest.mark.asyncio
    async def test_agent_stop_lifecycle(self, mock_logger, mock_config, mock_message_broker):
        """Test agent stop lifecycle."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        # Start the agent first
        await agent.start()
        
        # Test stopping
        await agent.stop()
        assert agent.state == AgentState.INACTIVE
        mock_message_broker.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_send_message(self, mock_logger, mock_config, mock_message_broker):
        """Test sending messages."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        await agent.start()
        
        # Test sending a message
        await agent.send_message("recipient", "command", {"action": "test"})
        
        mock_message_broker.publish.assert_called_once()
        call_args = mock_message_broker.publish.call_args
        message = call_args[0][0]
        
        assert message.sender == "test_agent"
        assert message.recipient == "recipient"
        assert message.message_type == "command"
        assert message.payload == {"action": "test"}
    
    @pytest.mark.asyncio
    async def test_agent_handle_message(self, mock_logger, mock_config, mock_message_broker):
        """Test handling incoming messages."""
        
        class ConcreteAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.processed_messages = []
            
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                self.processed_messages.append(message)
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        await agent.start()
        
        # Test handling a message
        message = AgentMessage(
            id="msg_123",
            sender="sender",
            recipient="test_agent",
            message_type="command",
            payload={"action": "test"}
        )
        
        await agent.handle_message(message)
        
        assert len(agent.processed_messages) == 1
        assert agent.processed_messages[0] == message
        assert agent.metrics["messages_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_execute_task(self, mock_logger, mock_config, mock_message_broker):
        """Test executing tasks."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed", "result": task.get("input", "") + "_processed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        await agent.start()
        
        # Test executing a task
        task = {"input": "test_data"}
        result = await agent.execute_task(task)
        
        assert result["status"] == "completed"
        assert result["result"] == "test_data_processed"
        assert agent.metrics["tasks_completed"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_health_check(self, mock_logger, mock_config, mock_message_broker):
        """Test agent health check."""
        
        class ConcreteAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.healthy = True
            
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return self.healthy
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        await agent.start()
        
        # Test healthy agent
        assert await agent.health_check() is True
        
        # Test unhealthy agent
        agent.healthy = False
        assert await agent.health_check() is False
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_logger, mock_config, mock_message_broker):
        """Test agent error handling."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                raise ValueError("Test error")
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        await agent.start()
        
        # Test error handling during message processing
        message = AgentMessage(
            id="msg_123",
            sender="sender",
            recipient="test_agent",
            message_type="command",
            payload={"action": "test"}
        )
        
        await agent.handle_message(message)
        
        assert agent.metrics["errors"] == 1
        mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_agent_metrics(self, mock_logger, mock_config, mock_message_broker):
        """Test agent metrics collection."""
        
        class ConcreteAgent(BaseAgent):
            async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
                return None
            
            async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "completed"}
            
            async def _health_check(self) -> bool:
                return True
        
        agent = ConcreteAgent(
            agent_id="test_agent",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
        
        await agent.start()
        
        # Initial metrics
        metrics = agent.get_metrics()
        assert metrics["messages_processed"] == 0
        assert metrics["tasks_completed"] == 0
        assert metrics["errors"] == 0
        assert metrics["uptime"] >= 0
        
        # Process a message
        message = AgentMessage(
            id="msg_123",
            sender="sender",
            recipient="test_agent",
            message_type="command",
            payload={"action": "test"}
        )
        await agent.handle_message(message)
        
        # Execute a task
        await agent.execute_task({"input": "test"})
        
        # Check updated metrics
        metrics = agent.get_metrics()
        assert metrics["messages_processed"] == 1
        assert metrics["tasks_completed"] == 1
        assert metrics["errors"] == 0