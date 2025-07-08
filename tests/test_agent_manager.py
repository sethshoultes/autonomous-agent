"""
Tests for the Agent Manager class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

from src.agents.manager import AgentManager, AgentRegistry, AgentConfig
from src.agents.base import BaseAgent, AgentState, AgentMessage
from src.agents.exceptions import AgentManagerError, AgentRegistrationError, AgentNotFoundError


class TestAgentConfig:
    """Test AgentConfig class."""
    
    def test_agent_config_creation(self):
        """Test creating an AgentConfig."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={"param1": "value1"},
            enabled=True,
            priority=1
        )
        
        assert config.agent_id == "test_agent"
        assert config.agent_type == "TestAgent"
        assert config.config == {"param1": "value1"}
        assert config.enabled is True
        assert config.priority == 1
    
    def test_agent_config_to_dict(self):
        """Test converting AgentConfig to dict."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={"param1": "value1"},
            enabled=True,
            priority=1
        )
        
        expected = {
            "agent_id": "test_agent",
            "agent_type": "TestAgent",
            "config": {"param1": "value1"},
            "enabled": True,
            "priority": 1
        }
        
        assert config.to_dict() == expected
    
    def test_agent_config_from_dict(self):
        """Test creating AgentConfig from dict."""
        data = {
            "agent_id": "test_agent",
            "agent_type": "TestAgent",
            "config": {"param1": "value1"},
            "enabled": True,
            "priority": 1
        }
        
        config = AgentConfig.from_dict(data)
        
        assert config.agent_id == "test_agent"
        assert config.agent_type == "TestAgent"
        assert config.config == {"param1": "value1"}
        assert config.enabled is True
        assert config.priority == 1


class TestAgentRegistry:
    """Test AgentRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a registry for testing."""
        return AgentRegistry()
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.INACTIVE
        agent.get_metrics.return_value = {"messages_processed": 0}
        return agent
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry.agents) == 0
        assert len(registry.agent_configs) == 0
    
    def test_register_agent(self, registry, mock_agent):
        """Test registering an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        registry.register_agent(mock_agent, config)
        
        assert "test_agent" in registry.agents
        assert "test_agent" in registry.agent_configs
        assert registry.agents["test_agent"] == mock_agent
        assert registry.agent_configs["test_agent"] == config
    
    def test_register_duplicate_agent(self, registry, mock_agent):
        """Test registering a duplicate agent raises error."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        registry.register_agent(mock_agent, config)
        
        with pytest.raises(AgentRegistrationError):
            registry.register_agent(mock_agent, config)
    
    def test_unregister_agent(self, registry, mock_agent):
        """Test unregistering an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        registry.register_agent(mock_agent, config)
        registry.unregister_agent("test_agent")
        
        assert "test_agent" not in registry.agents
        assert "test_agent" not in registry.agent_configs
    
    def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering a non-existent agent raises error."""
        with pytest.raises(AgentNotFoundError):
            registry.unregister_agent("nonexistent_agent")
    
    def test_get_agent(self, registry, mock_agent):
        """Test getting an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        registry.register_agent(mock_agent, config)
        
        retrieved_agent = registry.get_agent("test_agent")
        assert retrieved_agent == mock_agent
    
    def test_get_nonexistent_agent(self, registry):
        """Test getting a non-existent agent raises error."""
        with pytest.raises(AgentNotFoundError):
            registry.get_agent("nonexistent_agent")
    
    def test_list_agents(self, registry, mock_agent):
        """Test listing agents."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        registry.register_agent(mock_agent, config)
        
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0] == mock_agent
    
    def test_list_agent_configs(self, registry, mock_agent):
        """Test listing agent configurations."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        registry.register_agent(mock_agent, config)
        
        configs = registry.list_agent_configs()
        assert len(configs) == 1
        assert configs[0] == config


class TestAgentManager:
    """Test AgentManager class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "max_agents": 10,
            "heartbeat_interval": 30,
            "communication_timeout": 10,
            "retry_attempts": 3
        }
    
    @pytest.fixture
    def mock_message_broker(self):
        """Mock message broker for testing."""
        broker = Mock()
        broker.publish = AsyncMock()
        broker.subscribe = AsyncMock()
        broker.disconnect = AsyncMock()
        return broker
    
    @pytest.fixture
    def manager(self, mock_logger, mock_config, mock_message_broker):
        """Create an AgentManager for testing."""
        return AgentManager(
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker
        )
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.INACTIVE
        agent.start = AsyncMock()
        agent.stop = AsyncMock()
        agent.send_message = AsyncMock()
        agent.handle_message = AsyncMock()
        agent.execute_task = AsyncMock(return_value={"status": "completed"})
        agent.health_check = AsyncMock(return_value=True)
        agent.get_metrics = Mock(return_value={"messages_processed": 0})
        return agent
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config is not None
        assert manager.logger is not None
        assert manager.message_broker is not None
        assert isinstance(manager.registry, AgentRegistry)
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_manager(self, manager):
        """Test starting the manager."""
        await manager.start()
        assert manager.is_running is True
    
    @pytest.mark.asyncio
    async def test_stop_manager(self, manager):
        """Test stopping the manager."""
        await manager.start()
        await manager.stop()
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_register_agent(self, manager, mock_agent):
        """Test registering an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        
        assert "test_agent" in manager.registry.agents
        assert manager.registry.agents["test_agent"] == mock_agent
    
    @pytest.mark.asyncio
    async def test_unregister_agent(self, manager, mock_agent):
        """Test unregistering an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        await manager.unregister_agent("test_agent")
        
        assert "test_agent" not in manager.registry.agents
        mock_agent.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_agent(self, manager, mock_agent):
        """Test starting an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        await manager.start_agent("test_agent")
        
        mock_agent.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_agent(self, manager, mock_agent):
        """Test stopping an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        await manager.start_agent("test_agent")
        await manager.stop_agent("test_agent")
        
        mock_agent.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message(self, manager, mock_agent):
        """Test sending a message to an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        await manager.start_agent("test_agent")
        
        message = AgentMessage(
            id="msg_123",
            sender="manager",
            recipient="test_agent",
            message_type="command",
            payload={"action": "test"}
        )
        
        await manager.send_message(message)
        
        mock_agent.handle_message.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, manager):
        """Test broadcasting a message to all agents."""
        # Create multiple mock agents
        mock_agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.ACTIVE
            agent.handle_message = AsyncMock()
            mock_agents.append(agent)
            
            config = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type="TestAgent",
                config={},
                enabled=True,
                priority=1
            )
            
            await manager.register_agent(agent, config)
        
        message = AgentMessage(
            id="msg_123",
            sender="manager",
            recipient="broadcast",
            message_type="announcement",
            payload={"message": "Hello all agents"}
        )
        
        await manager.broadcast_message(message)
        
        # Verify all agents received the message
        for agent in mock_agents:
            agent.handle_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task(self, manager, mock_agent):
        """Test executing a task on an agent."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        await manager.start_agent("test_agent")
        
        task = {"action": "process", "data": "test_data"}
        result = await manager.execute_task("test_agent", task)
        
        mock_agent.execute_task.assert_called_once_with(task)
        assert result == {"status": "completed"}
    
    @pytest.mark.asyncio
    async def test_health_check_all_agents(self, manager):
        """Test health checking all agents."""
        # Create multiple mock agents
        mock_agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.ACTIVE
            agent.health_check = AsyncMock(return_value=True)
            mock_agents.append(agent)
            
            config = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type="TestAgent",
                config={},
                enabled=True,
                priority=1
            )
            
            await manager.register_agent(agent, config)
        
        health_status = await manager.health_check_all_agents()
        
        assert len(health_status) == 3
        for agent_id, status in health_status.items():
            assert status is True
    
    @pytest.mark.asyncio
    async def test_get_agent_metrics(self, manager, mock_agent):
        """Test getting agent metrics."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        
        metrics = await manager.get_agent_metrics("test_agent")
        
        mock_agent.get_metrics.assert_called_once()
        assert metrics == {"messages_processed": 0}
    
    @pytest.mark.asyncio
    async def test_get_all_agent_metrics(self, manager):
        """Test getting all agent metrics."""
        # Create multiple mock agents
        mock_agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.ACTIVE
            agent.get_metrics = Mock(return_value={"messages_processed": i})
            mock_agents.append(agent)
            
            config = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type="TestAgent",
                config={},
                enabled=True,
                priority=1
            )
            
            await manager.register_agent(agent, config)
        
        all_metrics = await manager.get_all_agent_metrics()
        
        assert len(all_metrics) == 3
        for i, (agent_id, metrics) in enumerate(all_metrics.items()):
            assert agent_id == f"agent_{i}"
            assert metrics == {"messages_processed": i}
    
    @pytest.mark.asyncio
    async def test_start_all_agents(self, manager):
        """Test starting all agents."""
        # Create multiple mock agents
        mock_agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.INACTIVE
            agent.start = AsyncMock()
            mock_agents.append(agent)
            
            config = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type="TestAgent",
                config={},
                enabled=True,
                priority=1
            )
            
            await manager.register_agent(agent, config)
        
        await manager.start_all_agents()
        
        # Verify all agents were started
        for agent in mock_agents:
            agent.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_all_agents(self, manager):
        """Test stopping all agents."""
        # Create multiple mock agents
        mock_agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.ACTIVE
            agent.stop = AsyncMock()
            mock_agents.append(agent)
            
            config = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type="TestAgent",
                config={},
                enabled=True,
                priority=1
            )
            
            await manager.register_agent(agent, config)
        
        await manager.stop_all_agents()
        
        # Verify all agents were stopped
        for agent in mock_agents:
            agent.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, manager, mock_agent):
        """Test error handling in manager operations."""
        config = AgentConfig(
            agent_id="test_agent",
            agent_type="TestAgent",
            config={},
            enabled=True,
            priority=1
        )
        
        await manager.register_agent(mock_agent, config)
        
        # Test error when agent execution fails
        mock_agent.execute_task.side_effect = Exception("Task failed")
        
        with pytest.raises(AgentManagerError):
            await manager.execute_task("test_agent", {"action": "test"})
        
        # Verify error was logged
        manager.logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_agent_not_found_error(self, manager):
        """Test error when agent is not found."""
        with pytest.raises(AgentNotFoundError):
            await manager.start_agent("nonexistent_agent")
        
        with pytest.raises(AgentNotFoundError):
            await manager.execute_task("nonexistent_agent", {"action": "test"})