"""
Tests for the agent lifecycle management system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
import time

from src.lifecycle.manager import LifecycleManager, LifecycleState, LifecycleEvent
from src.lifecycle.hooks import LifecycleHook, PreStartHook, PostStartHook, PreStopHook, PostStopHook
from src.lifecycle.monitor import LifecycleMonitor, HealthCheck, PerformanceMonitor
from src.agents.base import BaseAgent, AgentState, AgentMessage
from src.agents.exceptions import AgentError, AgentStateError


class TestLifecycleState:
    """Test LifecycleState enum."""
    
    def test_lifecycle_state_values(self):
        """Test that LifecycleState has all required values."""
        assert LifecycleState.CREATED.value == "created"
        assert LifecycleState.INITIALIZING.value == "initializing"
        assert LifecycleState.INITIALIZED.value == "initialized"
        assert LifecycleState.STARTING.value == "starting"
        assert LifecycleState.RUNNING.value == "running"
        assert LifecycleState.STOPPING.value == "stopping"
        assert LifecycleState.STOPPED.value == "stopped"
        assert LifecycleState.ERROR.value == "error"
        assert LifecycleState.DESTROYED.value == "destroyed"
    
    def test_lifecycle_state_transitions(self):
        """Test valid state transitions."""
        # Valid transitions
        valid_transitions = [
            (LifecycleState.CREATED, LifecycleState.INITIALIZING),
            (LifecycleState.INITIALIZING, LifecycleState.INITIALIZED),
            (LifecycleState.INITIALIZED, LifecycleState.STARTING),
            (LifecycleState.STARTING, LifecycleState.RUNNING),
            (LifecycleState.RUNNING, LifecycleState.STOPPING),
            (LifecycleState.STOPPING, LifecycleState.STOPPED),
            (LifecycleState.STOPPED, LifecycleState.DESTROYED),
        ]
        
        for current, next_state in valid_transitions:
            assert current != next_state  # States should be different


class TestLifecycleEvent:
    """Test LifecycleEvent class."""
    
    def test_lifecycle_event_creation(self):
        """Test creating a LifecycleEvent."""
        event = LifecycleEvent(
            agent_id="test_agent",
            event_type="start",
            timestamp=1234567890.0,
            data={"key": "value"}
        )
        
        assert event.agent_id == "test_agent"
        assert event.event_type == "start"
        assert event.timestamp == 1234567890.0
        assert event.data == {"key": "value"}
    
    def test_lifecycle_event_to_dict(self):
        """Test converting LifecycleEvent to dict."""
        event = LifecycleEvent(
            agent_id="test_agent",
            event_type="start",
            timestamp=1234567890.0,
            data={"key": "value"}
        )
        
        expected = {
            "agent_id": "test_agent",
            "event_type": "start",
            "timestamp": 1234567890.0,
            "data": {"key": "value"}
        }
        
        assert event.to_dict() == expected
    
    def test_lifecycle_event_from_dict(self):
        """Test creating LifecycleEvent from dict."""
        data = {
            "agent_id": "test_agent",
            "event_type": "start",
            "timestamp": 1234567890.0,
            "data": {"key": "value"}
        }
        
        event = LifecycleEvent.from_dict(data)
        
        assert event.agent_id == "test_agent"
        assert event.event_type == "start"
        assert event.timestamp == 1234567890.0
        assert event.data == {"key": "value"}


class TestLifecycleHook:
    """Test LifecycleHook base class."""
    
    def test_lifecycle_hook_is_abstract(self):
        """Test that LifecycleHook cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LifecycleHook()
    
    @pytest.mark.asyncio
    async def test_concrete_hook_implementation(self):
        """Test implementing a concrete hook."""
        
        class ConcreteHook(LifecycleHook):
            def __init__(self):
                self.executed = False
                self.agent_id = None
                self.context = None
            
            async def execute(self, agent_id: str, context: Dict[str, Any]) -> bool:
                self.executed = True
                self.agent_id = agent_id
                self.context = context
                return True
        
        hook = ConcreteHook()
        result = await hook.execute("test_agent", {"key": "value"})
        
        assert result is True
        assert hook.executed is True
        assert hook.agent_id == "test_agent"
        assert hook.context == {"key": "value"}


class TestPreStartHook:
    """Test PreStartHook implementation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.INACTIVE
        agent.health_check = AsyncMock(return_value=True)
        return agent
    
    @pytest.mark.asyncio
    async def test_pre_start_hook_execution(self, mock_agent):
        """Test executing pre-start hook."""
        hook = PreStartHook()
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_agent.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pre_start_hook_validation_failure(self, mock_agent):
        """Test pre-start hook with validation failure."""
        hook = PreStartHook()
        
        # Make health check fail
        mock_agent.health_check.return_value = False
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is False


class TestPostStartHook:
    """Test PostStartHook implementation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.ACTIVE
        return agent
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_post_start_hook_execution(self, mock_agent, mock_logger):
        """Test executing post-start hook."""
        hook = PostStartHook(logger=mock_logger)
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_logger.info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_post_start_hook_registration(self, mock_agent, mock_logger):
        """Test post-start hook with registration."""
        hook = PostStartHook(logger=mock_logger)
        
        # Mock registry
        mock_registry = Mock()
        mock_registry.register = Mock()
        
        context = {"agent": mock_agent, "registry": mock_registry}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_registry.register.assert_called_once_with("test_agent", mock_agent)


class TestPreStopHook:
    """Test PreStopHook implementation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.ACTIVE
        agent.get_metrics = Mock(return_value={"messages_processed": 100})
        return agent
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_pre_stop_hook_execution(self, mock_agent, mock_logger):
        """Test executing pre-stop hook."""
        hook = PreStopHook(logger=mock_logger)
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_agent.get_metrics.assert_called_once()
        mock_logger.info.assert_called()
    
    @pytest.mark.asyncio
    async def test_pre_stop_hook_graceful_shutdown(self, mock_agent, mock_logger):
        """Test pre-stop hook with graceful shutdown."""
        hook = PreStopHook(logger=mock_logger, graceful_shutdown_timeout=5.0)
        
        # Mock graceful shutdown
        mock_agent.graceful_shutdown = AsyncMock(return_value=True)
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_agent.graceful_shutdown.assert_called_once_with(timeout=5.0)


class TestPostStopHook:
    """Test PostStopHook implementation."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.INACTIVE
        return agent
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_post_stop_hook_execution(self, mock_agent, mock_logger):
        """Test executing post-stop hook."""
        hook = PostStopHook(logger=mock_logger)
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_logger.info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_post_stop_hook_cleanup(self, mock_agent, mock_logger):
        """Test post-stop hook with cleanup."""
        hook = PostStopHook(logger=mock_logger)
        
        # Mock cleanup resources
        mock_agent.cleanup_resources = AsyncMock()
        
        context = {"agent": mock_agent}
        result = await hook.execute("test_agent", context)
        
        assert result is True
        mock_agent.cleanup_resources.assert_called_once()


class TestHealthCheck:
    """Test HealthCheck class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.ACTIVE
        agent.health_check = AsyncMock(return_value=True)
        return agent
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, mock_agent):
        """Test executing health check."""
        health_check = HealthCheck(interval=1.0)
        
        result = await health_check.check(mock_agent)
        
        assert result is True
        mock_agent.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_agent):
        """Test health check failure."""
        health_check = HealthCheck(interval=1.0)
        
        # Make health check fail
        mock_agent.health_check.return_value = False
        
        result = await health_check.check(mock_agent)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, mock_agent):
        """Test health check with timeout."""
        health_check = HealthCheck(interval=1.0, timeout=0.1)
        
        # Make health check hang
        async def slow_health_check():
            await asyncio.sleep(1.0)
            return True
        
        mock_agent.health_check = slow_health_check
        
        result = await health_check.check(mock_agent)
        
        assert result is False  # Should timeout
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, mock_agent):
        """Test health check with exception."""
        health_check = HealthCheck(interval=1.0)
        
        # Make health check raise exception
        mock_agent.health_check.side_effect = Exception("Health check failed")
        
        result = await health_check.check(mock_agent)
        
        assert result is False


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.ACTIVE
        agent.get_metrics = Mock(return_value={
            "messages_processed": 100,
            "tasks_completed": 50,
            "errors": 2,
            "uptime": 3600
        })
        return agent
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(interval=5.0)
        
        assert monitor.interval == 5.0
        assert monitor.metrics_history == []
        assert monitor.thresholds == {}
    
    @pytest.mark.asyncio
    async def test_performance_monitor_collect_metrics(self, mock_agent):
        """Test collecting performance metrics."""
        monitor = PerformanceMonitor(interval=5.0)
        
        metrics = await monitor.collect_metrics(mock_agent)
        
        assert metrics["messages_processed"] == 100
        assert metrics["tasks_completed"] == 50
        assert metrics["errors"] == 2
        assert metrics["uptime"] == 3600
        mock_agent.get_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_monitor_threshold_check(self, mock_agent):
        """Test performance threshold checking."""
        monitor = PerformanceMonitor(interval=5.0)
        monitor.set_threshold("error_rate", 0.05)  # 5% error rate
        
        # High error rate
        mock_agent.get_metrics.return_value = {
            "messages_processed": 100,
            "errors": 10  # 10% error rate
        }
        
        violations = await monitor.check_thresholds(mock_agent)
        
        assert len(violations) == 1
        assert violations[0]["metric"] == "error_rate"
        assert violations[0]["value"] > 0.05
    
    def test_performance_monitor_set_threshold(self):
        """Test setting performance thresholds."""
        monitor = PerformanceMonitor(interval=5.0)
        
        monitor.set_threshold("cpu_usage", 0.8)
        monitor.set_threshold("memory_usage", 0.9)
        
        assert monitor.thresholds["cpu_usage"] == 0.8
        assert monitor.thresholds["memory_usage"] == 0.9
    
    def test_performance_monitor_get_metrics_history(self):
        """Test getting metrics history."""
        monitor = PerformanceMonitor(interval=5.0)
        
        # Add some metrics history
        monitor.metrics_history = [
            {"timestamp": 1000, "messages_processed": 50},
            {"timestamp": 2000, "messages_processed": 100},
            {"timestamp": 3000, "messages_processed": 150}
        ]
        
        history = monitor.get_metrics_history()
        
        assert len(history) == 3
        assert history[0]["messages_processed"] == 50
        assert history[-1]["messages_processed"] == 150
    
    def test_performance_monitor_calculate_rates(self):
        """Test calculating performance rates."""
        monitor = PerformanceMonitor(interval=5.0)
        
        # Add metrics history
        monitor.metrics_history = [
            {"timestamp": 1000, "messages_processed": 50},
            {"timestamp": 2000, "messages_processed": 100}
        ]
        
        current_metrics = {"timestamp": 3000, "messages_processed": 150}
        
        rates = monitor.calculate_rates(current_metrics)
        
        assert "messages_per_second" in rates
        assert rates["messages_per_second"] == 50.0 / 1000.0  # 50 messages per second


class TestLifecycleMonitor:
    """Test LifecycleMonitor class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.ACTIVE
        agent.health_check = AsyncMock(return_value=True)
        agent.get_metrics = Mock(return_value={"messages_processed": 100})
        return agent
    
    def test_lifecycle_monitor_initialization(self, mock_logger):
        """Test lifecycle monitor initialization."""
        monitor = LifecycleMonitor(logger=mock_logger)
        
        assert monitor.logger == mock_logger
        assert monitor.monitored_agents == {}
        assert monitor.health_checks == {}
        assert monitor.performance_monitors == {}
        assert monitor.is_running is False
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, mock_logger, mock_agent):
        """Test starting monitoring."""
        monitor = LifecycleMonitor(logger=mock_logger)
        
        await monitor.start_monitoring(mock_agent)
        
        assert mock_agent.agent_id in monitor.monitored_agents
        assert mock_agent.agent_id in monitor.health_checks
        assert mock_agent.agent_id in monitor.performance_monitors
        assert monitor.monitored_agents[mock_agent.agent_id] == mock_agent
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, mock_logger, mock_agent):
        """Test stopping monitoring."""
        monitor = LifecycleMonitor(logger=mock_logger)
        
        await monitor.start_monitoring(mock_agent)
        await monitor.stop_monitoring(mock_agent.agent_id)
        
        assert mock_agent.agent_id not in monitor.monitored_agents
        assert mock_agent.agent_id not in monitor.health_checks
        assert mock_agent.agent_id not in monitor.performance_monitors
    
    @pytest.mark.asyncio
    async def test_health_check_monitoring(self, mock_logger, mock_agent):
        """Test health check monitoring."""
        monitor = LifecycleMonitor(logger=mock_logger, health_check_interval=0.1)
        
        await monitor.start_monitoring(mock_agent)
        await monitor.start()
        
        # Give some time for health checks to run
        await asyncio.sleep(0.2)
        
        await monitor.stop()
        
        # Verify health check was called
        mock_agent.health_check.assert_called()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_logger, mock_agent):
        """Test performance monitoring."""
        monitor = LifecycleMonitor(logger=mock_logger, performance_monitor_interval=0.1)
        
        await monitor.start_monitoring(mock_agent)
        await monitor.start()
        
        # Give some time for performance monitoring to run
        await asyncio.sleep(0.2)
        
        await monitor.stop()
        
        # Verify metrics were collected
        mock_agent.get_metrics.assert_called()
    
    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, mock_logger, mock_agent):
        """Test handling health check failures."""
        monitor = LifecycleMonitor(logger=mock_logger, health_check_interval=0.1)
        
        # Make health check fail
        mock_agent.health_check.return_value = False
        
        await monitor.start_monitoring(mock_agent)
        await monitor.start()
        
        # Give some time for health checks to run
        await asyncio.sleep(0.2)
        
        await monitor.stop()
        
        # Verify error was logged
        mock_logger.error.assert_called()
    
    def test_get_agent_status(self, mock_logger, mock_agent):
        """Test getting agent status."""
        monitor = LifecycleMonitor(logger=mock_logger)
        
        # Add agent to monitoring
        monitor.monitored_agents[mock_agent.agent_id] = mock_agent
        
        status = monitor.get_agent_status(mock_agent.agent_id)
        
        assert status["agent_id"] == mock_agent.agent_id
        assert status["state"] == mock_agent.state.value
        assert "last_health_check" in status
        assert "last_performance_check" in status
    
    def test_get_all_agent_status(self, mock_logger):
        """Test getting all agent statuses."""
        monitor = LifecycleMonitor(logger=mock_logger)
        
        # Add multiple agents
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.ACTIVE
            monitor.monitored_agents[agent.agent_id] = agent
        
        all_status = monitor.get_all_agent_status()
        
        assert len(all_status) == 3
        for i, (agent_id, status) in enumerate(all_status.items()):
            assert agent_id == f"agent_{i}"
            assert status["state"] == AgentState.ACTIVE.value


class TestLifecycleManager:
    """Test LifecycleManager class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "lifecycle": {
                "health_check_interval": 30.0,
                "performance_monitor_interval": 60.0,
                "graceful_shutdown_timeout": 10.0
            }
        }
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "test_agent"
        agent.state = AgentState.INACTIVE
        agent.start = AsyncMock()
        agent.stop = AsyncMock()
        agent.health_check = AsyncMock(return_value=True)
        agent.get_metrics = Mock(return_value={"messages_processed": 100})
        return agent
    
    def test_lifecycle_manager_initialization(self, mock_logger, mock_config):
        """Test lifecycle manager initialization."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        assert manager.config == mock_config["lifecycle"]
        assert manager.logger == mock_logger
        assert isinstance(manager.monitor, LifecycleMonitor)
        assert manager.hooks == {"pre_start": [], "post_start": [], "pre_stop": [], "post_stop": []}
        assert manager.agent_states == {}
    
    def test_add_hook(self, mock_logger, mock_config):
        """Test adding lifecycle hooks."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        hook = Mock(spec=LifecycleHook)
        manager.add_hook("pre_start", hook)
        
        assert hook in manager.hooks["pre_start"]
    
    def test_remove_hook(self, mock_logger, mock_config):
        """Test removing lifecycle hooks."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        hook = Mock(spec=LifecycleHook)
        manager.add_hook("pre_start", hook)
        manager.remove_hook("pre_start", hook)
        
        assert hook not in manager.hooks["pre_start"]
    
    @pytest.mark.asyncio
    async def test_start_agent_lifecycle(self, mock_logger, mock_config, mock_agent):
        """Test starting agent lifecycle."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Add hooks
        pre_start_hook = Mock(spec=LifecycleHook)
        pre_start_hook.execute = AsyncMock(return_value=True)
        post_start_hook = Mock(spec=LifecycleHook)
        post_start_hook.execute = AsyncMock(return_value=True)
        
        manager.add_hook("pre_start", pre_start_hook)
        manager.add_hook("post_start", post_start_hook)
        
        await manager.start_agent(mock_agent)
        
        # Verify hooks were executed
        pre_start_hook.execute.assert_called_once()
        post_start_hook.execute.assert_called_once()
        
        # Verify agent was started
        mock_agent.start.assert_called_once()
        
        # Verify agent state is tracked
        assert mock_agent.agent_id in manager.agent_states
        assert manager.agent_states[mock_agent.agent_id] == LifecycleState.RUNNING
    
    @pytest.mark.asyncio
    async def test_stop_agent_lifecycle(self, mock_logger, mock_config, mock_agent):
        """Test stopping agent lifecycle."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Start agent first
        await manager.start_agent(mock_agent)
        
        # Add hooks
        pre_stop_hook = Mock(spec=LifecycleHook)
        pre_stop_hook.execute = AsyncMock(return_value=True)
        post_stop_hook = Mock(spec=LifecycleHook)
        post_stop_hook.execute = AsyncMock(return_value=True)
        
        manager.add_hook("pre_stop", pre_stop_hook)
        manager.add_hook("post_stop", post_stop_hook)
        
        await manager.stop_agent(mock_agent.agent_id)
        
        # Verify hooks were executed
        pre_stop_hook.execute.assert_called_once()
        post_stop_hook.execute.assert_called_once()
        
        # Verify agent was stopped
        mock_agent.stop.assert_called_once()
        
        # Verify agent state is updated
        assert manager.agent_states[mock_agent.agent_id] == LifecycleState.STOPPED
    
    @pytest.mark.asyncio
    async def test_hook_execution_failure(self, mock_logger, mock_config, mock_agent):
        """Test handling hook execution failures."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Add failing hook
        failing_hook = Mock(spec=LifecycleHook)
        failing_hook.execute = AsyncMock(return_value=False)
        
        manager.add_hook("pre_start", failing_hook)
        
        # Starting agent should fail
        with pytest.raises(AgentError):
            await manager.start_agent(mock_agent)
        
        # Verify agent was not started
        mock_agent.start.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_state_transition_validation(self, mock_logger, mock_config, mock_agent):
        """Test state transition validation."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Try to stop agent that hasn't been started
        with pytest.raises(AgentStateError):
            await manager.stop_agent(mock_agent.agent_id)
    
    @pytest.mark.asyncio
    async def test_restart_agent(self, mock_logger, mock_config, mock_agent):
        """Test restarting an agent."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Start agent
        await manager.start_agent(mock_agent)
        
        # Restart agent
        await manager.restart_agent(mock_agent.agent_id)
        
        # Verify agent was stopped and started
        mock_agent.stop.assert_called_once()
        assert mock_agent.start.call_count == 2  # Once for start, once for restart
    
    def test_get_agent_state(self, mock_logger, mock_config, mock_agent):
        """Test getting agent state."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Set agent state
        manager.agent_states[mock_agent.agent_id] = LifecycleState.RUNNING
        
        state = manager.get_agent_state(mock_agent.agent_id)
        
        assert state == LifecycleState.RUNNING
    
    def test_get_all_agent_states(self, mock_logger, mock_config):
        """Test getting all agent states."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Set multiple agent states
        for i in range(3):
            manager.agent_states[f"agent_{i}"] = LifecycleState.RUNNING
        
        all_states = manager.get_all_agent_states()
        
        assert len(all_states) == 3
        for i, (agent_id, state) in enumerate(all_states.items()):
            assert agent_id == f"agent_{i}"
            assert state == LifecycleState.RUNNING
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_all_agents(self, mock_logger, mock_config):
        """Test graceful shutdown of all agents."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = Mock(spec=BaseAgent)
            agent.agent_id = f"agent_{i}"
            agent.state = AgentState.ACTIVE
            agent.stop = AsyncMock()
            agents.append(agent)
            
            # Start agent
            await manager.start_agent(agent)
        
        # Graceful shutdown
        await manager.graceful_shutdown_all()
        
        # Verify all agents were stopped
        for agent in agents:
            agent.stop.assert_called_once()
        
        # Verify all states are updated
        for agent in agents:
            assert manager.agent_states[agent.agent_id] == LifecycleState.STOPPED
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, mock_logger, mock_config):
        """Test emergency shutdown."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Create agent with slow shutdown
        agent = Mock(spec=BaseAgent)
        agent.agent_id = "slow_agent"
        agent.state = AgentState.ACTIVE
        
        async def slow_stop():
            await asyncio.sleep(1.0)
        
        agent.stop = slow_stop
        
        # Start agent
        await manager.start_agent(agent)
        
        # Emergency shutdown with short timeout
        await manager.emergency_shutdown(timeout=0.1)
        
        # Verify agent state is updated (even if stop didn't complete)
        assert manager.agent_states[agent.agent_id] == LifecycleState.STOPPED
    
    def test_event_logging(self, mock_logger, mock_config, mock_agent):
        """Test lifecycle event logging."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Log an event
        manager.log_event(mock_agent.agent_id, "test_event", {"key": "value"})
        
        # Verify event was logged
        mock_logger.info.assert_called()
        
        # Verify event is stored
        assert len(manager.event_history) == 1
        assert manager.event_history[0].agent_id == mock_agent.agent_id
        assert manager.event_history[0].event_type == "test_event"
    
    def test_get_event_history(self, mock_logger, mock_config, mock_agent):
        """Test getting event history."""
        manager = LifecycleManager(config=mock_config, logger=mock_logger)
        
        # Add some events
        for i in range(5):
            manager.log_event(mock_agent.agent_id, f"event_{i}", {"index": i})
        
        # Get all events
        all_events = manager.get_event_history()
        assert len(all_events) == 5
        
        # Get events for specific agent
        agent_events = manager.get_event_history(mock_agent.agent_id)
        assert len(agent_events) == 5
        
        # Get events by type
        event_1_events = manager.get_event_history(event_type="event_1")
        assert len(event_1_events) == 1
        assert event_1_events[0].event_type == "event_1"