"""
Comprehensive tests for lifecycle hooks.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.lifecycle.hooks import (
    LifecycleHook,
    PreStartHook,
    PostStartHook,
    PreStopHook,
    PostStopHook
)


class TestLifecycleHookInterface:
    """Test the abstract LifecycleHook interface."""
    
    def test_lifecycle_hook_is_abstract(self):
        """Test that LifecycleHook cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LifecycleHook()
    
    def test_lifecycle_hook_execute_is_abstract(self):
        """Test that execute method is abstract."""
        class ConcreteHook(LifecycleHook):
            pass
        
        with pytest.raises(TypeError):
            ConcreteHook()


class TestPreStartHook:
    """Test PreStartHook functionality."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()
    
    @pytest.fixture
    def hook(self, mock_logger):
        """Create a PreStartHook instance."""
        return PreStartHook(logger=mock_logger)
    
    @pytest.fixture
    def valid_context(self):
        """Create a valid context for testing."""
        return {
            "agent": Mock(),
            "config": {
                "agent_type": "TestAgent",
                "enabled": True,
                "priority": 5,
                "config": {
                    "timeout": 30,
                    "max_retries": 3
                }
            }
        }
    
    def test_pre_start_hook_initialization(self, hook):
        """Test PreStartHook initialization."""
        assert hook.logger is not None
    
    def test_pre_start_hook_initialization_default_logger(self):
        """Test PreStartHook initialization with default logger."""
        hook = PreStartHook()
        assert hook.logger is not None
    
    @pytest.mark.asyncio
    async def test_execute_with_valid_context(self, hook, valid_context):
        """Test execute with valid context."""
        with patch.object(hook, '_validate_agent_config', return_value=True), \
             patch.object(hook, '_check_resource_availability', new_callable=AsyncMock, return_value=True):
            
            result = await hook.execute("test_agent", valid_context)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_config(self, hook, valid_context):
        """Test execute with invalid config."""
        with patch.object(hook, '_validate_agent_config', return_value=False):
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_with_insufficient_resources(self, hook, valid_context):
        """Test execute with insufficient resources."""
        with patch.object(hook, '_validate_agent_config', return_value=True), \
             patch.object(hook, '_check_resource_availability', new_callable=AsyncMock, return_value=False):
            
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_agent(self, hook):
        """Test execute with missing agent in context."""
        context = {"config": {"agent_type": "TestAgent"}}
        result = await hook.execute("test_agent", context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_config(self, hook):
        """Test execute with missing config in context."""
        context = {"agent": Mock()}
        result = await hook.execute("test_agent", context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_with_exception(self, hook, valid_context):
        """Test execute when an exception occurs."""
        with patch.object(hook, '_validate_agent_config', side_effect=Exception("Test error")):
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    def test_validate_agent_config_valid(self, hook):
        """Test _validate_agent_config with valid config."""
        config = {
            "agent_type": "TestAgent",
            "enabled": True,
            "priority": 5,
            "config": {}
        }
        assert hook._validate_agent_config(config) is True
    
    def test_validate_agent_config_missing_agent_type(self, hook):
        """Test _validate_agent_config with missing agent_type."""
        config = {"enabled": True, "priority": 5}
        assert hook._validate_agent_config(config) is False
    
    def test_validate_agent_config_invalid_agent_type(self, hook):
        """Test _validate_agent_config with invalid agent_type."""
        config = {"agent_type": "", "enabled": True, "priority": 5}
        assert hook._validate_agent_config(config) is False
    
    def test_validate_agent_config_missing_enabled(self, hook):
        """Test _validate_agent_config with missing enabled field."""
        config = {"agent_type": "TestAgent", "priority": 5}
        assert hook._validate_agent_config(config) is False
    
    def test_validate_agent_config_invalid_enabled(self, hook):
        """Test _validate_agent_config with invalid enabled field."""
        config = {"agent_type": "TestAgent", "enabled": "yes", "priority": 5}
        assert hook._validate_agent_config(config) is False
    
    def test_validate_agent_config_invalid_priority(self, hook):
        """Test _validate_agent_config with invalid priority."""
        config = {"agent_type": "TestAgent", "enabled": True, "priority": "high"}
        assert hook._validate_agent_config(config) is False
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_sufficient(self, hook):
        """Test _check_resource_availability with sufficient resources."""
        context = {"agent": Mock()}
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value = Mock(percent=50.0)
            mock_disk.return_value = Mock(percent=60.0)
            
            result = await hook._check_resource_availability("test_agent", context)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_insufficient_memory(self, hook):
        """Test _check_resource_availability with insufficient memory."""
        context = {"agent": Mock()}
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value = Mock(percent=95.0)
            mock_disk.return_value = Mock(percent=60.0)
            
            result = await hook._check_resource_availability("test_agent", context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_insufficient_disk(self, hook):
        """Test _check_resource_availability with insufficient disk space."""
        context = {"agent": Mock()}
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value = Mock(percent=50.0)
            mock_disk.return_value = Mock(percent=98.0)
            
            result = await hook._check_resource_availability("test_agent", context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_psutil_error(self, hook):
        """Test _check_resource_availability when psutil raises an error."""
        context = {"agent": Mock()}
        
        with patch('psutil.virtual_memory', side_effect=Exception("psutil error")):
            result = await hook._check_resource_availability("test_agent", context)
            assert result is False


class TestPostStartHook:
    """Test PostStartHook functionality."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()
    
    @pytest.fixture
    def hook(self, mock_logger):
        """Create a PostStartHook instance."""
        return PostStartHook(logger=mock_logger)
    
    @pytest.fixture
    def valid_context(self):
        """Create a valid context for testing."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value=True)
        return {
            "agent": mock_agent,
            "config": {"agent_type": "TestAgent"}
        }
    
    def test_post_start_hook_initialization(self, hook):
        """Test PostStartHook initialization."""
        assert hook.logger is not None
    
    @pytest.mark.asyncio
    async def test_execute_success(self, hook, valid_context):
        """Test successful execute."""
        with patch.object(hook, '_register_agent', new_callable=AsyncMock, return_value=True), \
             patch.object(hook, '_verify_agent_responsiveness', new_callable=AsyncMock, return_value=True), \
             patch.object(hook, '_initialize_metrics', new_callable=AsyncMock), \
             patch.object(hook, '_send_startup_notification', new_callable=AsyncMock):
            
            result = await hook.execute("test_agent", valid_context)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_execute_registration_failure(self, hook, valid_context):
        """Test execute when agent registration fails."""
        with patch.object(hook, '_register_agent', new_callable=AsyncMock, return_value=False):
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_responsiveness_failure(self, hook, valid_context):
        """Test execute when agent responsiveness check fails."""
        with patch.object(hook, '_register_agent', new_callable=AsyncMock, return_value=True), \
             patch.object(hook, '_verify_agent_responsiveness', new_callable=AsyncMock, return_value=False):
            
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_missing_agent(self, hook):
        """Test execute with missing agent."""
        context = {"config": {"agent_type": "TestAgent"}}
        result = await hook.execute("test_agent", context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_exception(self, hook, valid_context):
        """Test execute when exception occurs."""
        with patch.object(hook, '_register_agent', new_callable=AsyncMock, side_effect=Exception("Test error")):
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_register_agent_success(self, hook, valid_context):
        """Test _register_agent success."""
        # This is a placeholder since _register_agent just logs
        result = await hook._register_agent("test_agent", valid_context["agent"], valid_context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_agent_responsiveness_success(self, hook):
        """Test _verify_agent_responsiveness with responsive agent."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value=True)
        
        result = await hook._verify_agent_responsiveness(mock_agent)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_agent_responsiveness_failure(self, hook):
        """Test _verify_agent_responsiveness with unresponsive agent."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value=False)
        
        result = await hook._verify_agent_responsiveness(mock_agent)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_verify_agent_responsiveness_no_health_check(self, hook):
        """Test _verify_agent_responsiveness with agent that has no health_check."""
        mock_agent = Mock(spec=[])  # Agent without health_check method
        
        result = await hook._verify_agent_responsiveness(mock_agent)
        assert result is True  # Should pass if no health_check method
    
    @pytest.mark.asyncio
    async def test_verify_agent_responsiveness_exception(self, hook):
        """Test _verify_agent_responsiveness when health_check raises exception."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        
        result = await hook._verify_agent_responsiveness(mock_agent)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_metrics(self, hook):
        """Test _initialize_metrics."""
        mock_agent = Mock()
        # This method just logs, so we test that it doesn't raise an exception
        await hook._initialize_metrics("test_agent", mock_agent)
    
    @pytest.mark.asyncio
    async def test_send_startup_notification(self, hook):
        """Test _send_startup_notification."""
        context = {"config": {"agent_type": "TestAgent"}}
        # This method just logs, so we test that it doesn't raise an exception
        await hook._send_startup_notification("test_agent", context)


class TestPreStopHook:
    """Test PreStopHook functionality."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()
    
    @pytest.fixture
    def hook(self, mock_logger):
        """Create a PreStopHook instance."""
        return PreStopHook(logger=mock_logger, graceful_shutdown_timeout=5.0)
    
    @pytest.fixture
    def valid_context(self):
        """Create a valid context for testing."""
        mock_agent = Mock()
        mock_agent.stop = AsyncMock()
        return {
            "agent": mock_agent,
            "config": {"agent_type": "TestAgent"}
        }
    
    def test_pre_stop_hook_initialization(self, hook):
        """Test PreStopHook initialization."""
        assert hook.logger is not None
        assert hook.graceful_shutdown_timeout == 5.0
    
    def test_pre_stop_hook_default_timeout(self):
        """Test PreStopHook with default timeout."""
        hook = PreStopHook()
        assert hook.graceful_shutdown_timeout == 10.0
    
    @pytest.mark.asyncio
    async def test_execute_success(self, hook, valid_context):
        """Test successful execute."""
        with patch.object(hook, '_save_agent_state', new_callable=AsyncMock), \
             patch.object(hook, '_send_shutdown_notification', new_callable=AsyncMock):
            
            result = await hook.execute("test_agent", valid_context)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_execute_missing_agent(self, hook):
        """Test execute with missing agent."""
        context = {"config": {"agent_type": "TestAgent"}}
        result = await hook.execute("test_agent", context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_exception(self, hook, valid_context):
        """Test execute when exception occurs."""
        with patch.object(hook, '_save_agent_state', new_callable=AsyncMock, side_effect=Exception("Test error")):
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_save_agent_state(self, hook):
        """Test _save_agent_state."""
        mock_agent = Mock()
        # This method just logs, so we test that it doesn't raise an exception
        await hook._save_agent_state("test_agent", mock_agent)
    
    @pytest.mark.asyncio
    async def test_send_shutdown_notification(self, hook):
        """Test _send_shutdown_notification."""
        context = {"config": {"agent_type": "TestAgent"}}
        # This method just logs, so we test that it doesn't raise an exception
        await hook._send_shutdown_notification("test_agent", context)


class TestPostStopHook:
    """Test PostStopHook functionality."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()
    
    @pytest.fixture
    def hook(self, mock_logger):
        """Create a PostStopHook instance."""
        return PostStopHook(logger=mock_logger)
    
    @pytest.fixture
    def valid_context(self):
        """Create a valid context for testing."""
        mock_agent = Mock()
        return {
            "agent": mock_agent,
            "config": {"agent_type": "TestAgent"}
        }
    
    def test_post_stop_hook_initialization(self, hook):
        """Test PostStopHook initialization."""
        assert hook.logger is not None
    
    @pytest.mark.asyncio
    async def test_execute_success(self, hook, valid_context):
        """Test successful execute."""
        with patch.object(hook, '_deregister_agent', new_callable=AsyncMock), \
             patch.object(hook, '_archive_agent_data', new_callable=AsyncMock), \
             patch.object(hook, '_send_stop_notification', new_callable=AsyncMock):
            
            result = await hook.execute("test_agent", valid_context)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_execute_missing_agent(self, hook):
        """Test execute with missing agent."""
        context = {"config": {"agent_type": "TestAgent"}}
        result = await hook.execute("test_agent", context)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_exception(self, hook, valid_context):
        """Test execute when exception occurs."""
        with patch.object(hook, '_deregister_agent', new_callable=AsyncMock, side_effect=Exception("Test error")):
            result = await hook.execute("test_agent", valid_context)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_deregister_agent(self, hook):
        """Test _deregister_agent."""
        context = {"config": {"agent_type": "TestAgent"}}
        # This method just logs, so we test that it doesn't raise an exception
        await hook._deregister_agent("test_agent", context)
    
    @pytest.mark.asyncio
    async def test_archive_agent_data(self, hook):
        """Test _archive_agent_data."""
        mock_agent = Mock()
        # This method just logs, so we test that it doesn't raise an exception
        await hook._archive_agent_data("test_agent", mock_agent)
    
    @pytest.mark.asyncio
    async def test_send_stop_notification(self, hook):
        """Test _send_stop_notification."""
        context = {"config": {"agent_type": "TestAgent"}}
        # This method just logs, so we test that it doesn't raise an exception
        await hook._send_stop_notification("test_agent", context)


class TestHookIntegration:
    """Test hook integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_hooks_sequence(self):
        """Test a complete lifecycle hooks sequence."""
        mock_agent = Mock()
        mock_agent.health_check = AsyncMock(return_value=True)
        mock_agent.stop = AsyncMock()
        
        context = {
            "agent": mock_agent,
            "config": {
                "agent_type": "TestAgent",
                "enabled": True,
                "priority": 5,
                "config": {}
            }
        }
        
        # Test pre-start hook
        pre_start = PreStartHook()
        assert await pre_start.execute("test_agent", context) is True
        
        # Test post-start hook
        post_start = PostStartHook()
        assert await post_start.execute("test_agent", context) is True
        
        # Test pre-stop hook
        pre_stop = PreStopHook()
        assert await pre_stop.execute("test_agent", context) is True
        
        # Test post-stop hook
        post_stop = PostStopHook()
        assert await post_stop.execute("test_agent", context) is True
    
    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """Test hook error handling in various scenarios."""
        hooks = [PreStartHook(), PostStartHook(), PreStopHook(), PostStopHook()]
        
        # Test with empty context
        for hook in hooks:
            result = await hook.execute("test_agent", {})
            assert result is False
        
        # Test with None context
        for hook in hooks:
            result = await hook.execute("test_agent", None)
            assert result is False