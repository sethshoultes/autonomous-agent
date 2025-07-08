"""
Tests for the core.base module.

This module contains tests for the base classes that form the foundation
of the autonomous agent system.
"""

import pytest
from unittest.mock import AsyncMock
from typing import Any

from src.core.base import BaseAgent, BaseService, BaseEntity


class TestBaseEntity:
    """Test cases for the BaseEntity class."""

    def test_base_entity_creation(self) -> None:
        """Test that BaseEntity can be created with default values."""
        entity = BaseEntity()
        
        assert entity.id is not None
        assert len(entity.id) == 36  # UUID4 format
        assert entity.created_at is None
        assert entity.updated_at is None

    def test_base_entity_with_custom_values(self) -> None:
        """Test that BaseEntity can be created with custom values."""
        entity = BaseEntity(
            id="custom-id",
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-01T12:00:00Z",
        )
        
        assert entity.id == "custom-id"
        assert entity.created_at == "2023-01-01T00:00:00Z"
        assert entity.updated_at == "2023-01-01T12:00:00Z"


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, name: str, config: dict = None) -> None:
        super().__init__(name, config)
        self.initialized = False
        self.executed_data = None
        self.shutdown_called = False

    async def initialize(self) -> None:
        """Initialize the test agent."""
        self.initialized = True

    async def execute(self, input_data: Any) -> Any:
        """Execute the test agent."""
        self.executed_data = input_data
        return f"Processed: {input_data}"

    async def shutdown(self) -> None:
        """Shutdown the test agent."""
        self.shutdown_called = True


class TestBaseAgent:
    """Test cases for the BaseAgent abstract class."""

    def test_agent_initialization(self) -> None:
        """Test that agent can be initialized with name and config."""
        config = {"key": "value"}
        agent = ConcreteAgent("TestAgent", config)
        
        assert agent.name == "TestAgent"
        assert agent.config == config
        assert agent.is_running is False

    def test_agent_initialization_without_config(self) -> None:
        """Test that agent can be initialized without config."""
        agent = ConcreteAgent("TestAgent")
        
        assert agent.name == "TestAgent"
        assert agent.config == {}
        assert agent.is_running is False

    @pytest.mark.asyncio
    async def test_agent_start_and_stop(self) -> None:
        """Test that agent can be started and stopped."""
        agent = ConcreteAgent("TestAgent")
        
        # Initially not running
        assert agent.is_running is False
        assert agent.initialized is False
        
        # Start the agent
        await agent.start()
        assert agent.is_running is True
        assert agent.initialized is True
        
        # Stop the agent
        await agent.stop()
        assert agent.is_running is False
        assert agent.shutdown_called is True

    @pytest.mark.asyncio
    async def test_agent_start_idempotent(self) -> None:
        """Test that starting an already running agent is idempotent."""
        agent = ConcreteAgent("TestAgent")
        
        await agent.start()
        assert agent.is_running is True
        
        # Starting again should not cause issues
        await agent.start()
        assert agent.is_running is True

    @pytest.mark.asyncio
    async def test_agent_stop_idempotent(self) -> None:
        """Test that stopping a non-running agent is idempotent."""
        agent = ConcreteAgent("TestAgent")
        
        # Stop without starting
        await agent.stop()
        assert agent.is_running is False
        assert agent.shutdown_called is False

    @pytest.mark.asyncio
    async def test_agent_execute(self) -> None:
        """Test that agent can execute with input data."""
        agent = ConcreteAgent("TestAgent")
        test_data = {"test": "data"}
        
        result = await agent.execute(test_data)
        
        assert result == "Processed: {'test': 'data'}"
        assert agent.executed_data == test_data


class ConcreteService(BaseService):
    """Concrete implementation of BaseService for testing."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.is_healthy = True
        self.initialized = False
        self.cleaned_up = False

    async def health_check(self) -> bool:
        """Check the health of the test service."""
        return self.is_healthy

    async def initialize(self) -> None:
        """Initialize the test service."""
        self.initialized = True

    async def cleanup(self) -> None:
        """Clean up the test service."""
        self.cleaned_up = True


class TestBaseService:
    """Test cases for the BaseService abstract class."""

    def test_service_initialization(self) -> None:
        """Test that service can be initialized with name."""
        service = ConcreteService("TestService")
        
        assert service.name == "TestService"

    @pytest.mark.asyncio
    async def test_service_health_check(self) -> None:
        """Test that service health check works."""
        service = ConcreteService("TestService")
        
        # Initially healthy
        assert await service.health_check() is True
        
        # Make unhealthy
        service.is_healthy = False
        assert await service.health_check() is False

    @pytest.mark.asyncio
    async def test_service_initialize_and_cleanup(self) -> None:
        """Test that service can be initialized and cleaned up."""
        service = ConcreteService("TestService")
        
        # Initially not initialized
        assert service.initialized is False
        assert service.cleaned_up is False
        
        # Initialize
        await service.initialize()
        assert service.initialized is True
        
        # Cleanup
        await service.cleanup()
        assert service.cleaned_up is True