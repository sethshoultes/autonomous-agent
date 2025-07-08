"""
Base classes for the Autonomous Agent System.

This module defines the fundamental abstract base classes that enforce
SOLID principles throughout the system architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    """Base entity with common fields for all domain objects."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        validate_assignment = True


class BaseAgent(ABC):
    """
    Abstract base class for all autonomous agents.

    This class enforces the Single Responsibility Principle by defining
    a clear contract for agent behavior while allowing for specific
    implementations.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the agent with a name and optional configuration.

        Args:
            name: The agent's name
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._is_running = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """
        Execute the agent's main logic.

        Args:
            input_data: Input data for the agent to process

        Returns:
            The result of the agent's execution
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and shutdown the agent."""
        pass

    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._is_running

    async def start(self) -> None:
        """Start the agent if not already running."""
        if not self._is_running:
            await self.initialize()
            self._is_running = True

    async def stop(self) -> None:
        """Stop the agent if currently running."""
        if self._is_running:
            await self.shutdown()
            self._is_running = False


class BaseService(ABC):
    """
    Abstract base class for all services in the system.

    This class promotes the Dependency Inversion Principle by defining
    a common interface for all services.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the service with a name.

        Args:
            name: The service's name
        """
        self.name = name

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check the health of the service.

        Returns:
            True if the service is healthy, False otherwise
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the service."""
        pass
