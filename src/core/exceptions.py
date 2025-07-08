"""
Custom exceptions for the Autonomous Agent System.

This module defines all custom exceptions used throughout the system,
promoting clear error handling and debugging.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, agent_name: str = "Unknown") -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            agent_name: Name of the agent that caused the error
        """
        super().__init__(message)
        self.agent_name = agent_name
        self.message = message

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return f"Agent '{self.agent_name}': {self.message}"


class ServiceError(Exception):
    """Base exception for all service-related errors."""

    def __init__(self, message: str, service_name: str = "Unknown") -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            service_name: Name of the service that caused the error
        """
        super().__init__(message)
        self.service_name = service_name
        self.message = message

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return f"Service '{self.service_name}': {self.message}"


class ConfigurationError(Exception):
    """Exception raised when there's a configuration error."""

    def __init__(self, message: str, config_key: str = "Unknown") -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
        """
        super().__init__(message)
        self.config_key = config_key
        self.message = message

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return f"Configuration error for '{self.config_key}': {self.message}"


class ValidationError(Exception):
    """Exception raised when data validation fails."""

    def __init__(self, message: str, field_name: str = "Unknown") -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            field_name: Name of the field that failed validation
        """
        super().__init__(message)
        self.field_name = field_name
        self.message = message

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return f"Validation error for '{self.field_name}': {self.message}"
