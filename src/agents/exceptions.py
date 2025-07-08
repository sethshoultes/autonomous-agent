"""
Custom exceptions for the autonomous agent system.
"""

from typing import Any, Dict, Optional


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.context = context or {}
        self.cause = cause


class AgentStateError(AgentError):
    """Exception raised when an invalid state transition is attempted."""
    pass


class AgentCommunicationError(AgentError):
    """Exception raised when agent communication fails."""
    pass


class AgentManagerError(AgentError):
    """Exception raised when agent manager operations fail."""
    pass


class AgentRegistrationError(AgentManagerError):
    """Exception raised when agent registration fails."""
    pass


class AgentNotFoundError(AgentManagerError):
    """Exception raised when an agent is not found."""
    pass


class ConfigError(AgentError):
    """Exception raised when configuration errors occur."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigNotFoundError(ConfigError):
    """Exception raised when configuration is not found."""
    pass


class CommunicationError(AgentError):
    """Exception raised when communication operations fail."""
    pass


class MessageValidationError(CommunicationError):
    """Exception raised when message validation fails."""
    pass


class MessageRoutingError(CommunicationError):
    """Exception raised when message routing fails."""
    pass
