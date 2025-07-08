"""
Custom exceptions for the Autonomous Agent System.

This module defines all custom exceptions used throughout the system,
promoting clear error handling and debugging.
"""

import time
from typing import Any, Dict, Optional


class BaseAgentException(Exception):
    """Base exception class with enhanced error context."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception with context and cause tracking.

        Args:
            message: Error message
            context: Additional context information
            cause: Root cause exception if this is a wrapped exception
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = time.time()

    def __str__(self) -> str:
        """Return a detailed string representation of the exception."""
        parts = [self.message]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "timestamp": self.timestamp,
        }


class AgentError(BaseAgentException):
    """Base exception for all agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_name: str = "Unknown",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            agent_name: Name of the agent that caused the error
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {"agent_name": agent_name}
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, enhanced_context, cause)
        self.agent_name = agent_name


class ServiceError(BaseAgentException):
    """Base exception for all service-related errors."""

    def __init__(
        self,
        message: str,
        service_name: str = "Unknown",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            service_name: Name of the service that caused the error
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {"service_name": service_name}
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, enhanced_context, cause)
        self.service_name = service_name


class ConfigurationError(BaseAgentException):
    """Exception raised when there's a configuration error."""

    def __init__(
        self,
        message: str,
        config_key: str = "Unknown",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {"config_key": config_key}
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, enhanced_context, cause)
        self.config_key = config_key


class ValidationError(BaseAgentException):
    """Exception raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str = "Unknown",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            field_name: Name of the field that failed validation
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {"field_name": field_name}
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, enhanced_context, cause)
        self.field_name = field_name


# Add additional specialized exceptions
class RetryableError(BaseAgentException):
    """Exception for errors that can be retried."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        max_retries: int = 3,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            max_retries: Maximum number of retries allowed
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {
            "retry_after": retry_after,
            "max_retries": max_retries,
        }
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, enhanced_context, cause)
        self.retry_after = retry_after
        self.max_retries = max_retries


class RateLimitError(RetryableError):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            limit: Rate limit number
            window: Rate limit window in seconds
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {
            "limit": limit,
            "window": window,
        }
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, retry_after, 5, enhanced_context, cause)
        self.limit = limit
        self.window = window


class AuthenticationError(BaseAgentException):
    """Exception raised for authentication failures."""
    pass


class AuthorizationError(BaseAgentException):
    """Exception raised for authorization failures."""
    pass


class TimeoutError(RetryableError):
    """Exception raised when operations timeout."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_duration: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            timeout_duration: Duration that caused the timeout
            context: Additional context information
            cause: Root cause exception
        """
        enhanced_context = {"timeout_duration": timeout_duration}
        if context:
            enhanced_context.update(context)
        
        super().__init__(message, None, 3, enhanced_context, cause)
        self.timeout_duration = timeout_duration
