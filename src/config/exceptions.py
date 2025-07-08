"""
Configuration-specific exceptions.
"""

from typing import Any, Dict, Optional


class ConfigError(Exception):
    """Base exception for configuration-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.context = context or {}
        self.cause = cause


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigNotFoundError(ConfigError):
    """Exception raised when configuration is not found."""
    pass
