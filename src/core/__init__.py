"""
Core module for the Autonomous Agent System.

This module contains the fundamental components and abstractions
that form the backbone of the autonomous agent system.
"""

from .base import BaseAgent, BaseService
from .exceptions import AgentError, ConfigurationError, ServiceError

__all__ = [
    "AgentError",
    "BaseAgent",
    "BaseService",
    "ConfigurationError",
    "ServiceError",
]
