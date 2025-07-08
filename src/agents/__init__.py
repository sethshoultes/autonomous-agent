"""
Agents module for the autonomous agent system.
"""

from .base import AgentInterface, AgentMessage, AgentState, BaseAgent
from .exceptions import (
    AgentCommunicationError,
    AgentError,
    AgentManagerError,
    AgentNotFoundError,
    AgentRegistrationError,
    AgentStateError,
)
from .manager import AgentConfig, AgentManager, AgentRegistry
from .research import (
    ResearchAgent,
    ResearchTask,
    ResearchResult,
    ContentItem,
    ResearchReport,
    ResearchQuery,
    ResearchException,
    RateLimitError,
    ContentExtractionError,
    CacheError,
)

__all__ = [
    "AgentCommunicationError",
    "AgentConfig",
    # Exceptions
    "AgentError",
    "AgentInterface",
    # Manager classes
    "AgentManager",
    "AgentManagerError",
    "AgentMessage",
    "AgentNotFoundError",
    "AgentRegistrationError",
    "AgentRegistry",
    "AgentState",
    "AgentStateError",
    # Base classes
    "BaseAgent",
    # Research Agent
    "ResearchAgent",
    "ResearchTask",
    "ResearchResult",
    "ContentItem",
    "ResearchReport",
    "ResearchQuery",
    "ResearchException",
    "RateLimitError",
    "ContentExtractionError",
    "CacheError",
]
