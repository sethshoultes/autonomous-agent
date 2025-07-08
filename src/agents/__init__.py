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
try:
    from .code_agent import CodeAgent
    CODE_AGENT_AVAILABLE = True
except ImportError:
    CODE_AGENT_AVAILABLE = False
    CodeAgent = None

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

# Conditionally add CodeAgent to __all__ if available
if CODE_AGENT_AVAILABLE:
    __all__.append("CodeAgent")
