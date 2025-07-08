"""
Database operations package.

This package provides comprehensive CRUD operations for all database models
with proper error handling, transaction management, and performance optimization.
"""

from .base import BaseRepository, QueryBuilder, TransactionManager
from .users import UserRepository, UserProfileRepository, UserPreferenceRepository
from .emails import EmailRepository, EmailThreadRepository, EmailAnalyticsRepository
from .research import ResearchRepository, KnowledgeBaseRepository
from .agents import AgentRepository, AgentTaskRepository, AgentMetricsRepository
from .analytics import AnalyticsRepository
from .search import SearchRepository

__all__ = [
    # Base operations
    "BaseRepository",
    "QueryBuilder",
    "TransactionManager",
    
    # User operations
    "UserRepository",
    "UserProfileRepository", 
    "UserPreferenceRepository",
    
    # Email operations
    "EmailRepository",
    "EmailThreadRepository",
    "EmailAnalyticsRepository",
    
    # Research operations
    "ResearchRepository",
    "KnowledgeBaseRepository",
    
    # Agent operations
    "AgentRepository",
    "AgentTaskRepository",
    "AgentMetricsRepository",
    
    # Analytics operations
    "AnalyticsRepository",
    
    # Search operations
    "SearchRepository",
]