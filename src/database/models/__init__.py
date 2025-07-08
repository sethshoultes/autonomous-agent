"""
Database models for the autonomous agent system.

This module provides SQLModel-based database models for all agent data types:
- User accounts and authentication
- Email processing and analytics
- Research findings and knowledge base
- Code review history and metrics
- Agent performance and usage statistics
- Audit trails and compliance data
"""

from .base import BaseModel, TimestampMixin
from .users import User, UserProfile, UserPreference, UserSession
from .emails import Email, EmailThread, EmailAnalytics, EmailAttachment
from .research import ResearchQuery, ResearchResult, ResearchSource, KnowledgeBase
from .code import CodeReview, CodeMetrics, CodeRepository, CodeChange
from .agents import Agent, AgentTask, AgentMetrics, AgentConfiguration
from .intelligence import Decision, TaskPlan, LearningEvent, Pattern
from .audit import AuditLog, SecurityEvent, ComplianceRecord
from .analytics import PerformanceMetric, UsageStatistic, SystemHealth

__all__ = [
    # Base models
    "BaseModel",
    "TimestampMixin",
    
    # User models
    "User",
    "UserProfile",
    "UserPreference",
    "UserSession",
    
    # Email models
    "Email",
    "EmailThread",
    "EmailAnalytics",
    "EmailAttachment",
    
    # Research models
    "ResearchQuery",
    "ResearchResult",
    "ResearchSource",
    "KnowledgeBase",
    
    # Code models
    "CodeReview",
    "CodeMetrics",
    "CodeRepository",
    "CodeChange",
    
    # Agent models
    "Agent",
    "AgentTask",
    "AgentMetrics",
    "AgentConfiguration",
    
    # Intelligence models
    "Decision",
    "TaskPlan",
    "LearningEvent",
    "Pattern",
    
    # Audit models
    "AuditLog",
    "SecurityEvent",
    "ComplianceRecord",
    
    # Analytics models
    "PerformanceMetric",
    "UsageStatistic",
    "SystemHealth",
]