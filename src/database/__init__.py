"""
Database package for the autonomous agent system.

This package provides comprehensive PostgreSQL integration with:
- Async connection management and pooling
- SQLModel-based type-safe database operations
- Schema migrations and version control
- Performance optimization and monitoring
- Data archiving and retention policies
"""

from .connection import DatabaseManager, get_database_manager
from .models import *
from .operations import *
from .migrations import MigrationManager
from .performance import PerformanceManager
from .archive import ArchiveManager

__all__ = [
    "DatabaseManager",
    "get_database_manager",
    "MigrationManager",
    "PerformanceManager",
    "ArchiveManager",
]