"""
Database migration system.

This module provides comprehensive database migration capabilities including
schema versioning, migration execution, rollback support, and migration management.
"""

from .manager import MigrationManager
from .base import BaseMigration, MigrationOperation
from .operations import (
    CreateTableOperation,
    DropTableOperation,
    AddColumnOperation,
    DropColumnOperation,
    CreateIndexOperation,
    DropIndexOperation,
    AlterColumnOperation,
    RenameTableOperation,
    RenameColumnOperation,
    ExecuteSQLOperation,
)

__all__ = [
    "MigrationManager",
    "BaseMigration",
    "MigrationOperation",
    "CreateTableOperation",
    "DropTableOperation",
    "AddColumnOperation",
    "DropColumnOperation",
    "CreateIndexOperation",
    "DropIndexOperation",
    "AlterColumnOperation",
    "RenameTableOperation",
    "RenameColumnOperation",
    "ExecuteSQLOperation",
]