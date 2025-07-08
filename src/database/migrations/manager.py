"""
Database migration manager.

This module provides comprehensive migration management including execution,
rollback, versioning, and migration tracking.
"""

import logging
import os
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..connection import get_database_manager
from ..models.base import BaseModel
from .base import BaseMigration, MigrationStatus
from ...core.exceptions import CoreError


class MigrationError(CoreError):
    """Base exception for migration errors."""
    pass


class MigrationNotFoundError(MigrationError):
    """Exception raised when a migration is not found."""
    pass


class MigrationAlreadyAppliedError(MigrationError):
    """Exception raised when attempting to apply an already applied migration."""
    pass


Base = declarative_base()


class MigrationRecord(Base):
    """Database table for tracking applied migrations."""
    
    __tablename__ = "migration_history"
    
    id = Column(Integer, primary_key=True)
    migration_name = Column(String(255), unique=True, nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow)
    rolled_back_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="applied")
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)


class MigrationManager:
    """
    Database migration manager.
    
    Manages database schema migrations with support for versioning,
    rollback, and migration tracking.
    """
    
    def __init__(self, migrations_dir: str, logger: logging.Logger):
        """
        Initialize migration manager.
        
        Args:
            migrations_dir: Directory containing migration files
            logger: Logger instance for migration operations
        """
        self.migrations_dir = Path(migrations_dir)
        self.logger = logger
        self._db_manager = None
        self._migrations_cache = {}
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py if it doesn't exist
        init_file = self.migrations_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
    
    async def _get_db_manager(self):
        """Get database manager instance."""
        if self._db_manager is None:
            self._db_manager = await get_database_manager()
        return self._db_manager
    
    async def initialize(self) -> None:
        """Initialize the migration system."""
        try:
            self.logger.info("Initializing migration system...")
            
            # Create migration history table
            await self._create_migration_table()
            
            # Load available migrations
            await self._load_migrations()
            
            self.logger.info("Migration system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize migration system: {e}")
            raise MigrationError(f"Migration system initialization failed: {e}")
    
    async def _create_migration_table(self) -> None:
        """Create the migration history table if it doesn't exist."""
        db_manager = await self._get_db_manager()
        
        # Use synchronous engine for table creation
        sync_engine = create_engine(
            db_manager.database_url.replace("postgresql+asyncpg://", "postgresql://")
        )
        
        Base.metadata.create_all(sync_engine)
        
        self.logger.debug("Migration history table created/verified")
    
    async def _load_migrations(self) -> None:
        """Load all migration files from the migrations directory."""
        self._migrations_cache = {}
        
        # Find all Python files in migrations directory
        migration_files = sorted(
            self.migrations_dir.glob("*.py"),
            key=lambda p: p.stem
        )
        
        for migration_file in migration_files:
            if migration_file.stem.startswith("__"):
                continue
            
            try:
                # Load migration module
                spec = importlib.util.spec_from_file_location(
                    f"migrations.{migration_file.stem}",
                    migration_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find migration class
                migration_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type) and
                        issubclass(attr, BaseMigration) and
                        attr != BaseMigration
                    ):
                        migration_class = attr
                        break
                
                if migration_class:
                    self._migrations_cache[migration_file.stem] = migration_class
                    self.logger.debug(f"Loaded migration: {migration_file.stem}")
                else:
                    self.logger.warning(f"No migration class found in {migration_file}")
                    
            except Exception as e:
                self.logger.error(f"Error loading migration {migration_file}: {e}")
                raise MigrationError(f"Failed to load migration {migration_file}: {e}")
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration names."""
        db_manager = await self._get_db_manager()
        
        query = """
        SELECT migration_name 
        FROM migration_history 
        WHERE status = 'applied' 
        ORDER BY applied_at
        """
        
        result = await db_manager.execute_query(query)
        return [row['migration_name'] for row in result]
    
    async def get_pending_migrations(self) -> List[str]:
        """Get list of pending migration names."""
        applied_migrations = await self.get_applied_migrations()
        all_migrations = sorted(self._migrations_cache.keys())
        
        return [name for name in all_migrations if name not in applied_migrations]
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status."""
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()
        
        return {
            "total_migrations": len(self._migrations_cache),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_migrations": applied,
            "pending_migrations": pending,
            "last_applied": applied[-1] if applied else None,
            "next_pending": pending[0] if pending else None
        }
    
    async def create_migration(self, name: str, description: str = "") -> str:
        """Create a new migration file."""
        # Generate timestamp prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        migration_name = f"{timestamp}_{name}"
        migration_file = self.migrations_dir / f"{migration_name}.py"
        
        # Check if migration already exists
        if migration_file.exists():
            raise MigrationError(f"Migration {migration_name} already exists")
        
        # Generate migration template
        template = f'''"""
{description or f"Migration: {name}"}

Created: {datetime.now().isoformat()}
"""

from typing import Any, Dict, List

from ..base import BaseMigration, MigrationOperation
from ..operations import *


class Migration(BaseMigration):
    """
    {description or f"Migration: {name}"}
    """
    
    def __init__(self):
        super().__init__(
            name="{migration_name}",
            description="{description or f"Migration: {name}"}",
            version="{timestamp}"
        )
    
    async def up(self) -> List[MigrationOperation]:
        """Apply the migration."""
        return [
            # Add your migration operations here
            # Example:
            # CreateTableOperation(
            #     table_name="example_table",
            #     columns=[
            #         {{"name": "id", "type": "UUID", "primary_key": True}},
            #         {{"name": "name", "type": "VARCHAR(255)", "nullable": False}},
            #         {{"name": "created_at", "type": "TIMESTAMP", "default": "NOW()"}}
            #     ]
            # ),
            ExecuteSQLOperation("-- Add your SQL here")
        ]
    
    async def down(self) -> List[MigrationOperation]:
        """Rollback the migration."""
        return [
            # Add your rollback operations here
            # Example:
            # DropTableOperation(table_name="example_table"),
            ExecuteSQLOperation("-- Add your rollback SQL here")
        ]
    
    async def validate(self) -> bool:
        """Validate the migration before applying."""
        # Add custom validation logic here
        return True
'''
        
        # Write migration file
        migration_file.write_text(template)
        
        self.logger.info(f"Created migration: {migration_name}")
        return migration_name
    
    async def apply_migration(self, migration_name: str) -> None:
        """Apply a specific migration."""
        if migration_name not in self._migrations_cache:
            raise MigrationNotFoundError(f"Migration {migration_name} not found")
        
        # Check if already applied
        applied_migrations = await self.get_applied_migrations()
        if migration_name in applied_migrations:
            raise MigrationAlreadyAppliedError(f"Migration {migration_name} already applied")
        
        start_time = datetime.now()
        
        try:
            # Create migration instance
            migration_class = self._migrations_cache[migration_name]
            migration = migration_class()
            
            # Validate migration
            if not await migration.validate():
                raise MigrationError(f"Migration {migration_name} validation failed")
            
            self.logger.info(f"Applying migration: {migration_name}")
            
            # Get migration operations
            operations = await migration.up()
            
            # Execute operations
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_connection() as conn:
                async with conn.transaction():
                    for operation in operations:
                        await operation.execute(conn)
            
            # Record successful migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self._record_migration(migration_name, "applied", execution_time)
            
            self.logger.info(f"Successfully applied migration: {migration_name}")
            
        except Exception as e:
            # Record failed migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self._record_migration(migration_name, "failed", execution_time, str(e))
            
            self.logger.error(f"Failed to apply migration {migration_name}: {e}")
            raise MigrationError(f"Migration {migration_name} failed: {e}")
    
    async def rollback_migration(self, migration_name: str) -> None:
        """Rollback a specific migration."""
        if migration_name not in self._migrations_cache:
            raise MigrationNotFoundError(f"Migration {migration_name} not found")
        
        # Check if migration is applied
        applied_migrations = await self.get_applied_migrations()
        if migration_name not in applied_migrations:
            raise MigrationError(f"Migration {migration_name} is not applied")
        
        start_time = datetime.now()
        
        try:
            # Create migration instance
            migration_class = self._migrations_cache[migration_name]
            migration = migration_class()
            
            self.logger.info(f"Rolling back migration: {migration_name}")
            
            # Get rollback operations
            operations = await migration.down()
            
            # Execute operations
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_connection() as conn:
                async with conn.transaction():
                    for operation in operations:
                        await operation.execute(conn)
            
            # Record rollback
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self._record_rollback(migration_name, execution_time)
            
            self.logger.info(f"Successfully rolled back migration: {migration_name}")
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            await self._record_migration(migration_name, "rollback_failed", execution_time, str(e))
            
            self.logger.error(f"Failed to rollback migration {migration_name}: {e}")
            raise MigrationError(f"Migration rollback {migration_name} failed: {e}")
    
    async def apply_all_pending(self) -> List[str]:
        """Apply all pending migrations."""
        pending_migrations = await self.get_pending_migrations()
        applied = []
        
        for migration_name in pending_migrations:
            try:
                await self.apply_migration(migration_name)
                applied.append(migration_name)
            except Exception as e:
                self.logger.error(f"Stopping migration chain at {migration_name}: {e}")
                break
        
        return applied
    
    async def rollback_to(self, target_migration: str) -> List[str]:
        """Rollback to a specific migration."""
        applied_migrations = await self.get_applied_migrations()
        
        # Find target migration index
        try:
            target_index = applied_migrations.index(target_migration)
        except ValueError:
            raise MigrationError(f"Target migration {target_migration} not found in applied migrations")
        
        # Get migrations to rollback (in reverse order)
        migrations_to_rollback = applied_migrations[target_index + 1:]
        migrations_to_rollback.reverse()
        
        rolled_back = []
        
        for migration_name in migrations_to_rollback:
            try:
                await self.rollback_migration(migration_name)
                rolled_back.append(migration_name)
            except Exception as e:
                self.logger.error(f"Stopping rollback chain at {migration_name}: {e}")
                break
        
        return rolled_back
    
    async def _record_migration(
        self,
        migration_name: str,
        status: str,
        execution_time: int,
        error_message: Optional[str] = None
    ) -> None:
        """Record migration in history table."""
        db_manager = await self._get_db_manager()
        
        query = """
        INSERT INTO migration_history (migration_name, status, execution_time_ms, error_message)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (migration_name) 
        DO UPDATE SET 
            status = EXCLUDED.status,
            execution_time_ms = EXCLUDED.execution_time_ms,
            error_message = EXCLUDED.error_message,
            applied_at = NOW()
        """
        
        await db_manager.execute_query(
            query,
            {
                "migration_name": migration_name,
                "status": status,
                "execution_time_ms": execution_time,
                "error_message": error_message
            }
        )
    
    async def _record_rollback(self, migration_name: str, execution_time: int) -> None:
        """Record migration rollback."""
        db_manager = await self._get_db_manager()
        
        query = """
        UPDATE migration_history 
        SET 
            status = 'rolled_back',
            rolled_back_at = NOW(),
            execution_time_ms = $2
        WHERE migration_name = $1
        """
        
        await db_manager.execute_query(
            query,
            {"migration_name": migration_name, "execution_time_ms": execution_time}
        )
    
    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get complete migration history."""
        db_manager = await self._get_db_manager()
        
        query = """
        SELECT 
            migration_name,
            applied_at,
            rolled_back_at,
            status,
            error_message,
            execution_time_ms
        FROM migration_history
        ORDER BY applied_at DESC
        """
        
        result = await db_manager.execute_query(query)
        return [dict(row) for row in result]
    
    async def validate_schema(self) -> Dict[str, Any]:
        """Validate current database schema against migrations."""
        # This would implement schema validation logic
        # For now, return basic status
        status = await self.get_migration_status()
        
        return {
            "valid": status["pending_count"] == 0,
            "issues": [] if status["pending_count"] == 0 else [
                f"{status['pending_count']} pending migrations"
            ],
            "migration_status": status
        }
    
    async def cleanup_failed_migrations(self) -> int:
        """Clean up failed migration records."""
        db_manager = await self._get_db_manager()
        
        query = """
        DELETE FROM migration_history 
        WHERE status IN ('failed', 'rollback_failed')
        """
        
        result = await db_manager.execute_query(query)
        return len(result) if result else 0