"""
Base database operations and utilities.

This module provides base classes and utilities for database operations
including generic CRUD operations, query building, and transaction management.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlmodel import SQLModel

from ..connection import get_database_manager
from ..models.base import BaseModel
from ...core.exceptions import CoreError


class DatabaseOperationError(CoreError):
    """Base exception for database operation errors."""
    pass


class RecordNotFoundError(DatabaseOperationError):
    """Exception raised when a record is not found."""
    pass


class ValidationError(DatabaseOperationError):
    """Exception raised for validation errors."""
    pass


class TransactionError(DatabaseOperationError):
    """Exception raised for transaction errors."""
    pass


ModelType = TypeVar("ModelType", bound=BaseModel)


class QueryBuilder:
    """
    Query builder for constructing complex database queries.
    
    Provides a fluent interface for building SQLAlchemy queries with
    filtering, sorting, pagination, and relationship loading.
    """
    
    def __init__(self, model_class: Type[ModelType]):
        """Initialize query builder with model class."""
        self.model_class = model_class
        self._query = select(model_class)
        self._filters = []
        self._orders = []
        self._limit = None
        self._offset = None
        self._includes = []
    
    def filter(self, *conditions) -> "QueryBuilder":
        """Add filter conditions to the query."""
        self._filters.extend(conditions)
        return self
    
    def filter_by(self, **kwargs) -> "QueryBuilder":
        """Add filter conditions using keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self.model_class, key):
                self._filters.append(getattr(self.model_class, key) == value)
        return self
    
    def where(self, condition) -> "QueryBuilder":
        """Add where condition to the query."""
        self._filters.append(condition)
        return self
    
    def order_by(self, *columns) -> "QueryBuilder":
        """Add ordering to the query."""
        self._orders.extend(columns)
        return self
    
    def order_by_desc(self, column) -> "QueryBuilder":
        """Add descending order to the query."""
        self._orders.append(desc(column))
        return self
    
    def limit(self, limit: int) -> "QueryBuilder":
        """Set query limit."""
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> "QueryBuilder":
        """Set query offset."""
        self._offset = offset
        return self
    
    def paginate(self, page: int, per_page: int) -> "QueryBuilder":
        """Add pagination to the query."""
        self._limit = per_page
        self._offset = (page - 1) * per_page
        return self
    
    def include(self, *relationships) -> "QueryBuilder":
        """Include relationships in the query."""
        self._includes.extend(relationships)
        return self
    
    def search(self, query: str, *fields) -> "QueryBuilder":
        """Add full-text search to the query."""
        if not fields:
            # Use default searchable fields if available
            fields = getattr(self.model_class, '__searchable_fields__', [])
        
        if fields:
            search_conditions = []
            for field in fields:
                if hasattr(self.model_class, field):
                    column = getattr(self.model_class, field)
                    search_conditions.append(column.ilike(f"%{query}%"))
            
            if search_conditions:
                self._filters.append(or_(*search_conditions))
        
        return self
    
    def build(self):
        """Build the final query."""
        query = self._query
        
        # Apply filters
        if self._filters:
            query = query.where(and_(*self._filters))
        
        # Apply ordering
        if self._orders:
            query = query.order_by(*self._orders)
        
        # Apply includes
        if self._includes:
            query = query.options(*[selectinload(rel) for rel in self._includes])
        
        # Apply limit and offset
        if self._limit is not None:
            query = query.limit(self._limit)
        
        if self._offset is not None:
            query = query.offset(self._offset)
        
        return query


class TransactionManager:
    """
    Transaction manager for handling database transactions.
    
    Provides context managers for transaction handling with proper
    rollback on errors and commit on success.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize transaction manager with session."""
        self.session = session
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        try:
            # Start transaction
            await self.session.begin()
            
            yield self.session
            
            # Commit transaction
            await self.session.commit()
            
        except Exception as e:
            # Rollback transaction on error
            await self.session.rollback()
            raise TransactionError(f"Transaction failed: {e}")
    
    @asynccontextmanager
    async def savepoint(self, name: str = None):
        """Context manager for savepoints within transactions."""
        savepoint = None
        try:
            # Create savepoint
            savepoint = await self.session.begin_nested()
            
            yield savepoint
            
            # Commit savepoint
            await savepoint.commit()
            
        except Exception as e:
            # Rollback to savepoint on error
            if savepoint:
                await savepoint.rollback()
            raise TransactionError(f"Savepoint '{name}' failed: {e}")


class BaseRepository(ABC):
    """
    Base repository class for database operations.
    
    Provides common CRUD operations and utilities that can be extended
    by specific model repositories.
    """
    
    def __init__(self, model_class: Type[ModelType], logger: logging.Logger):
        """Initialize repository with model class and logger."""
        self.model_class = model_class
        self.logger = logger
        self._db_manager = None
    
    async def _get_db_manager(self):
        """Get database manager instance."""
        if self._db_manager is None:
            self._db_manager = await get_database_manager()
        return self._db_manager
    
    async def _get_session(self) -> AsyncSession:
        """Get database session."""
        db_manager = await self._get_db_manager()
        return db_manager.get_session()
    
    def query(self) -> QueryBuilder:
        """Create a new query builder for this model."""
        return QueryBuilder(self.model_class)
    
    async def create(self, **kwargs) -> ModelType:
        """Create a new record."""
        try:
            async with (await self._get_session()) as session:
                instance = self.model_class(**kwargs)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)
                
                self.logger.debug(f"Created {self.model_class.__name__} with ID: {instance.id}")
                return instance
                
        except Exception as e:
            self.logger.error(f"Error creating {self.model_class.__name__}: {e}")
            raise DatabaseOperationError(f"Failed to create {self.model_class.__name__}: {e}")
    
    async def get_by_id(self, id: UUID) -> Optional[ModelType]:
        """Get a record by ID."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(self.model_class).where(self.model_class.id == id)
                )
                return result.scalar_one_or_none()
                
        except Exception as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by ID {id}: {e}")
            raise DatabaseOperationError(f"Failed to get {self.model_class.__name__}: {e}")
    
    async def get_by_id_or_raise(self, id: UUID) -> ModelType:
        """Get a record by ID or raise exception if not found."""
        instance = await self.get_by_id(id)
        if instance is None:
            raise RecordNotFoundError(f"{self.model_class.__name__} with ID {id} not found")
        return instance
    
    async def get_many(self, ids: List[UUID]) -> List[ModelType]:
        """Get multiple records by IDs."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(self.model_class).where(self.model_class.id.in_(ids))
                )
                return result.scalars().all()
                
        except Exception as e:
            self.logger.error(f"Error getting multiple {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to get {self.model_class.__name__} records: {e}")
    
    async def update(self, id: UUID, **kwargs) -> Optional[ModelType]:
        """Update a record by ID."""
        try:
            async with (await self._get_session()) as session:
                instance = await session.get(self.model_class, id)
                if instance is None:
                    return None
                
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                
                await session.commit()
                await session.refresh(instance)
                
                self.logger.debug(f"Updated {self.model_class.__name__} with ID: {id}")
                return instance
                
        except Exception as e:
            self.logger.error(f"Error updating {self.model_class.__name__} with ID {id}: {e}")
            raise DatabaseOperationError(f"Failed to update {self.model_class.__name__}: {e}")
    
    async def update_or_raise(self, id: UUID, **kwargs) -> ModelType:
        """Update a record by ID or raise exception if not found."""
        instance = await self.update(id, **kwargs)
        if instance is None:
            raise RecordNotFoundError(f"{self.model_class.__name__} with ID {id} not found")
        return instance
    
    async def delete(self, id: UUID) -> bool:
        """Delete a record by ID."""
        try:
            async with (await self._get_session()) as session:
                instance = await session.get(self.model_class, id)
                if instance is None:
                    return False
                
                await session.delete(instance)
                await session.commit()
                
                self.logger.debug(f"Deleted {self.model_class.__name__} with ID: {id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting {self.model_class.__name__} with ID {id}: {e}")
            raise DatabaseOperationError(f"Failed to delete {self.model_class.__name__}: {e}")
    
    async def delete_or_raise(self, id: UUID) -> None:
        """Delete a record by ID or raise exception if not found."""
        deleted = await self.delete(id)
        if not deleted:
            raise RecordNotFoundError(f"{self.model_class.__name__} with ID {id} not found")
    
    async def soft_delete(self, id: UUID) -> bool:
        """Soft delete a record by ID (if model supports it)."""
        if not hasattr(self.model_class, 'soft_delete'):
            raise DatabaseOperationError(f"{self.model_class.__name__} does not support soft delete")
        
        try:
            async with (await self._get_session()) as session:
                instance = await session.get(self.model_class, id)
                if instance is None:
                    return False
                
                instance.soft_delete()
                await session.commit()
                
                self.logger.debug(f"Soft deleted {self.model_class.__name__} with ID: {id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error soft deleting {self.model_class.__name__} with ID {id}: {e}")
            raise DatabaseOperationError(f"Failed to soft delete {self.model_class.__name__}: {e}")
    
    async def list_all(self, include_deleted: bool = False) -> List[ModelType]:
        """List all records."""
        try:
            async with (await self._get_session()) as session:
                query = select(self.model_class)
                
                # Filter out soft-deleted records if applicable
                if hasattr(self.model_class, 'is_deleted') and not include_deleted:
                    query = query.where(self.model_class.is_deleted == False)
                
                result = await session.execute(query)
                return result.scalars().all()
                
        except Exception as e:
            self.logger.error(f"Error listing {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to list {self.model_class.__name__} records: {e}")
    
    async def list_paginated(
        self,
        page: int = 1,
        per_page: int = 20,
        include_deleted: bool = False
    ) -> Dict[str, Any]:
        """List records with pagination."""
        try:
            async with (await self._get_session()) as session:
                # Build base query
                query = select(self.model_class)
                count_query = select(func.count(self.model_class.id))
                
                # Filter out soft-deleted records if applicable
                if hasattr(self.model_class, 'is_deleted') and not include_deleted:
                    query = query.where(self.model_class.is_deleted == False)
                    count_query = count_query.where(self.model_class.is_deleted == False)
                
                # Get total count
                total_result = await session.execute(count_query)
                total = total_result.scalar()
                
                # Apply pagination
                offset = (page - 1) * per_page
                query = query.limit(per_page).offset(offset)
                
                # Execute query
                result = await session.execute(query)
                items = result.scalars().all()
                
                # Calculate pagination metadata
                total_pages = (total + per_page - 1) // per_page
                has_next = page < total_pages
                has_prev = page > 1
                
                return {
                    "items": items,
                    "total": total,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev
                }
                
        except Exception as e:
            self.logger.error(f"Error listing paginated {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to list paginated {self.model_class.__name__} records: {e}")
    
    async def count(self, include_deleted: bool = False) -> int:
        """Count total records."""
        try:
            async with (await self._get_session()) as session:
                query = select(func.count(self.model_class.id))
                
                # Filter out soft-deleted records if applicable
                if hasattr(self.model_class, 'is_deleted') and not include_deleted:
                    query = query.where(self.model_class.is_deleted == False)
                
                result = await session.execute(query)
                return result.scalar()
                
        except Exception as e:
            self.logger.error(f"Error counting {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to count {self.model_class.__name__} records: {e}")
    
    async def exists(self, id: UUID) -> bool:
        """Check if a record exists."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(func.count(self.model_class.id)).where(self.model_class.id == id)
                )
                return result.scalar() > 0
                
        except Exception as e:
            self.logger.error(f"Error checking existence of {self.model_class.__name__} with ID {id}: {e}")
            raise DatabaseOperationError(f"Failed to check existence of {self.model_class.__name__}: {e}")
    
    async def find_by(self, **kwargs) -> List[ModelType]:
        """Find records by field values."""
        try:
            async with (await self._get_session()) as session:
                query = select(self.model_class)
                
                for key, value in kwargs.items():
                    if hasattr(self.model_class, key):
                        query = query.where(getattr(self.model_class, key) == value)
                
                result = await session.execute(query)
                return result.scalars().all()
                
        except Exception as e:
            self.logger.error(f"Error finding {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to find {self.model_class.__name__} records: {e}")
    
    async def find_one_by(self, **kwargs) -> Optional[ModelType]:
        """Find a single record by field values."""
        results = await self.find_by(**kwargs)
        return results[0] if results else None
    
    async def find_one_by_or_raise(self, **kwargs) -> ModelType:
        """Find a single record by field values or raise exception."""
        instance = await self.find_one_by(**kwargs)
        if instance is None:
            raise RecordNotFoundError(f"{self.model_class.__name__} not found with criteria: {kwargs}")
        return instance
    
    async def bulk_create(self, records: List[Dict[str, Any]]) -> List[ModelType]:
        """Create multiple records in bulk."""
        try:
            async with (await self._get_session()) as session:
                instances = [self.model_class(**record) for record in records]
                session.add_all(instances)
                await session.commit()
                
                # Refresh all instances to get generated IDs
                for instance in instances:
                    await session.refresh(instance)
                
                self.logger.debug(f"Bulk created {len(instances)} {self.model_class.__name__} records")
                return instances
                
        except Exception as e:
            self.logger.error(f"Error bulk creating {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to bulk create {self.model_class.__name__} records: {e}")
    
    async def bulk_update(self, updates: List[Dict[str, Any]]) -> List[ModelType]:
        """Update multiple records in bulk."""
        try:
            async with (await self._get_session()) as session:
                updated_instances = []
                
                for update in updates:
                    id = update.pop('id', None)
                    if id is None:
                        continue
                    
                    instance = await session.get(self.model_class, id)
                    if instance is None:
                        continue
                    
                    for key, value in update.items():
                        if hasattr(instance, key):
                            setattr(instance, key, value)
                    
                    updated_instances.append(instance)
                
                await session.commit()
                
                # Refresh all instances
                for instance in updated_instances:
                    await session.refresh(instance)
                
                self.logger.debug(f"Bulk updated {len(updated_instances)} {self.model_class.__name__} records")
                return updated_instances
                
        except Exception as e:
            self.logger.error(f"Error bulk updating {self.model_class.__name__} records: {e}")
            raise DatabaseOperationError(f"Failed to bulk update {self.model_class.__name__} records: {e}")
    
    async def execute_query(self, query_builder: QueryBuilder) -> List[ModelType]:
        """Execute a query builder query."""
        try:
            async with (await self._get_session()) as session:
                query = query_builder.build()
                result = await session.execute(query)
                return result.scalars().all()
                
        except Exception as e:
            self.logger.error(f"Error executing query for {self.model_class.__name__}: {e}")
            raise DatabaseOperationError(f"Failed to execute query for {self.model_class.__name__}: {e}")
    
    async def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query."""
        try:
            db_manager = await self._get_db_manager()
            return await db_manager.execute_query(query, params)
            
        except Exception as e:
            self.logger.error(f"Error executing raw query: {e}")
            raise DatabaseOperationError(f"Failed to execute raw query: {e}")
    
    @abstractmethod
    async def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository-specific statistics."""
        pass