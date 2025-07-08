"""
Base database models and mixins.

This module provides common functionality for all database models including
timestamps, UUID primary keys, and common model behaviors.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import Field
from sqlmodel import Column, DateTime, SQLModel, text
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID


class TimestampMixin(SQLModel):
    """Mixin for models that need timestamp tracking."""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
        description="Timestamp when the record was created"
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            DateTime(timezone=True),
            onupdate=func.now(),
            server_default=func.now()
        ),
        description="Timestamp when the record was last updated"
    )


class BaseModel(TimestampMixin, SQLModel):
    """
    Base model class with common fields and behaviors.
    
    All database models should inherit from this class to ensure
    consistent structure and behavior across the system.
    """
    
    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column=Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4),
        description="Unique identifier for the record"
    )
    
    metadata_: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="metadata",
        description="Additional metadata stored as JSON"
    )
    
    class Config:
        """Model configuration."""
        arbitrary_types_allowed = True
        use_enum_values = True
        validate_assignment = True
        populate_by_name = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            field: getattr(self, field) 
            for field in self.__fields__.keys()
            if hasattr(self, field)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Create model instance from dictionary."""
        return cls(**data)
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model instance from dictionary."""
        for field, value in data.items():
            if hasattr(self, field):
                setattr(self, field, value)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


class SoftDeleteMixin(SQLModel):
    """Mixin for models that support soft deletion."""
    
    is_deleted: bool = Field(
        default=False,
        description="Whether the record is soft-deleted"
    )
    
    deleted_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the record was soft-deleted"
    )
    
    def soft_delete(self) -> None:
        """Mark the record as soft-deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class VersionedMixin(SQLModel):
    """Mixin for models that need version tracking."""
    
    version: int = Field(
        default=1,
        description="Version number for optimistic locking"
    )
    
    def increment_version(self) -> None:
        """Increment the version number."""
        self.version += 1


class SearchableMixin(SQLModel):
    """Mixin for models that need full-text search capabilities."""
    
    search_vector: Optional[str] = Field(
        default=None,
        description="Full-text search vector"
    )
    
    def update_search_vector(self, *fields: str) -> None:
        """Update the search vector from specified fields."""
        # This would be implemented to update the PostgreSQL search vector
        pass


class AuditMixin(SQLModel):
    """Mixin for models that need audit tracking."""
    
    created_by: Optional[UUID] = Field(
        default=None,
        description="ID of the user who created this record"
    )
    
    updated_by: Optional[UUID] = Field(
        default=None,
        description="ID of the user who last updated this record"
    )
    
    def set_created_by(self, user_id: UUID) -> None:
        """Set the creator of the record."""
        self.created_by = user_id
    
    def set_updated_by(self, user_id: UUID) -> None:
        """Set the updater of the record."""
        self.updated_by = user_id