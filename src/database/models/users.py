"""
User-related database models.

This module defines models for user accounts, authentication, preferences,
and session management in the autonomous agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import EmailStr, Field, validator
from sqlmodel import Column, ForeignKey, Index, Relationship, SQLModel, String, Text
from sqlalchemy import Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from .base import BaseModel, SoftDeleteMixin, AuditMixin


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    AGENT = "agent"


class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(BaseModel, SoftDeleteMixin, AuditMixin, table=True):
    """
    User account model.
    
    Represents a user in the autonomous agent system with authentication
    and authorization capabilities.
    """
    
    __tablename__ = "users"
    
    # Basic user information
    email: EmailStr = Field(
        unique=True,
        index=True,
        description="User's email address (unique)"
    )
    
    username: str = Field(
        unique=True,
        index=True,
        min_length=3,
        max_length=50,
        description="Unique username"
    )
    
    full_name: str = Field(
        max_length=100,
        description="User's full name"
    )
    
    # Authentication
    password_hash: str = Field(
        description="Hashed password"
    )
    
    password_salt: str = Field(
        description="Password salt"
    )
    
    # Authorization
    role: UserRole = Field(
        default=UserRole.USER,
        description="User role for authorization"
    )
    
    permissions: List[str] = Field(
        default=[],
        sa_column=Column(JSONB),
        description="List of specific permissions"
    )
    
    # Status and verification
    status: UserStatus = Field(
        default=UserStatus.PENDING,
        description="User account status"
    )
    
    is_email_verified: bool = Field(
        default=False,
        description="Whether email has been verified"
    )
    
    email_verified_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when email was verified"
    )
    
    # Security settings
    two_factor_enabled: bool = Field(
        default=False,
        description="Whether 2FA is enabled"
    )
    
    two_factor_secret: Optional[str] = Field(
        default=None,
        description="2FA secret key"
    )
    
    # Login tracking
    last_login_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last successful login"
    )
    
    last_login_ip: Optional[str] = Field(
        default=None,
        description="IP address of last login"
    )
    
    failed_login_attempts: int = Field(
        default=0,
        description="Number of consecutive failed login attempts"
    )
    
    locked_until: Optional[datetime] = Field(
        default=None,
        description="Account lock expiration time"
    )
    
    # Preferences
    timezone: str = Field(
        default="UTC",
        description="User's timezone preference"
    )
    
    locale: str = Field(
        default="en",
        description="User's locale/language preference"
    )
    
    # Relationships
    profile: Optional["UserProfile"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )
    
    preferences: List["UserPreference"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )
    
    sessions: List["UserSession"] = Relationship(
        back_populates="user",
        cascade_delete=True
    )
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if not v:
            raise ValueError('Email is required')
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v:
            raise ValueError('Username is required')
        if not v.replace('_', '').replace('-', '').replace('.', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, underscores, hyphens, and dots')
        return v.lower()
    
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE and not self.is_deleted
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def can_login(self) -> bool:
        """Check if user can login."""
        return self.is_active() and not self.is_locked()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def add_permission(self, permission: str) -> None:
        """Add a permission to the user."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str) -> None:
        """Remove a permission from the user."""
        if permission in self.permissions:
            self.permissions.remove(permission)


class UserProfile(BaseModel, table=True):
    """
    Extended user profile information.
    
    Stores additional user information that's not critical for authentication
    but useful for personalization and user experience.
    """
    
    __tablename__ = "user_profiles"
    
    # Foreign key to user
    user_id: UUID = Field(
        foreign_key="users.id",
        unique=True,
        description="Reference to the user"
    )
    
    # Personal information
    first_name: Optional[str] = Field(
        default=None,
        max_length=50,
        description="User's first name"
    )
    
    last_name: Optional[str] = Field(
        default=None,
        max_length=50,
        description="User's last name"
    )
    
    display_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Preferred display name"
    )
    
    bio: Optional[str] = Field(
        default=None,
        max_length=500,
        description="User biography"
    )
    
    avatar_url: Optional[str] = Field(
        default=None,
        description="URL to user's avatar image"
    )
    
    # Contact information
    phone: Optional[str] = Field(
        default=None,
        description="User's phone number"
    )
    
    location: Optional[str] = Field(
        default=None,
        description="User's location"
    )
    
    # Professional information
    company: Optional[str] = Field(
        default=None,
        description="User's company"
    )
    
    job_title: Optional[str] = Field(
        default=None,
        description="User's job title"
    )
    
    # Social links
    website: Optional[str] = Field(
        default=None,
        description="User's website"
    )
    
    github_username: Optional[str] = Field(
        default=None,
        description="GitHub username"
    )
    
    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL"
    )
    
    # Preferences
    email_notifications: bool = Field(
        default=True,
        description="Whether to receive email notifications"
    )
    
    push_notifications: bool = Field(
        default=True,
        description="Whether to receive push notifications"
    )
    
    # Additional data
    custom_fields: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Custom profile fields"
    )
    
    # Relationships
    user: User = Relationship(back_populates="profile")


class UserPreference(BaseModel, table=True):
    """
    User preference settings.
    
    Stores user-specific configuration and preferences for the autonomous
    agent system behavior.
    """
    
    __tablename__ = "user_preferences"
    
    # Foreign key to user
    user_id: UUID = Field(
        foreign_key="users.id",
        description="Reference to the user"
    )
    
    # Preference identification
    category: str = Field(
        max_length=100,
        description="Preference category (e.g., 'email', 'research')"
    )
    
    key: str = Field(
        max_length=100,
        description="Preference key within category"
    )
    
    value: Any = Field(
        sa_column=Column(JSONB),
        description="Preference value (JSON)"
    )
    
    # Preference metadata
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description"
    )
    
    is_system_default: bool = Field(
        default=False,
        description="Whether this is a system default preference"
    )
    
    # Relationships
    user: User = Relationship(back_populates="preferences")
    
    class Config:
        """Model configuration."""
        # Ensure unique preference per user
        indexes = [
            Index("idx_user_preferences_user_category_key", "user_id", "category", "key", unique=True)
        ]


class UserSession(BaseModel, table=True):
    """
    User session tracking.
    
    Tracks user sessions for security and analytics purposes.
    """
    
    __tablename__ = "user_sessions"
    
    # Foreign key to user
    user_id: UUID = Field(
        foreign_key="users.id",
        description="Reference to the user"
    )
    
    # Session identification
    session_token: str = Field(
        unique=True,
        index=True,
        description="Unique session token"
    )
    
    refresh_token: Optional[str] = Field(
        default=None,
        description="Refresh token for session renewal"
    )
    
    # Session metadata
    ip_address: Optional[str] = Field(
        default=None,
        description="IP address of the session"
    )
    
    user_agent: Optional[str] = Field(
        default=None,
        description="User agent string"
    )
    
    device_info: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Device information"
    )
    
    # Session timing
    expires_at: datetime = Field(
        description="Session expiration time"
    )
    
    last_activity_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last activity timestamp"
    )
    
    # Session status
    is_active: bool = Field(
        default=True,
        description="Whether the session is active"
    )
    
    revoked_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when session was revoked"
    )
    
    revoked_by: Optional[UUID] = Field(
        default=None,
        description="ID of user who revoked the session"
    )
    
    # Relationships
    user: User = Relationship(back_populates="sessions")
    
    def is_valid(self) -> bool:
        """Check if session is valid."""
        return (
            self.is_active and
            self.expires_at > datetime.utcnow() and
            self.revoked_at is None
        )
    
    def refresh(self, new_expires_at: datetime) -> None:
        """Refresh the session with new expiration time."""
        self.expires_at = new_expires_at
        self.last_activity_at = datetime.utcnow()
    
    def revoke(self, revoked_by: Optional[UUID] = None) -> None:
        """Revoke the session."""
        self.is_active = False
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by