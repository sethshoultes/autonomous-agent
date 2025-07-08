"""
User Management Service.

This module provides comprehensive user management functionality including
user profiles, preferences, activity tracking, and administrative operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, EmailStr, validator

from ..database.models.users import User, UserRole, UserStatus
from ..database.connection import get_database_connection
from ..monitoring.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class UserSortBy(str, Enum):
    """User sorting options."""
    CREATED_AT = "created_at"
    LAST_LOGIN = "last_login"
    USERNAME = "username"
    EMAIL = "email"
    FULL_NAME = "full_name"


class UserFilterBy(str, Enum):
    """User filtering options."""
    ALL = "all"
    ACTIVE = "active"
    INACTIVE = "inactive"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    LOCKED = "locked"
    ADMIN = "admin"
    USER = "user"


class ActivityType(str, Enum):
    """User activity types."""
    LOGIN = "login"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PROFILE_UPDATE = "profile_update"
    PREFERENCES_UPDATE = "preferences_update"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    EMAIL_VERIFIED = "email_verified"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"


class UserProfileUpdate(BaseModel):
    """User profile update request."""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    phone: Optional[str] = Field(None, max_length=20)
    location: Optional[str] = Field(None, max_length=100)
    company: Optional[str] = Field(None, max_length=100)
    job_title: Optional[str] = Field(None, max_length=100)
    website: Optional[str] = Field(None, max_length=200)
    github_username: Optional[str] = Field(None, max_length=100)
    linkedin_url: Optional[str] = Field(None, max_length=200)
    timezone: Optional[str] = Field(None, max_length=50)
    locale: Optional[str] = Field(None, max_length=10)


class UserPreferencesUpdate(BaseModel):
    """User preferences update request."""
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    agent_notifications: Optional[bool] = None
    research_privacy: Optional[str] = None
    code_review_auto_assign: Optional[bool] = None
    theme: Optional[str] = None
    language: Optional[str] = None
    custom_preferences: Optional[Dict[str, Any]] = None


class UserListQuery(BaseModel):
    """User list query parameters."""
    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)
    sort_by: UserSortBy = UserSortBy.CREATED_AT
    sort_order: str = Field("desc", regex="^(asc|desc)$")
    filter_by: UserFilterBy = UserFilterBy.ALL
    search: Optional[str] = None


class UserActivity(BaseModel):
    """User activity model."""
    id: UUID
    user_id: UUID
    activity_type: ActivityType
    activity_details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any]


class UserStats(BaseModel):
    """User statistics model."""
    total_users: int
    active_users: int
    verified_users: int
    admin_users: int
    users_last_24h: int
    users_last_7d: int
    users_last_30d: int


class UserManagementService:
    """User management service."""
    
    def __init__(self, config: Dict[str, Any], metrics_collector: Optional[MetricsCollector] = None):
        self.config = config
        self.metrics_collector = metrics_collector
        self.db = get_database_connection()
        
        # Configuration
        self.default_timezone = config.get('default_timezone', 'UTC')
        self.default_locale = config.get('default_locale', 'en')
        self.max_profile_image_size = config.get('max_profile_image_size', 5 * 1024 * 1024)  # 5MB
        
        # Activity retention
        self.activity_retention_days = config.get('activity_retention_days', 90)
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        try:
            return await self._get_user_by_id(user_id)
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {str(e)}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            return await self._get_user_by_username(username)
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            return await self._get_user_by_email(email)
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {str(e)}")
            return None
    
    async def list_users(self, query: UserListQuery) -> Tuple[List[User], int]:
        """List users with filtering and pagination."""
        try:
            users, total = await self._list_users(query)
            
            # Update metrics
            if self.metrics_collector:
                await self.metrics_collector.increment_counter(
                    'user_management_list_requests_total',
                    labels={'filter': query.filter_by.value}
                )
            
            return users, total
            
        except Exception as e:
            logger.error(f"Failed to list users: {str(e)}")
            return [], 0
    
    async def update_user_profile(self, user_id: UUID, 
                                 profile_update: UserProfileUpdate) -> bool:
        """Update user profile."""
        try:
            # Get current user
            user = await self._get_user_by_id(user_id)
            if not user:
                return False
            
            # Update profile
            success = await self._update_user_profile(user_id, profile_update)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.PROFILE_UPDATE,
                    {'updated_fields': list(profile_update.dict(exclude_unset=True).keys())}
                )
                
                # Update metrics
                if self.metrics_collector:
                    await self.metrics_collector.increment_counter(
                        'user_profile_updates_total'
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update user profile {user_id}: {str(e)}")
            return False
    
    async def update_user_preferences(self, user_id: UUID, 
                                    preferences: UserPreferencesUpdate) -> bool:
        """Update user preferences."""
        try:
            # Get current user
            user = await self._get_user_by_id(user_id)
            if not user:
                return False
            
            # Update preferences
            success = await self._update_user_preferences(user_id, preferences)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.PREFERENCES_UPDATE,
                    {'updated_preferences': list(preferences.dict(exclude_unset=True).keys())}
                )
                
                # Update metrics
                if self.metrics_collector:
                    await self.metrics_collector.increment_counter(
                        'user_preferences_updates_total'
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update user preferences {user_id}: {str(e)}")
            return False
    
    async def get_user_preferences(self, user_id: UUID) -> Dict[str, Any]:
        """Get user preferences."""
        try:
            return await self._get_user_preferences(user_id)
        except Exception as e:
            logger.error(f"Failed to get user preferences {user_id}: {str(e)}")
            return {}
    
    async def get_user_activity(self, user_id: UUID, 
                              limit: int = 50, 
                              offset: int = 0) -> List[UserActivity]:
        """Get user activity history."""
        try:
            return await self._get_user_activity(user_id, limit, offset)
        except Exception as e:
            logger.error(f"Failed to get user activity {user_id}: {str(e)}")
            return []
    
    async def log_user_activity(self, user_id: UUID, 
                              activity_type: ActivityType,
                              details: Dict[str, Any],
                              ip_address: Optional[str] = None,
                              user_agent: Optional[str] = None) -> bool:
        """Log user activity."""
        try:
            return await self._log_user_activity(
                user_id, activity_type, details, ip_address, user_agent
            )
        except Exception as e:
            logger.error(f"Failed to log user activity {user_id}: {str(e)}")
            return False
    
    async def get_user_stats(self) -> UserStats:
        """Get user statistics."""
        try:
            stats = await self._get_user_stats()
            return UserStats(**stats)
        except Exception as e:
            logger.error(f"Failed to get user stats: {str(e)}")
            return UserStats(
                total_users=0,
                active_users=0,
                verified_users=0,
                admin_users=0,
                users_last_24h=0,
                users_last_7d=0,
                users_last_30d=0
            )
    
    async def search_users(self, query: str, 
                          limit: int = 20) -> List[User]:
        """Search users by username, email, or full name."""
        try:
            return await self._search_users(query, limit)
        except Exception as e:
            logger.error(f"Failed to search users: {str(e)}")
            return []
    
    async def export_user_data(self, user_id: UUID) -> Dict[str, Any]:
        """Export user data for GDPR compliance."""
        try:
            # Get user data
            user = await self._get_user_by_id(user_id)
            if not user:
                return {}
            
            # Get related data
            profile = await self._get_user_profile(user_id)
            preferences = await self._get_user_preferences(user_id)
            activity = await self._get_user_activity(user_id, limit=1000)
            sessions = await self._get_user_sessions(user_id)
            oauth_links = await self._get_user_oauth_links(user_id)
            
            # Compile export data
            export_data = {
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'full_name': user.full_name,
                    'created_at': user.created_at.isoformat(),
                    'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None,
                    'timezone': user.timezone,
                    'locale': user.locale
                },
                'profile': profile,
                'preferences': preferences,
                'activity': [
                    {
                        'type': activity.activity_type,
                        'details': activity.activity_details,
                        'timestamp': activity.created_at.isoformat()
                    }
                    for activity in activity
                ],
                'sessions': sessions,
                'oauth_links': oauth_links,
                'exported_at': datetime.utcnow().isoformat()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export user data {user_id}: {str(e)}")
            return {}
    
    async def delete_user_data(self, user_id: UUID) -> bool:
        """Delete user data (GDPR right to be forgotten)."""
        try:
            # Soft delete user
            success = await self._soft_delete_user(user_id)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.ACCOUNT_LOCKED,
                    {'reason': 'user_requested_deletion'}
                )
                
                # Update metrics
                if self.metrics_collector:
                    await self.metrics_collector.increment_counter(
                        'user_deletions_total'
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete user data {user_id}: {str(e)}")
            return False
    
    # Administrative functions
    
    async def admin_update_user_role(self, admin_user_id: UUID, 
                                   user_id: UUID, 
                                   new_role: UserRole) -> bool:
        """Update user role (admin only)."""
        try:
            # Verify admin user
            admin_user = await self._get_user_by_id(admin_user_id)
            if not admin_user or admin_user.role != UserRole.ADMIN:
                return False
            
            # Update user role
            success = await self._update_user_role(user_id, new_role)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.PROFILE_UPDATE,
                    {
                        'role_changed_by': str(admin_user_id),
                        'new_role': new_role.value
                    }
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update user role {user_id}: {str(e)}")
            return False
    
    async def admin_lock_user(self, admin_user_id: UUID, 
                            user_id: UUID, 
                            reason: str,
                            duration: Optional[timedelta] = None) -> bool:
        """Lock user account (admin only)."""
        try:
            # Verify admin user
            admin_user = await self._get_user_by_id(admin_user_id)
            if not admin_user or admin_user.role != UserRole.ADMIN:
                return False
            
            # Lock user
            locked_until = None
            if duration:
                locked_until = datetime.utcnow() + duration
            
            success = await self._lock_user(user_id, locked_until)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.ACCOUNT_LOCKED,
                    {
                        'locked_by': str(admin_user_id),
                        'reason': reason,
                        'locked_until': locked_until.isoformat() if locked_until else None
                    }
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to lock user {user_id}: {str(e)}")
            return False
    
    async def admin_unlock_user(self, admin_user_id: UUID, 
                              user_id: UUID) -> bool:
        """Unlock user account (admin only)."""
        try:
            # Verify admin user
            admin_user = await self._get_user_by_id(admin_user_id)
            if not admin_user or admin_user.role != UserRole.ADMIN:
                return False
            
            # Unlock user
            success = await self._unlock_user(user_id)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.ACCOUNT_UNLOCKED,
                    {'unlocked_by': str(admin_user_id)}
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to unlock user {user_id}: {str(e)}")
            return False
    
    async def admin_verify_user_email(self, admin_user_id: UUID, 
                                    user_id: UUID) -> bool:
        """Verify user email (admin only)."""
        try:
            # Verify admin user
            admin_user = await self._get_user_by_id(admin_user_id)
            if not admin_user or admin_user.role != UserRole.ADMIN:
                return False
            
            # Verify email
            success = await self._verify_user_email(user_id)
            
            if success:
                # Log activity
                await self._log_user_activity(
                    user_id,
                    ActivityType.EMAIL_VERIFIED,
                    {'verified_by': str(admin_user_id)}
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to verify user email {user_id}: {str(e)}")
            return False
    
    async def cleanup_old_data(self) -> bool:
        """Clean up old user data."""
        try:
            # Clean up old activities
            cutoff_date = datetime.utcnow() - timedelta(days=self.activity_retention_days)
            await self._cleanup_old_activities(cutoff_date)
            
            # Clean up expired sessions
            await self._cleanup_expired_sessions()
            
            # Clean up old verification tokens
            await self._cleanup_old_tokens()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return False
    
    # Database operations (to be implemented with actual database calls)
    
    async def _get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        # Implementation depends on your database layer
        pass
    
    async def _list_users(self, query: UserListQuery) -> Tuple[List[User], int]:
        """List users with filtering and pagination."""
        # Implementation depends on your database layer
        pass
    
    async def _update_user_profile(self, user_id: UUID, 
                                 profile_update: UserProfileUpdate) -> bool:
        """Update user profile."""
        # Implementation depends on your database layer
        pass
    
    async def _update_user_preferences(self, user_id: UUID, 
                                     preferences: UserPreferencesUpdate) -> bool:
        """Update user preferences."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_preferences(self, user_id: UUID) -> Dict[str, Any]:
        """Get user preferences."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_profile(self, user_id: UUID) -> Dict[str, Any]:
        """Get user profile."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_activity(self, user_id: UUID, 
                               limit: int, offset: int) -> List[UserActivity]:
        """Get user activity."""
        # Implementation depends on your database layer
        pass
    
    async def _log_user_activity(self, user_id: UUID, 
                               activity_type: ActivityType,
                               details: Dict[str, Any],
                               ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None) -> bool:
        """Log user activity."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_stats(self) -> Dict[str, int]:
        """Get user statistics."""
        # Implementation depends on your database layer
        pass
    
    async def _search_users(self, query: str, limit: int) -> List[User]:
        """Search users."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_sessions(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get user sessions."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_oauth_links(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get user OAuth links."""
        # Implementation depends on your database layer
        pass
    
    async def _soft_delete_user(self, user_id: UUID) -> bool:
        """Soft delete user."""
        # Implementation depends on your database layer
        pass
    
    async def _update_user_role(self, user_id: UUID, new_role: UserRole) -> bool:
        """Update user role."""
        # Implementation depends on your database layer
        pass
    
    async def _lock_user(self, user_id: UUID, 
                        locked_until: Optional[datetime]) -> bool:
        """Lock user account."""
        # Implementation depends on your database layer
        pass
    
    async def _unlock_user(self, user_id: UUID) -> bool:
        """Unlock user account."""
        # Implementation depends on your database layer
        pass
    
    async def _verify_user_email(self, user_id: UUID) -> bool:
        """Verify user email."""
        # Implementation depends on your database layer
        pass
    
    async def _cleanup_old_activities(self, cutoff_date: datetime) -> None:
        """Clean up old activities."""
        # Implementation depends on your database layer
        pass
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        # Implementation depends on your database layer
        pass
    
    async def _cleanup_old_tokens(self) -> None:
        """Clean up old tokens."""
        # Implementation depends on your database layer
        pass