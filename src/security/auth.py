"""
Authentication and authorization system for the Autonomous Agent System.

This module provides comprehensive authentication and authorization
functionality including JWT tokens, API keys, and role-based access control.
"""

import jwt
import hashlib
import hmac
import secrets
import time
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from fastapi import HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from passlib.hash import bcrypt
import redis.asyncio as redis
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"
    READONLY = "readonly"


class Permission(Enum):
    """Permissions for fine-grained access control."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class User:
    """User data model."""
    id: str
    username: str
    email: str
    roles: List[Role] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """API key data model."""
    id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[Permission] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class JWTManager:
    """JWT token management."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 30
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token for user."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create refresh token for user."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user data (this would typically come from database)
        user_id = payload.get("sub")
        # user = await self.get_user_by_id(user_id)  # Implement this
        
        # For now, create a basic user object
        user = User(id=user_id, username="", email="")
        
        return self.create_access_token(user)


class APIKeyManager:
    """API key management."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.api_keys = {}  # In-memory storage for development
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key."""
        return self.pwd_context.hash(api_key)
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify an API key against its hash."""
        return self.pwd_context.verify(api_key, hashed_key)
    
    async def create_api_key(self, 
                           name: str, 
                           user_id: str, 
                           permissions: List[Permission],
                           expires_at: Optional[datetime] = None) -> Tuple[str, APIKey]:
        """Create a new API key."""
        api_key = self.generate_api_key()
        key_hash = self.hash_api_key(api_key)
        
        api_key_obj = APIKey(
            id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at
        )
        
        # Store in Redis or memory
        if self.redis_client:
            await self._store_api_key_redis(api_key_obj)
        else:
            self.api_keys[api_key_obj.id] = api_key_obj
        
        return api_key, api_key_obj
    
    async def verify_api_key_access(self, api_key: str) -> Optional[APIKey]:
        """Verify API key and return associated data."""
        # Get all API keys and check each one
        if self.redis_client:
            api_keys = await self._get_all_api_keys_redis()
        else:
            api_keys = list(self.api_keys.values())
        
        for key_obj in api_keys:
            if not key_obj.is_active:
                continue
            
            if key_obj.expires_at and key_obj.expires_at < datetime.utcnow():
                continue
            
            if self.verify_api_key(api_key, key_obj.key_hash):
                # Update last used time
                key_obj.last_used = datetime.utcnow()
                await self._update_api_key_last_used(key_obj)
                return key_obj
        
        return None
    
    async def _store_api_key_redis(self, api_key: APIKey) -> None:
        """Store API key in Redis."""
        key = f"api_key:{api_key.id}"
        data = {
            "id": api_key.id,
            "key_hash": api_key.key_hash,
            "name": api_key.name,
            "user_id": api_key.user_id,
            "permissions": [p.value for p in api_key.permissions],
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "created_at": api_key.created_at.isoformat(),
            "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            "is_active": api_key.is_active,
            "metadata": api_key.metadata
        }
        
        await self.redis_client.hset(key, mapping=data)
        if api_key.expires_at:
            await self.redis_client.expireat(key, api_key.expires_at)
    
    async def _get_all_api_keys_redis(self) -> List[APIKey]:
        """Get all API keys from Redis."""
        keys = await self.redis_client.keys("api_key:*")
        api_keys = []
        
        for key in keys:
            data = await self.redis_client.hgetall(key)
            if data:
                api_key = APIKey(
                    id=data["id"],
                    key_hash=data["key_hash"],
                    name=data["name"],
                    user_id=data["user_id"],
                    permissions=[Permission(p) for p in data["permissions"].split(",") if p],
                    expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
                    created_at=datetime.fromisoformat(data["created_at"]),
                    last_used=datetime.fromisoformat(data["last_used"]) if data["last_used"] else None,
                    is_active=data["is_active"] == "True",
                    metadata=data.get("metadata", {})
                )
                api_keys.append(api_key)
        
        return api_keys
    
    async def _update_api_key_last_used(self, api_key: APIKey) -> None:
        """Update API key last used time."""
        if self.redis_client:
            key = f"api_key:{api_key.id}"
            await self.redis_client.hset(key, "last_used", api_key.last_used.isoformat())
        else:
            # Already updated in memory


class AuthManager:
    """Main authentication and authorization manager."""
    
    def __init__(self, 
                 jwt_secret: str,
                 redis_client: Optional[redis.Redis] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.jwt_manager = JWTManager(jwt_secret)
        self.api_key_manager = APIKeyManager(redis_client)
        self.redis_client = redis_client
        
        # Security settings
        self.max_login_attempts = self.config.get('max_login_attempts', 5)
        self.lockout_duration = self.config.get('lockout_duration', 300)  # 5 minutes
        
        # In-memory user storage for development
        self.users = {}
        self.login_attempts = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@example.com",
            roles=[Role.ADMIN],
            permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN]
        )
        self.users["admin"] = admin_user
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        # Check for account lockout
        if await self._is_account_locked(username):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to too many failed attempts"
            )
        
        # Get user (this would typically come from database)
        user = self.users.get(username)
        if not user:
            await self._record_failed_attempt(username)
            return None
        
        # Verify password (for demo purposes, any password works for admin)
        if username == "admin" or self._verify_password(password, user):
            await self._clear_failed_attempts(username)
            user.last_login = datetime.utcnow()
            return user
        else:
            await self._record_failed_attempt(username)
            return None
    
    async def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate user with JWT token."""
        try:
            payload = self.jwt_manager.verify_token(token)
            user_id = payload.get("sub")
            
            # Get user (this would typically come from database)
            user = next((u for u in self.users.values() if u.id == user_id), None)
            if user and user.is_active:
                return user
            
        except HTTPException:
            pass
        
        return None
    
    async def authenticate_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
        """Authenticate with API key."""
        key_obj = await self.api_key_manager.verify_api_key_access(api_key)
        if not key_obj:
            return None
        
        # Get user
        user = next((u for u in self.users.values() if u.id == key_obj.user_id), None)
        if user and user.is_active:
            return user, key_obj
        
        return None
    
    def _verify_password(self, password: str, user: User) -> bool:
        """Verify password (placeholder implementation)."""
        # In a real implementation, this would check against hashed passwords
        return True
    
    async def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked."""
        if username not in self.login_attempts:
            return False
        
        attempts = self.login_attempts[username]
        if attempts['count'] >= self.max_login_attempts:
            if time.time() - attempts['last_attempt'] < self.lockout_duration:
                return True
            else:
                # Lockout expired, reset attempts
                del self.login_attempts[username]
        
        return False
    
    async def _record_failed_attempt(self, username: str) -> None:
        """Record failed login attempt."""
        if username not in self.login_attempts:
            self.login_attempts[username] = {'count': 0, 'last_attempt': 0}
        
        self.login_attempts[username]['count'] += 1
        self.login_attempts[username]['last_attempt'] = time.time()
    
    async def _clear_failed_attempts(self, username: str) -> None:
        """Clear failed login attempts."""
        if username in self.login_attempts:
            del self.login_attempts[username]
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions or Permission.ADMIN in user.permissions
    
    def has_role(self, user: User, role: Role) -> bool:
        """Check if user has specific role."""
        return role in user.roles or Role.ADMIN in user.roles
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get('current_user')
                if not user or not self.has_permission(user, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, role: Role):
        """Decorator to require specific role."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get('current_user')
                if not user or not self.has_role(user, role):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient role"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# FastAPI security schemes
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends()
) -> User:
    """Get current user from JWT token."""
    token = credentials.credentials
    user = await auth_manager.authenticate_token(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: AuthManager = Depends()
) -> Optional[User]:
    """Get current user from JWT token (optional)."""
    if not credentials:
        return None
    
    token = credentials.credentials
    return await auth_manager.authenticate_token(token)


async def get_api_key_user(
    request: Request,
    auth_manager: AuthManager = Depends()
) -> Optional[Tuple[User, APIKey]]:
    """Get user from API key."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None
    
    return await auth_manager.authenticate_api_key(api_key)


def require_auth(allow_api_key: bool = True):
    """Decorator to require authentication."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            auth_manager = kwargs.get('auth_manager')
            
            user = None
            
            # Try JWT authentication first
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                user = await auth_manager.authenticate_token(token)
            
            # Try API key authentication if JWT failed and allowed
            if not user and allow_api_key:
                api_key = request.headers.get("X-API-Key")
                if api_key:
                    result = await auth_manager.authenticate_api_key(api_key)
                    if result:
                        user, _ = result
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            kwargs['current_user'] = user
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Pydantic models for API requests/responses
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=100)


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default=[])
    expires_days: Optional[int] = Field(None, ge=1, le=365)


class CreateAPIKeyResponse(BaseModel):
    api_key: str
    id: str
    name: str
    permissions: List[str]
    expires_at: Optional[datetime]


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]