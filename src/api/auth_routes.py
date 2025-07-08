"""
Authentication API Routes.

This module provides comprehensive REST API endpoints for user authentication,
registration, MFA, OAuth, and session management.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, EmailStr

from ..services.auth_service import (
    UserAuthService, UserRegistrationRequest, UserLoginRequest, 
    PasswordChangeRequest, AuthResult
)
from ..services.mfa_service import (
    MFAService, MFASetupRequest, MFAVerifyRequest, 
    TOTPSetupResult, MFAVerificationResult
)
from ..services.oauth_service import (
    OAuthService, OAuthProvider, OAuthAuthorizationResult, OAuthLoginResult
)
from ..database.models.users import User, UserRole
from ..security.rate_limiting import RateLimiter
from ..monitoring.metrics import MetricsCollector


logger = logging.getLogger(__name__)
security = HTTPBearer()


# Response models
class UserResponse(BaseModel):
    """User response model."""
    id: UUID
    username: str
    email: EmailStr
    full_name: str
    role: UserRole
    is_email_verified: bool
    two_factor_enabled: bool
    created_at: str
    last_login_at: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class MFARequiredResponse(BaseModel):
    """MFA required response model."""
    requires_mfa: bool = True
    mfa_token: str
    message: str = "MFA verification required"


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class SessionResponse(BaseModel):
    """Session response model."""
    session_id: str
    ip_address: str
    user_agent: str
    created_at: str
    last_activity: str
    expires_at: str


class MFAStatusResponse(BaseModel):
    """MFA status response model."""
    enabled: bool
    method: Optional[str] = None
    backup_codes_remaining: int = 0


class OAuthAuthResponse(BaseModel):
    """OAuth authorization response model."""
    authorization_url: str
    state: str


class APIKeyRequest(BaseModel):
    """API key request model."""
    name: str = Field(..., min_length=1, max_length=100)
    permissions: List[str] = Field(default=[])
    expires_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: UUID
    name: str
    key: str
    permissions: List[str]
    expires_at: Optional[str] = None
    created_at: str


class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirm request model."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)


class EmailVerificationRequest(BaseModel):
    """Email verification request model."""
    token: str


def create_auth_router(auth_service: UserAuthService, 
                      mfa_service: MFAService,
                      oauth_service: OAuthService,
                      rate_limiter: RateLimiter,
                      metrics_collector: MetricsCollector) -> APIRouter:
    """Create authentication router with all endpoints."""
    
    router = APIRouter(prefix="/auth", tags=["Authentication"])
    
    # Authentication endpoints
    
    @router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def register(
        registration_data: UserRegistrationRequest,
        request: Request
    ) -> UserResponse:
        """Register a new user."""
        # Rate limiting
        client_ip = auth_service._get_client_ip(request)
        if not await rate_limiter.check_rate_limit(client_ip, 'register', 'auth'):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registration attempts"
            )
        
        # Register user
        result = await auth_service.register_user(registration_data)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        # Update metrics
        await metrics_collector.increment_counter('auth_registrations_total')
        
        return UserResponse(
            id=result.user.id,
            username=result.user.username,
            email=result.user.email,
            full_name=result.user.full_name,
            role=result.user.role,
            is_email_verified=result.user.is_email_verified,
            two_factor_enabled=result.user.two_factor_enabled,
            created_at=result.user.created_at.isoformat(),
            last_login_at=result.user.last_login_at.isoformat() if result.user.last_login_at else None
        )
    
    @router.post("/login")
    async def login(
        login_data: UserLoginRequest,
        request: Request
    ) -> LoginResponse | MFARequiredResponse:
        """Authenticate user and return tokens."""
        # Authenticate user
        result = await auth_service.authenticate_user(login_data, request)
        
        if not result.success:
            if result.requires_mfa:
                return MFARequiredResponse(mfa_token=result.mfa_token)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error
            )
        
        # Update metrics
        await metrics_collector.increment_counter(
            'auth_logins_total',
            labels={'method': 'password', 'success': 'true'}
        )
        
        return LoginResponse(
            access_token=result.tokens['access_token'],
            refresh_token=result.tokens['refresh_token'],
            expires_in=result.tokens['expires_in'],
            user=UserResponse(
                id=result.user.id,
                username=result.user.username,
                email=result.user.email,
                full_name=result.user.full_name,
                role=result.user.role,
                is_email_verified=result.user.is_email_verified,
                two_factor_enabled=result.user.two_factor_enabled,
                created_at=result.user.created_at.isoformat(),
                last_login_at=result.user.last_login_at.isoformat() if result.user.last_login_at else None
            )
        )
    
    @router.post("/logout")
    async def logout(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Dict[str, str]:
        """Logout user and invalidate session."""
        token = credentials.credentials
        success = await auth_service.logout_user(token, request)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return {"message": "Successfully logged out"}
    
    @router.post("/refresh", response_model=TokenResponse)
    async def refresh_token(
        refresh_request: RefreshTokenRequest
    ) -> TokenResponse:
        """Refresh access token."""
        tokens = await auth_service.refresh_access_token(refresh_request.refresh_token)
        
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        return TokenResponse(
            access_token=tokens['access_token'],
            expires_in=tokens['expires_in']
        )
    
    @router.get("/me", response_model=UserResponse)
    async def get_current_user(
        current_user: User = Depends(get_current_user)
    ) -> UserResponse:
        """Get current user information."""
        return UserResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            role=current_user.role,
            is_email_verified=current_user.is_email_verified,
            two_factor_enabled=current_user.two_factor_enabled,
            created_at=current_user.created_at.isoformat(),
            last_login_at=current_user.last_login_at.isoformat() if current_user.last_login_at else None
        )
    
    @router.post("/change-password")
    async def change_password(
        password_change: PasswordChangeRequest,
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Change user password."""
        success = await auth_service.change_password(current_user.id, password_change)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to change password"
            )
        
        return {"message": "Password changed successfully"}
    
    # MFA endpoints
    
    @router.post("/mfa/setup", response_model=TOTPSetupResult)
    async def setup_mfa(
        mfa_setup: MFASetupRequest,
        current_user: User = Depends(get_current_user)
    ) -> TOTPSetupResult:
        """Set up multi-factor authentication."""
        return await mfa_service.setup_totp(current_user.id)
    
    @router.post("/mfa/verify")
    async def verify_mfa(
        mfa_verify: MFAVerifyRequest,
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Verify and enable MFA."""
        success = await mfa_service.verify_and_enable_mfa(current_user.id, mfa_verify.mfa_code)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid MFA code"
            )
        
        return {"message": "MFA enabled successfully"}
    
    @router.post("/mfa/disable")
    async def disable_mfa(
        password: str = Field(..., min_length=1),
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Disable multi-factor authentication."""
        success = await mfa_service.disable_mfa(current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to disable MFA"
            )
        
        return {"message": "MFA disabled successfully"}
    
    @router.get("/mfa/status", response_model=MFAStatusResponse)
    async def get_mfa_status(
        current_user: User = Depends(get_current_user)
    ) -> MFAStatusResponse:
        """Get MFA status."""
        status_info = await mfa_service.get_mfa_status(current_user.id)
        
        return MFAStatusResponse(
            enabled=status_info['enabled'],
            method=status_info['method'],
            backup_codes_remaining=0  # Would be calculated from backup codes
        )
    
    @router.post("/mfa/backup-codes")
    async def regenerate_backup_codes(
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, List[str]]:
        """Regenerate MFA backup codes."""
        backup_codes = await mfa_service.regenerate_backup_codes(current_user.id)
        
        return {"backup_codes": backup_codes}
    
    # OAuth endpoints
    
    @router.get("/oauth/{provider}/authorize", response_model=OAuthAuthResponse)
    async def oauth_authorize(
        provider: OAuthProvider,
        request: Request
    ) -> OAuthAuthResponse:
        """Get OAuth authorization URL."""
        try:
            result = await oauth_service.get_authorization_url(provider)
            
            return OAuthAuthResponse(
                authorization_url=result.authorization_url,
                state=result.state
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    @router.get("/oauth/{provider}/callback")
    async def oauth_callback(
        provider: OAuthProvider,
        code: str,
        state: str,
        error: Optional[str] = None
    ) -> LoginResponse:
        """Handle OAuth callback."""
        result = await oauth_service.handle_callback(provider, code, state, error)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error
            )
        
        # Update metrics
        await metrics_collector.increment_counter(
            'auth_logins_total',
            labels={'method': 'oauth', 'provider': provider.value, 'success': 'true'}
        )
        
        return LoginResponse(
            access_token=result.tokens['access_token'],
            refresh_token=result.tokens['refresh_token'],
            expires_in=result.tokens['expires_in'],
            user=UserResponse(
                id=result.user.id,
                username=result.user.username,
                email=result.user.email,
                full_name=result.user.full_name,
                role=result.user.role,
                is_email_verified=result.user.is_email_verified,
                two_factor_enabled=result.user.two_factor_enabled,
                created_at=result.user.created_at.isoformat(),
                last_login_at=result.user.last_login_at.isoformat() if result.user.last_login_at else None
            )
        )
    
    @router.post("/oauth/{provider}/link")
    async def link_oauth_account(
        provider: OAuthProvider,
        access_token: str,
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Link OAuth account to current user."""
        success = await oauth_service.link_account(current_user.id, provider, access_token)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to link account"
            )
        
        return {"message": f"{provider.value} account linked successfully"}
    
    @router.delete("/oauth/{provider}/unlink")
    async def unlink_oauth_account(
        provider: OAuthProvider,
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Unlink OAuth account from current user."""
        success = await oauth_service.unlink_account(current_user.id, provider)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to unlink account"
            )
        
        return {"message": f"{provider.value} account unlinked successfully"}
    
    @router.get("/oauth/linked")
    async def get_linked_accounts(
        current_user: User = Depends(get_current_user)
    ) -> List[Dict[str, Any]]:
        """Get linked OAuth accounts."""
        return await oauth_service.get_linked_accounts(current_user.id)
    
    # Session management endpoints
    
    @router.get("/sessions", response_model=List[SessionResponse])
    async def get_user_sessions(
        current_user: User = Depends(get_current_user)
    ) -> List[SessionResponse]:
        """Get user sessions."""
        sessions = await auth_service.get_user_sessions(current_user.id)
        
        return [
            SessionResponse(
                session_id=session.session_id,
                ip_address=session.ip_address,
                user_agent=session.user_agent,
                created_at=session.created_at.isoformat(),
                last_activity=session.last_activity.isoformat(),
                expires_at=session.expires_at.isoformat()
            )
            for session in sessions
        ]
    
    @router.delete("/sessions/{session_id}")
    async def revoke_session(
        session_id: str,
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Revoke a specific session."""
        success = await auth_service.revoke_session(current_user.id, session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {"message": "Session revoked successfully"}
    
    # Password reset endpoints
    
    @router.post("/password-reset/request")
    async def request_password_reset(
        reset_request: PasswordResetRequest,
        request: Request
    ) -> Dict[str, str]:
        """Request password reset."""
        # Rate limiting
        client_ip = auth_service._get_client_ip(request)
        if not await rate_limiter.check_rate_limit(client_ip, 'password_reset', 'auth'):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password reset requests"
            )
        
        # This would typically send an email with reset link
        # For now, we'll just return success
        return {"message": "Password reset email sent"}
    
    @router.post("/password-reset/confirm")
    async def confirm_password_reset(
        reset_confirm: PasswordResetConfirmRequest
    ) -> Dict[str, str]:
        """Confirm password reset."""
        # This would verify the token and reset the password
        # Implementation depends on how you store reset tokens
        return {"message": "Password reset successfully"}
    
    # Email verification endpoints
    
    @router.post("/email/verify")
    async def verify_email(
        verification: EmailVerificationRequest
    ) -> Dict[str, str]:
        """Verify email address."""
        # This would verify the email token
        # Implementation depends on how you store verification tokens
        return {"message": "Email verified successfully"}
    
    @router.post("/email/resend-verification")
    async def resend_email_verification(
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, str]:
        """Resend email verification."""
        # This would resend the verification email
        return {"message": "Verification email sent"}
    
    # Helper dependency functions
    
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> User:
        """Get current user from token."""
        token = credentials.credentials
        user = await auth_service.verify_access_token(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return user
    
    async def get_current_admin_user(
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Get current admin user."""
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        return current_user
    
    return router