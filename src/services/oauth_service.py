"""
OAuth2 Integration Service.

This module provides OAuth2 integration for external providers like Google, GitHub,
Microsoft, and other popular authentication providers.
"""

import secrets
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4
from urllib.parse import urlencode, parse_qs, urlparse

import httpx
from pydantic import BaseModel, Field, EmailStr

from ..database.models.users import User, UserRole, UserStatus
from ..database.connection import get_database_connection


logger = logging.getLogger(__name__)


class OAuthProvider(str, Enum):
    """Supported OAuth providers."""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"


class OAuthScope(str, Enum):
    """OAuth scopes."""
    EMAIL = "email"
    PROFILE = "profile"
    OPENID = "openid"
    READ_USER = "read:user"
    USER_EMAIL = "user:email"


class OAuthUserInfo(BaseModel):
    """OAuth user information."""
    provider: OAuthProvider
    provider_user_id: str
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    profile_url: Optional[str] = None
    verified: bool = False
    locale: Optional[str] = None


class OAuthTokens(BaseModel):
    """OAuth tokens."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    expires_at: Optional[datetime] = None
    scope: Optional[str] = None


class OAuthAuthorizationResult(BaseModel):
    """OAuth authorization result."""
    authorization_url: str
    state: str
    code_verifier: Optional[str] = None  # For PKCE


class OAuthLoginResult(BaseModel):
    """OAuth login result."""
    success: bool
    user: Optional[User] = None
    tokens: Optional[Dict[str, str]] = None
    user_info: Optional[OAuthUserInfo] = None
    requires_linking: bool = False
    error: Optional[str] = None


class OAuthProviderConfig(BaseModel):
    """OAuth provider configuration."""
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    user_info_url: str
    scopes: List[str]
    redirect_uri: str
    use_pkce: bool = False


class OAuthService:
    """OAuth2 integration service."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = get_database_connection()
        
        # Provider configurations
        self.providers = self._load_provider_configs()
        
        # HTTP client for API requests
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # State storage (in production, use Redis or database)
        self.state_storage = {}
    
    def _load_provider_configs(self) -> Dict[OAuthProvider, OAuthProviderConfig]:
        """Load OAuth provider configurations."""
        providers = {}
        
        # Google OAuth configuration
        if 'google' in self.config.get('oauth_providers', {}):
            google_config = self.config['oauth_providers']['google']
            providers[OAuthProvider.GOOGLE] = OAuthProviderConfig(
                client_id=google_config['client_id'],
                client_secret=google_config['client_secret'],
                authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
                token_url="https://oauth2.googleapis.com/token",
                user_info_url="https://www.googleapis.com/oauth2/v2/userinfo",
                scopes=["openid", "email", "profile"],
                redirect_uri=google_config['redirect_uri']
            )
        
        # GitHub OAuth configuration
        if 'github' in self.config.get('oauth_providers', {}):
            github_config = self.config['oauth_providers']['github']
            providers[OAuthProvider.GITHUB] = OAuthProviderConfig(
                client_id=github_config['client_id'],
                client_secret=github_config['client_secret'],
                authorization_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                user_info_url="https://api.github.com/user",
                scopes=["user:email", "read:user"],
                redirect_uri=github_config['redirect_uri']
            )
        
        # Microsoft OAuth configuration
        if 'microsoft' in self.config.get('oauth_providers', {}):
            microsoft_config = self.config['oauth_providers']['microsoft']
            providers[OAuthProvider.MICROSOFT] = OAuthProviderConfig(
                client_id=microsoft_config['client_id'],
                client_secret=microsoft_config['client_secret'],
                authorization_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
                user_info_url="https://graph.microsoft.com/v1.0/me",
                scopes=["openid", "profile", "email"],
                redirect_uri=microsoft_config['redirect_uri'],
                use_pkce=True
            )
        
        return providers
    
    async def get_authorization_url(self, provider: OAuthProvider, 
                                  state_data: Optional[Dict[str, Any]] = None) -> OAuthAuthorizationResult:
        """Get OAuth authorization URL."""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not configured")
            
            provider_config = self.providers[provider]
            
            # Generate state parameter
            state = secrets.token_urlsafe(32)
            
            # Store state data
            self.state_storage[state] = {
                'provider': provider,
                'created_at': datetime.utcnow(),
                'data': state_data or {}
            }
            
            # Prepare authorization parameters
            auth_params = {
                'client_id': provider_config.client_id,
                'redirect_uri': provider_config.redirect_uri,
                'scope': ' '.join(provider_config.scopes),
                'response_type': 'code',
                'state': state,
                'access_type': 'offline'  # For refresh tokens
            }
            
            # PKCE support
            code_verifier = None
            if provider_config.use_pkce:
                code_verifier = secrets.token_urlsafe(32)
                code_challenge = self._generate_pkce_challenge(code_verifier)
                auth_params.update({
                    'code_challenge': code_challenge,
                    'code_challenge_method': 'S256'
                })
                
                # Store code verifier
                self.state_storage[state]['code_verifier'] = code_verifier
            
            # Provider-specific parameters
            if provider == OAuthProvider.GOOGLE:
                auth_params['prompt'] = 'consent'
            elif provider == OAuthProvider.GITHUB:
                auth_params['allow_signup'] = 'true'
            elif provider == OAuthProvider.MICROSOFT:
                auth_params['prompt'] = 'select_account'
            
            # Build authorization URL
            authorization_url = f"{provider_config.authorization_url}?{urlencode(auth_params)}"
            
            return OAuthAuthorizationResult(
                authorization_url=authorization_url,
                state=state,
                code_verifier=code_verifier
            )
            
        except Exception as e:
            logger.error(f"Failed to get authorization URL for {provider}: {str(e)}")
            raise
    
    async def handle_callback(self, provider: OAuthProvider, code: str, 
                            state: str, error: Optional[str] = None) -> OAuthLoginResult:
        """Handle OAuth callback."""
        try:
            # Check for error in callback
            if error:
                return OAuthLoginResult(
                    success=False,
                    error=f"OAuth error: {error}"
                )
            
            # Validate state
            if state not in self.state_storage:
                return OAuthLoginResult(
                    success=False,
                    error="Invalid state parameter"
                )
            
            state_data = self.state_storage[state]
            if state_data['provider'] != provider:
                return OAuthLoginResult(
                    success=False,
                    error="Provider mismatch"
                )
            
            # Check state expiration (5 minutes)
            if datetime.utcnow() - state_data['created_at'] > timedelta(minutes=5):
                return OAuthLoginResult(
                    success=False,
                    error="State expired"
                )
            
            # Exchange code for tokens
            tokens = await self._exchange_code_for_tokens(provider, code, state_data)
            if not tokens:
                return OAuthLoginResult(
                    success=False,
                    error="Failed to exchange code for tokens"
                )
            
            # Get user info from provider
            user_info = await self._get_user_info(provider, tokens)
            if not user_info:
                return OAuthLoginResult(
                    success=False,
                    error="Failed to get user information"
                )
            
            # Find or create user
            user = await self._find_or_create_user(user_info, tokens)
            
            # Generate authentication tokens
            auth_tokens = await self._generate_auth_tokens(user)
            
            # Clean up state
            del self.state_storage[state]
            
            return OAuthLoginResult(
                success=True,
                user=user,
                tokens=auth_tokens,
                user_info=user_info
            )
            
        except Exception as e:
            logger.error(f"OAuth callback failed for {provider}: {str(e)}")
            return OAuthLoginResult(
                success=False,
                error="OAuth authentication failed"
            )
    
    async def link_account(self, user_id: UUID, provider: OAuthProvider, 
                          access_token: str) -> bool:
        """Link OAuth account to existing user."""
        try:
            # Get user info from provider
            tokens = OAuthTokens(access_token=access_token)
            user_info = await self._get_user_info(provider, tokens)
            
            if not user_info:
                return False
            
            # Check if OAuth account is already linked
            existing_link = await self._get_oauth_link(provider, user_info.provider_user_id)
            if existing_link:
                return False
            
            # Link account
            await self._store_oauth_link(user_id, user_info, tokens)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to link {provider} account for user {user_id}: {str(e)}")
            return False
    
    async def unlink_account(self, user_id: UUID, provider: OAuthProvider) -> bool:
        """Unlink OAuth account from user."""
        try:
            # Remove OAuth link
            await self._remove_oauth_link(user_id, provider)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to unlink {provider} account for user {user_id}: {str(e)}")
            return False
    
    async def get_linked_accounts(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get linked OAuth accounts for user."""
        try:
            return await self._get_user_oauth_links(user_id)
            
        except Exception as e:
            logger.error(f"Failed to get linked accounts for user {user_id}: {str(e)}")
            return []
    
    async def refresh_oauth_token(self, user_id: UUID, provider: OAuthProvider) -> Optional[OAuthTokens]:
        """Refresh OAuth token."""
        try:
            # Get OAuth link
            oauth_link = await self._get_user_oauth_link(user_id, provider)
            if not oauth_link or not oauth_link.get('refresh_token'):
                return None
            
            # Refresh token
            new_tokens = await self._refresh_provider_token(provider, oauth_link['refresh_token'])
            if not new_tokens:
                return None
            
            # Update stored tokens
            await self._update_oauth_tokens(user_id, provider, new_tokens)
            
            return new_tokens
            
        except Exception as e:
            logger.error(f"Failed to refresh OAuth token for user {user_id}, provider {provider}: {str(e)}")
            return None
    
    # Helper methods
    
    def _generate_pkce_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge."""
        import hashlib
        import base64
        
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    async def _exchange_code_for_tokens(self, provider: OAuthProvider, 
                                       code: str, state_data: Dict[str, Any]) -> Optional[OAuthTokens]:
        """Exchange authorization code for tokens."""
        try:
            provider_config = self.providers[provider]
            
            # Prepare token request
            token_data = {
                'client_id': provider_config.client_id,
                'client_secret': provider_config.client_secret,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': provider_config.redirect_uri
            }
            
            # Add PKCE verifier if used
            if provider_config.use_pkce and 'code_verifier' in state_data:
                token_data['code_verifier'] = state_data['code_verifier']
            
            # Make token request
            headers = {'Accept': 'application/json'}
            response = await self.http_client.post(
                provider_config.token_url,
                data=token_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.status_code} {response.text}")
                return None
            
            token_response = response.json()
            
            # Parse tokens
            expires_at = None
            if 'expires_in' in token_response:
                expires_at = datetime.utcnow() + timedelta(seconds=token_response['expires_in'])
            
            return OAuthTokens(
                access_token=token_response['access_token'],
                refresh_token=token_response.get('refresh_token'),
                token_type=token_response.get('token_type', 'Bearer'),
                expires_in=token_response.get('expires_in'),
                expires_at=expires_at,
                scope=token_response.get('scope')
            )
            
        except Exception as e:
            logger.error(f"Token exchange failed for {provider}: {str(e)}")
            return None
    
    async def _get_user_info(self, provider: OAuthProvider, tokens: OAuthTokens) -> Optional[OAuthUserInfo]:
        """Get user information from provider."""
        try:
            provider_config = self.providers[provider]
            
            # Make user info request
            headers = {
                'Authorization': f'{tokens.token_type} {tokens.access_token}',
                'Accept': 'application/json'
            }
            
            response = await self.http_client.get(
                provider_config.user_info_url,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"User info request failed: {response.status_code} {response.text}")
                return None
            
            user_data = response.json()
            
            # Parse user info based on provider
            if provider == OAuthProvider.GOOGLE:
                return OAuthUserInfo(
                    provider=provider,
                    provider_user_id=user_data['id'],
                    email=user_data.get('email'),
                    name=user_data.get('name'),
                    username=user_data.get('email', '').split('@')[0],
                    avatar_url=user_data.get('picture'),
                    verified=user_data.get('verified_email', False),
                    locale=user_data.get('locale')
                )
            
            elif provider == OAuthProvider.GITHUB:
                return OAuthUserInfo(
                    provider=provider,
                    provider_user_id=str(user_data['id']),
                    email=user_data.get('email'),
                    name=user_data.get('name'),
                    username=user_data.get('login'),
                    avatar_url=user_data.get('avatar_url'),
                    profile_url=user_data.get('html_url'),
                    verified=True  # GitHub emails are verified
                )
            
            elif provider == OAuthProvider.MICROSOFT:
                return OAuthUserInfo(
                    provider=provider,
                    provider_user_id=user_data['id'],
                    email=user_data.get('mail') or user_data.get('userPrincipalName'),
                    name=user_data.get('displayName'),
                    username=user_data.get('userPrincipalName', '').split('@')[0],
                    verified=True
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user info from {provider}: {str(e)}")
            return None
    
    async def _find_or_create_user(self, user_info: OAuthUserInfo, 
                                  tokens: OAuthTokens) -> User:
        """Find existing user or create new one."""
        try:
            # Check if OAuth account is already linked
            existing_link = await self._get_oauth_link(user_info.provider, user_info.provider_user_id)
            if existing_link:
                user = await self._get_user_by_id(existing_link['user_id'])
                if user:
                    # Update OAuth tokens
                    await self._update_oauth_tokens(user.id, user_info.provider, tokens)
                    return user
            
            # Check if user exists by email
            if user_info.email:
                user = await self._get_user_by_email(user_info.email)
                if user:
                    # Link OAuth account to existing user
                    await self._store_oauth_link(user.id, user_info, tokens)
                    return user
            
            # Create new user
            user_id = uuid4()
            user_data = {
                'id': user_id,
                'username': user_info.username or f"{user_info.provider}_{user_info.provider_user_id}",
                'email': user_info.email or f"{user_info.provider_user_id}@{user_info.provider}.local",
                'full_name': user_info.name or user_info.username or '',
                'password_hash': '',  # OAuth users don't have passwords
                'password_salt': '',
                'role': UserRole.USER,
                'status': UserStatus.ACTIVE,  # OAuth users are pre-verified
                'is_email_verified': user_info.verified,
                'created_at': datetime.utcnow()
            }
            
            # Create user
            user = await self._create_user(user_data)
            
            # Link OAuth account
            await self._store_oauth_link(user.id, user_info, tokens)
            
            return user
            
        except Exception as e:
            logger.error(f"Failed to find or create user: {str(e)}")
            raise
    
    async def _refresh_provider_token(self, provider: OAuthProvider, 
                                    refresh_token: str) -> Optional[OAuthTokens]:
        """Refresh token with provider."""
        try:
            provider_config = self.providers[provider]
            
            # Prepare refresh request
            refresh_data = {
                'client_id': provider_config.client_id,
                'client_secret': provider_config.client_secret,
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'
            }
            
            # Make refresh request
            headers = {'Accept': 'application/json'}
            response = await self.http_client.post(
                provider_config.token_url,
                data=refresh_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.status_code} {response.text}")
                return None
            
            token_response = response.json()
            
            # Parse tokens
            expires_at = None
            if 'expires_in' in token_response:
                expires_at = datetime.utcnow() + timedelta(seconds=token_response['expires_in'])
            
            return OAuthTokens(
                access_token=token_response['access_token'],
                refresh_token=token_response.get('refresh_token', refresh_token),
                token_type=token_response.get('token_type', 'Bearer'),
                expires_in=token_response.get('expires_in'),
                expires_at=expires_at,
                scope=token_response.get('scope')
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed for {provider}: {str(e)}")
            return None
    
    # Database operations (to be implemented with actual database calls)
    
    async def _get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        # Implementation depends on your database layer
        pass
    
    async def _create_user(self, user_data: Dict[str, Any]) -> User:
        """Create user."""
        # Implementation depends on your database layer
        pass
    
    async def _get_oauth_link(self, provider: OAuthProvider, provider_user_id: str) -> Optional[Dict[str, Any]]:
        """Get OAuth link by provider and user ID."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_oauth_link(self, user_id: UUID, provider: OAuthProvider) -> Optional[Dict[str, Any]]:
        """Get OAuth link for user and provider."""
        # Implementation depends on your database layer
        pass
    
    async def _get_user_oauth_links(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all OAuth links for user."""
        # Implementation depends on your database layer
        pass
    
    async def _store_oauth_link(self, user_id: UUID, user_info: OAuthUserInfo, 
                               tokens: OAuthTokens) -> None:
        """Store OAuth link."""
        # Implementation depends on your database layer
        pass
    
    async def _update_oauth_tokens(self, user_id: UUID, provider: OAuthProvider, 
                                  tokens: OAuthTokens) -> None:
        """Update OAuth tokens."""
        # Implementation depends on your database layer
        pass
    
    async def _remove_oauth_link(self, user_id: UUID, provider: OAuthProvider) -> None:
        """Remove OAuth link."""
        # Implementation depends on your database layer
        pass
    
    async def _generate_auth_tokens(self, user: User) -> Dict[str, str]:
        """Generate authentication tokens."""
        # Implementation depends on your auth service
        pass
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()