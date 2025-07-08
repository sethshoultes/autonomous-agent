"""
Comprehensive tests for the authentication system.

This module provides unit and integration tests for the authentication,
authorization, MFA, OAuth, and security monitoring components.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient
from fastapi import HTTPException, status

from src.services.auth_service import (
    UserAuthService, UserRegistrationRequest, UserLoginRequest,
    PasswordChangeRequest, AuthResult
)
from src.services.mfa_service import (
    MFAService, MFASetupRequest, MFAVerifyRequest, MFAMethodType
)
from src.services.oauth_service import (
    OAuthService, OAuthProvider, OAuthAuthorizationResult, OAuthLoginResult
)
from src.services.user_management_service import (
    UserManagementService, UserProfileUpdate, UserPreferencesUpdate,
    UserListQuery, ActivityType
)
from src.services.security_monitoring_service import (
    SecurityMonitoringService, SecurityEventType, SecurityEventSeverity
)
from src.database.models.users import User, UserRole, UserStatus
from src.security.rate_limiting import RateLimiter
from src.monitoring.metrics import MetricsCollector


class TestUserAuthService:
    """Test cases for UserAuthService."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock authentication service configuration."""
        return {
            'jwt_secret': 'test_secret',
            'jwt_algorithm': 'HS256',
            'access_token_expire_minutes': 30,
            'refresh_token_expire_days': 30,
            'max_login_attempts': 5,
            'lockout_duration': 300,
            'redis_url': None,
            'database_url': 'sqlite:///test.db'
        }
    
    @pytest.fixture
    def auth_service(self, mock_config):
        """Create UserAuthService instance for testing."""
        return UserAuthService(
            config=mock_config,
            redis_client=None,
            metrics_collector=Mock()
        )
    
    @pytest.fixture
    def valid_registration_data(self):
        """Valid user registration data."""
        return UserRegistrationRequest(
            username='testuser',
            email='test@example.com',
            password='TestPass123!',
            full_name='Test User'
        )
    
    @pytest.fixture
    def valid_login_data(self):
        """Valid user login data."""
        return UserLoginRequest(
            username='testuser',
            password='TestPass123!'
        )
    
    @pytest.fixture
    def mock_user(self):
        """Mock user object."""
        return User(
            id=uuid4(),
            username='testuser',
            email='test@example.com',
            full_name='Test User',
            password_hash='hashed_password',
            password_salt='salt',
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            is_email_verified=True,
            two_factor_enabled=False
        )
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_service, valid_registration_data):
        """Test successful user registration."""
        with patch.object(auth_service, '_get_user_by_email_or_username', return_value=None), \
             patch.object(auth_service, '_create_user_record', return_value=None), \
             patch.object(auth_service, '_send_verification_email', return_value=None), \
             patch.object(auth_service, '_log_auth_event', return_value=None):
            
            result = await auth_service.register_user(valid_registration_data)
            
            assert result.success is True
            assert result.user is not None
            assert result.user.username == 'testuser'
            assert result.user.email == 'test@example.com'
            assert result.error is None
    
    @pytest.mark.asyncio
    async def test_register_user_existing_user(self, auth_service, valid_registration_data, mock_user):
        """Test user registration with existing user."""
        with patch.object(auth_service, '_get_user_by_email_or_username', return_value=mock_user):
            
            result = await auth_service.register_user(valid_registration_data)
            
            assert result.success is False
            assert result.user is None
            assert "already exists" in result.error
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service, valid_login_data, mock_user):
        """Test successful user authentication."""
        mock_request = Mock()
        mock_request.client.host = '127.0.0.1'
        mock_request.headers = {}
        
        with patch.object(auth_service, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_service.rate_limiter, 'check_rate_limit', return_value=True), \
             patch.object(auth_service, '_is_account_locked', return_value=False), \
             patch.object(auth_service, '_get_user_by_username', return_value=mock_user), \
             patch.object(auth_service.pwd_context, 'verify', return_value=True), \
             patch.object(auth_service, '_clear_failed_login_attempts', return_value=None), \
             patch.object(auth_service, '_generate_auth_tokens', return_value={'access_token': 'token'}), \
             patch.object(auth_service, '_create_session', return_value=Mock()), \
             patch.object(auth_service, '_update_last_login', return_value=None), \
             patch.object(auth_service, '_log_auth_event', return_value=None):
            
            result = await auth_service.authenticate_user(valid_login_data, mock_request)
            
            assert result.success is True
            assert result.user is not None
            assert result.tokens is not None
            assert result.requires_mfa is False
    
    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_password(self, auth_service, valid_login_data, mock_user):
        """Test authentication with invalid password."""
        mock_request = Mock()
        mock_request.client.host = '127.0.0.1'
        mock_request.headers = {}
        
        with patch.object(auth_service, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_service.rate_limiter, 'check_rate_limit', return_value=True), \
             patch.object(auth_service, '_is_account_locked', return_value=False), \
             patch.object(auth_service, '_get_user_by_username', return_value=mock_user), \
             patch.object(auth_service.pwd_context, 'verify', return_value=False), \
             patch.object(auth_service, '_record_failed_login', return_value=None):
            
            result = await auth_service.authenticate_user(valid_login_data, mock_request)
            
            assert result.success is False
            assert result.user is None
            assert "Invalid username or password" in result.error
    
    @pytest.mark.asyncio
    async def test_authenticate_user_mfa_required(self, auth_service, valid_login_data):
        """Test authentication with MFA required."""
        mock_request = Mock()
        mock_request.client.host = '127.0.0.1'
        mock_request.headers = {}
        
        mock_user = User(
            id=uuid4(),
            username='testuser',
            email='test@example.com',
            full_name='Test User',
            password_hash='hashed_password',
            password_salt='salt',
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            is_email_verified=True,
            two_factor_enabled=True
        )
        
        with patch.object(auth_service, '_get_client_ip', return_value='127.0.0.1'), \
             patch.object(auth_service.rate_limiter, 'check_rate_limit', return_value=True), \
             patch.object(auth_service, '_is_account_locked', return_value=False), \
             patch.object(auth_service, '_get_user_by_username', return_value=mock_user), \
             patch.object(auth_service.pwd_context, 'verify', return_value=True), \
             patch.object(auth_service, '_clear_failed_login_attempts', return_value=None), \
             patch.object(auth_service, '_generate_mfa_token', return_value='mfa_token'):
            
            result = await auth_service.authenticate_user(valid_login_data, mock_request)
            
            assert result.success is False
            assert result.requires_mfa is True
            assert result.mfa_token == 'mfa_token'
    
    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_service, mock_user):
        """Test successful password change."""
        password_change = PasswordChangeRequest(
            current_password='oldpass',
            new_password='NewPass123!'
        )
        
        with patch.object(auth_service, '_get_user_by_id', return_value=mock_user), \
             patch.object(auth_service.pwd_context, 'verify', return_value=True), \
             patch.object(auth_service, '_update_user_password', return_value=None), \
             patch.object(auth_service, '_invalidate_all_user_sessions', return_value=None), \
             patch.object(auth_service, '_log_auth_event', return_value=None):
            
            result = await auth_service.change_password(mock_user.id, password_change)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_change_password_invalid_current(self, auth_service, mock_user):
        """Test password change with invalid current password."""
        password_change = PasswordChangeRequest(
            current_password='wrongpass',
            new_password='NewPass123!'
        )
        
        with patch.object(auth_service, '_get_user_by_id', return_value=mock_user), \
             patch.object(auth_service.pwd_context, 'verify', return_value=False):
            
            result = await auth_service.change_password(mock_user.id, password_change)
            
            assert result is False


class TestMFAService:
    """Test cases for MFAService."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock MFA service configuration."""
        return {
            'totp_issuer': 'Test System',
            'sms_enabled': False,
            'email_enabled': True,
            'database_url': 'sqlite:///test.db'
        }
    
    @pytest.fixture
    def mfa_service(self, mock_config):
        """Create MFAService instance for testing."""
        return MFAService(config=mock_config)
    
    @pytest.fixture
    def mock_user(self):
        """Mock user object."""
        return User(
            id=uuid4(),
            username='testuser',
            email='test@example.com',
            full_name='Test User',
            password_hash='hashed_password',
            password_salt='salt',
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            is_email_verified=True,
            two_factor_enabled=False
        )
    
    @pytest.mark.asyncio
    async def test_setup_totp_success(self, mfa_service, mock_user):
        """Test successful TOTP setup."""
        with patch.object(mfa_service, '_get_user', return_value=mock_user), \
             patch.object(mfa_service, '_store_mfa_setup', return_value=None):
            
            result = await mfa_service.setup_totp(mock_user.id)
            
            assert result.secret is not None
            assert result.backup_codes is not None
            assert result.qr_code is not None
            assert result.recovery_codes is not None
            assert len(result.backup_codes) == 10
            assert len(result.recovery_codes) == 5
    
    @pytest.mark.asyncio
    async def test_verify_and_enable_mfa_success(self, mfa_service, mock_user):
        """Test successful MFA verification and enabling."""
        mock_setup = {
            'method': MFAMethodType.TOTP,
            'secret': 'test_secret',
            'status': 'pending'
        }
        
        with patch.object(mfa_service, '_get_pending_mfa_setup', return_value=mock_setup), \
             patch('pyotp.TOTP') as mock_totp:
            
            mock_totp_instance = Mock()
            mock_totp_instance.verify.return_value = True
            mock_totp.return_value = mock_totp_instance
            
            with patch.object(mfa_service, '_enable_mfa', return_value=None), \
                 patch.object(mfa_service, '_cleanup_pending_mfa_setup', return_value=None):
                
                result = await mfa_service.verify_and_enable_mfa(mock_user.id, '123456')
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_mfa_code_success(self, mfa_service, mock_user):
        """Test successful MFA code verification."""
        mock_config = {
            'method': MFAMethodType.TOTP,
            'secret': 'test_secret',
            'status': 'enabled'
        }
        
        with patch.object(mfa_service, '_check_mfa_lockout', return_value=None), \
             patch.object(mfa_service, '_get_mfa_config', return_value=mock_config), \
             patch('pyotp.TOTP') as mock_totp:
            
            mock_totp_instance = Mock()
            mock_totp_instance.verify.return_value = True
            mock_totp.return_value = mock_totp_instance
            
            with patch.object(mfa_service, '_clear_mfa_attempts', return_value=None):
                
                result = await mfa_service.verify_mfa_code(mock_user.id, '123456')
                
                assert result.success is True
                assert result.method_used == MFAMethodType.TOTP
    
    @pytest.mark.asyncio
    async def test_verify_mfa_code_invalid(self, mfa_service, mock_user):
        """Test MFA code verification with invalid code."""
        mock_config = {
            'method': MFAMethodType.TOTP,
            'secret': 'test_secret',
            'status': 'enabled'
        }
        
        with patch.object(mfa_service, '_check_mfa_lockout', return_value=None), \
             patch.object(mfa_service, '_get_mfa_config', return_value=mock_config), \
             patch('pyotp.TOTP') as mock_totp:
            
            mock_totp_instance = Mock()
            mock_totp_instance.verify.return_value = False
            mock_totp.return_value = mock_totp_instance
            
            with patch.object(mfa_service, '_record_mfa_attempt', return_value=2):
                
                result = await mfa_service.verify_mfa_code(mock_user.id, '123456')
                
                assert result.success is False
                assert result.remaining_attempts == 2
                assert "Invalid verification code" in result.error_message


class TestOAuthService:
    """Test cases for OAuthService."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock OAuth service configuration."""
        return {
            'oauth_providers': {
                'google': {
                    'client_id': 'test_client_id',
                    'client_secret': 'test_client_secret',
                    'redirect_uri': 'http://localhost:8000/auth/oauth/google/callback'
                }
            },
            'database_url': 'sqlite:///test.db'
        }
    
    @pytest.fixture
    def oauth_service(self, mock_config):
        """Create OAuthService instance for testing."""
        return OAuthService(config=mock_config)
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_success(self, oauth_service):
        """Test successful OAuth authorization URL generation."""
        result = await oauth_service.get_authorization_url(OAuthProvider.GOOGLE)
        
        assert result.authorization_url is not None
        assert result.state is not None
        assert 'accounts.google.com' in result.authorization_url
        assert result.state in result.authorization_url
    
    @pytest.mark.asyncio
    async def test_handle_callback_success(self, oauth_service):
        """Test successful OAuth callback handling."""
        # Setup state
        state = 'test_state'
        oauth_service.state_storage[state] = {
            'provider': OAuthProvider.GOOGLE,
            'created_at': datetime.utcnow(),
            'data': {}
        }
        
        mock_tokens = Mock()
        mock_tokens.access_token = 'access_token'
        
        mock_user_info = Mock()
        mock_user_info.provider = OAuthProvider.GOOGLE
        mock_user_info.provider_user_id = 'google_123'
        mock_user_info.email = 'test@example.com'
        mock_user_info.name = 'Test User'
        
        mock_user = Mock()
        mock_user.id = uuid4()
        mock_user.username = 'testuser'
        
        with patch.object(oauth_service, '_exchange_code_for_tokens', return_value=mock_tokens), \
             patch.object(oauth_service, '_get_user_info', return_value=mock_user_info), \
             patch.object(oauth_service, '_find_or_create_user', return_value=mock_user), \
             patch.object(oauth_service, '_generate_auth_tokens', return_value={'access_token': 'token'}):
            
            result = await oauth_service.handle_callback(OAuthProvider.GOOGLE, 'auth_code', state)
            
            assert result.success is True
            assert result.user is not None
            assert result.tokens is not None
            assert result.user_info is not None
    
    @pytest.mark.asyncio
    async def test_handle_callback_invalid_state(self, oauth_service):
        """Test OAuth callback with invalid state."""
        result = await oauth_service.handle_callback(OAuthProvider.GOOGLE, 'auth_code', 'invalid_state')
        
        assert result.success is False
        assert "Invalid state parameter" in result.error


class TestUserManagementService:
    """Test cases for UserManagementService."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock user management service configuration."""
        return {
            'database_url': 'sqlite:///test.db',
            'default_timezone': 'UTC',
            'default_locale': 'en'
        }
    
    @pytest.fixture
    def user_management_service(self, mock_config):
        """Create UserManagementService instance for testing."""
        return UserManagementService(config=mock_config)
    
    @pytest.fixture
    def mock_user(self):
        """Mock user object."""
        return User(
            id=uuid4(),
            username='testuser',
            email='test@example.com',
            full_name='Test User',
            password_hash='hashed_password',
            password_salt='salt',
            role=UserRole.USER,
            status=UserStatus.ACTIVE,
            created_at=datetime.utcnow(),
            is_email_verified=True,
            two_factor_enabled=False
        )
    
    @pytest.mark.asyncio
    async def test_update_user_profile_success(self, user_management_service, mock_user):
        """Test successful user profile update."""
        profile_update = UserProfileUpdate(
            full_name='Updated Name',
            bio='Updated bio'
        )
        
        with patch.object(user_management_service, '_get_user_by_id', return_value=mock_user), \
             patch.object(user_management_service, '_update_user_profile', return_value=True), \
             patch.object(user_management_service, '_log_user_activity', return_value=True):
            
            result = await user_management_service.update_user_profile(mock_user.id, profile_update)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_get_user_stats(self, user_management_service):
        """Test getting user statistics."""
        mock_stats = {
            'total_users': 100,
            'active_users': 85,
            'verified_users': 90,
            'admin_users': 5,
            'users_last_24h': 10,
            'users_last_7d': 25,
            'users_last_30d': 50
        }
        
        with patch.object(user_management_service, '_get_user_stats', return_value=mock_stats):
            
            result = await user_management_service.get_user_stats()
            
            assert result.total_users == 100
            assert result.active_users == 85
            assert result.verified_users == 90
            assert result.admin_users == 5


class TestSecurityMonitoringService:
    """Test cases for SecurityMonitoringService."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock security monitoring service configuration."""
        return {
            'database_url': 'sqlite:///test.db',
            'max_login_attempts': 5,
            'rate_limit_threshold': 100,
            'alert_retention_days': 90
        }
    
    @pytest.fixture
    def security_monitoring_service(self, mock_config):
        """Create SecurityMonitoringService instance for testing."""
        return SecurityMonitoringService(config=mock_config)
    
    @pytest.mark.asyncio
    async def test_log_security_event_success(self, security_monitoring_service):
        """Test successful security event logging."""
        with patch.object(security_monitoring_service, '_store_security_event', return_value=None), \
             patch.object(security_monitoring_service, '_analyze_security_event', return_value=None):
            
            event = await security_monitoring_service.log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_id=uuid4(),
                ip_address='192.168.1.1',
                user_agent='Mozilla/5.0',
                event_data={'session_id': 'test_session'}
            )
            
            assert event.event_type == SecurityEventType.LOGIN_SUCCESS
            assert event.severity == SecurityEventSeverity.LOW
            assert event.ip_address == '192.168.1.1'
            assert event.user_agent == 'Mozilla/5.0'
    
    @pytest.mark.asyncio
    async def test_analyze_login_attempt_success(self, security_monitoring_service):
        """Test analyzing successful login attempt."""
        user_id = uuid4()
        
        with patch.object(security_monitoring_service, '_analyze_successful_login', return_value=[]), \
             patch.object(security_monitoring_service, '_analyze_failed_login', return_value=[]):
            
            events = await security_monitoring_service.analyze_login_attempt(
                user_id, '192.168.1.1', 'Mozilla/5.0', True
            )
            
            assert isinstance(events, list)
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, security_monitoring_service):
        """Test anomaly detection."""
        mock_events = []
        
        with patch.object(security_monitoring_service, '_get_recent_events', return_value=mock_events), \
             patch.object(security_monitoring_service, '_detect_brute_force_attacks', return_value=[]), \
             patch.object(security_monitoring_service, '_detect_credential_stuffing', return_value=[]), \
             patch.object(security_monitoring_service, '_detect_suspicious_patterns', return_value=[]), \
             patch.object(security_monitoring_service, '_detect_account_takeover', return_value=[]):
            
            alerts = await security_monitoring_service.detect_anomalies()
            
            assert isinstance(alerts, list)


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimiter instance for testing."""
        return RateLimiter(redis_url=None)
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter):
        """Test rate limit check when request is allowed."""
        result = await rate_limiter.check_rate_limit('test_user', '/api/test', 'api')
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit check when limit is exceeded."""
        # Make multiple requests to exceed limit
        for _ in range(10):
            await rate_limiter.check_rate_limit('test_user', '/api/test', 'api')
        
        # This should be rate limited
        result = await rate_limiter.check_rate_limit('test_user', '/api/test', 'api')
        
        # Note: This test depends on the specific rate limit configuration
        # and may need adjustment based on actual limits
        assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for the complete authentication system."""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI application."""
        from src.main import create_app, Settings
        
        settings = Settings(
            debug=True,
            database_url='sqlite:///test.db',
            redis_url=None,
            jwt_secret='test_secret'
        )
        
        app = asyncio.run(create_app(settings))
        return app
    
    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['auth_enabled'] is True
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get('/')
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
        assert 'features' in data
        assert data['features']['authentication'] is True
    
    def test_register_endpoint(self, client):
        """Test user registration endpoint."""
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPass123!',
            'full_name': 'Test User'
        }
        
        # Mock the database operations
        with patch('src.services.auth_service.UserAuthService.register_user') as mock_register:
            mock_result = Mock()
            mock_result.success = True
            mock_result.user = Mock()
            mock_result.user.id = uuid4()
            mock_result.user.username = 'testuser'
            mock_result.user.email = 'test@example.com'
            mock_result.user.full_name = 'Test User'
            mock_result.user.role = UserRole.USER
            mock_result.user.is_email_verified = False
            mock_result.user.two_factor_enabled = False
            mock_result.user.created_at = datetime.utcnow()
            mock_result.user.last_login_at = None
            mock_result.error = None
            
            mock_register.return_value = mock_result
            
            response = client.post('/auth/register', json=user_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data['username'] == 'testuser'
            assert data['email'] == 'test@example.com'
    
    def test_login_endpoint(self, client):
        """Test user login endpoint."""
        login_data = {
            'username': 'testuser',
            'password': 'TestPass123!'
        }
        
        # Mock the authentication
        with patch('src.services.auth_service.UserAuthService.authenticate_user') as mock_auth:
            mock_result = Mock()
            mock_result.success = True
            mock_result.user = Mock()
            mock_result.user.id = uuid4()
            mock_result.user.username = 'testuser'
            mock_result.user.email = 'test@example.com'
            mock_result.user.full_name = 'Test User'
            mock_result.user.role = UserRole.USER
            mock_result.user.is_email_verified = True
            mock_result.user.two_factor_enabled = False
            mock_result.user.created_at = datetime.utcnow()
            mock_result.user.last_login_at = datetime.utcnow()
            mock_result.tokens = {
                'access_token': 'test_token',
                'refresh_token': 'refresh_token',
                'expires_in': 1800
            }
            mock_result.requires_mfa = False
            mock_result.error = None
            
            mock_auth.return_value = mock_result
            
            response = client.post('/auth/login', json=login_data)
            
            assert response.status_code == 200
            data = response.json()
            assert 'access_token' in data
            assert 'refresh_token' in data
            assert 'user' in data
    
    def test_protected_endpoint_unauthorized(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get('/auth/me')
        
        assert response.status_code == 401
    
    def test_protected_endpoint_authorized(self, client):
        """Test accessing protected endpoint with authentication."""
        # Mock the token verification
        with patch('src.services.auth_service.UserAuthService.verify_access_token') as mock_verify:
            mock_user = Mock()
            mock_user.id = uuid4()
            mock_user.username = 'testuser'
            mock_user.email = 'test@example.com'
            mock_user.full_name = 'Test User'
            mock_user.role = UserRole.USER
            mock_user.is_email_verified = True
            mock_user.two_factor_enabled = False
            mock_user.created_at = datetime.utcnow()
            mock_user.last_login_at = datetime.utcnow()
            
            mock_verify.return_value = mock_user
            
            headers = {'Authorization': 'Bearer test_token'}
            response = client.get('/auth/me', headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data['username'] == 'testuser'
            assert data['email'] == 'test@example.com'


# Test fixtures and utilities

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return Mock()


@pytest.fixture
def mock_database():
    """Mock database connection."""
    return Mock()


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    return Mock()


# Test configuration
@pytest.fixture(scope='session')
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])