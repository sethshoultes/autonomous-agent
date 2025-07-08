#!/usr/bin/env python3
"""
Authentication System Integration Example.

This script demonstrates how to integrate and use the comprehensive authentication
and authorization system in the autonomous agent system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our services
from src.services.auth_service import (
    UserAuthService, UserRegistrationRequest, UserLoginRequest, 
    PasswordChangeRequest
)
from src.services.mfa_service import MFAService, MFASetupRequest, MFAVerifyRequest
from src.services.oauth_service import OAuthService, OAuthProvider
from src.services.user_management_service import (
    UserManagementService, UserProfileUpdate, UserPreferencesUpdate
)
from src.services.security_monitoring_service import (
    SecurityMonitoringService, SecurityEventType, SecurityEventSeverity
)
from src.security.rate_limiting import RateLimiter
from src.monitoring.metrics import MetricsCollector


class AuthenticationDemo:
    """Demonstration of the authentication system capabilities."""
    
    def __init__(self):
        """Initialize the demo with mock configuration."""
        self.config = {
            'jwt_secret': 'demo_secret_key_change_in_production',
            'jwt_algorithm': 'HS256',
            'access_token_expire_minutes': 30,
            'refresh_token_expire_days': 30,
            'max_login_attempts': 5,
            'lockout_duration': 300,
            'redis_url': None,  # Use in-memory for demo
            'database_url': 'sqlite:///demo.db',
            'totp_issuer': 'Autonomous Agent Demo',
            'sms_enabled': False,
            'email_enabled': True,
            'oauth_providers': {
                'google': {
                    'client_id': 'demo_google_client_id',
                    'client_secret': 'demo_google_client_secret',
                    'redirect_uri': 'http://localhost:8000/auth/oauth/google/callback'
                }
            }
        }
        
        # Initialize services
        self.metrics_collector = MetricsCollector()
        self.auth_service = UserAuthService(
            config=self.config,
            redis_client=None,
            metrics_collector=self.metrics_collector
        )
        self.mfa_service = MFAService(config=self.config)
        self.oauth_service = OAuthService(config=self.config)
        self.user_management_service = UserManagementService(
            config=self.config,
            metrics_collector=self.metrics_collector
        )
        self.security_monitoring_service = SecurityMonitoringService(
            config=self.config,
            metrics_collector=self.metrics_collector
        )
        self.rate_limiter = RateLimiter(redis_url=None, config=self.config)
    
    async def demonstrate_user_registration(self):
        """Demonstrate user registration process."""
        logger.info("=== User Registration Demo ===")
        
        # Create registration request
        registration_data = UserRegistrationRequest(
            username='demo_user',
            email='demo@example.com',
            password='SecurePass123!',
            full_name='Demo User'
        )
        
        # Register user
        result = await self.auth_service.register_user(registration_data)
        
        if result.success:
            logger.info(f"User registered successfully: {result.user.username}")
            logger.info(f"User ID: {result.user.id}")
            logger.info(f"Email verification required: {not result.user.is_email_verified}")
            return result.user
        else:
            logger.error(f"Registration failed: {result.error}")
            return None
    
    async def demonstrate_user_login(self, user):
        """Demonstrate user login process."""
        logger.info("=== User Login Demo ===")
        
        # Create mock request object
        class MockRequest:
            def __init__(self):
                self.client = MockClient()
                self.headers = {'user-agent': 'Demo Client 1.0'}
        
        class MockClient:
            def __init__(self):
                self.host = '127.0.0.1'
        
        mock_request = MockRequest()
        
        # Create login request
        login_data = UserLoginRequest(
            username='demo_user',
            password='SecurePass123!'
        )
        
        # Attempt login
        result = await self.auth_service.authenticate_user(login_data, mock_request)
        
        if result.success:
            logger.info(f"Login successful for user: {result.user.username}")
            logger.info(f"Access token generated: {result.tokens['access_token'][:20]}...")
            logger.info(f"Token expires in: {result.tokens['expires_in']} seconds")
            return result.tokens
        elif result.requires_mfa:
            logger.info("MFA verification required")
            logger.info(f"MFA token: {result.mfa_token}")
            return None
        else:
            logger.error(f"Login failed: {result.error}")
            return None
    
    async def demonstrate_mfa_setup(self, user):
        """Demonstrate MFA setup process."""
        logger.info("=== MFA Setup Demo ===")
        
        # Setup TOTP
        totp_result = await self.mfa_service.setup_totp(user.id)
        
        logger.info("TOTP setup completed")
        logger.info(f"Secret: {totp_result.secret}")
        logger.info(f"Backup codes: {len(totp_result.backup_codes)} generated")
        logger.info(f"Recovery codes: {len(totp_result.recovery_codes)} generated")
        logger.info(f"QR code generated: {len(totp_result.qr_code)} bytes")
        
        # Simulate MFA verification and enabling
        # In real implementation, user would scan QR code and enter TOTP code
        import pyotp
        totp = pyotp.TOTP(totp_result.secret)
        current_code = totp.now()
        
        logger.info(f"Current TOTP code: {current_code}")
        
        # Verify and enable MFA
        verify_result = await self.mfa_service.verify_and_enable_mfa(user.id, current_code)
        
        if verify_result:
            logger.info("MFA enabled successfully")
        else:
            logger.error("MFA verification failed")
        
        return verify_result
    
    async def demonstrate_oauth_flow(self):
        """Demonstrate OAuth flow."""
        logger.info("=== OAuth Flow Demo ===")
        
        # Get authorization URL
        auth_result = await self.oauth_service.get_authorization_url(OAuthProvider.GOOGLE)
        
        logger.info("OAuth authorization URL generated")
        logger.info(f"URL: {auth_result.authorization_url}")
        logger.info(f"State: {auth_result.state}")
        
        # Simulate OAuth callback (in real implementation, this would be triggered by OAuth provider)
        logger.info("OAuth callback would be handled by the OAuth provider")
        
        return auth_result
    
    async def demonstrate_user_management(self, user):
        """Demonstrate user management capabilities."""
        logger.info("=== User Management Demo ===")
        
        # Update user profile
        profile_update = UserProfileUpdate(
            full_name='Updated Demo User',
            bio='This is a demonstration user profile',
            location='Demo City, Demo State',
            company='Demo Company',
            timezone='UTC'
        )
        
        profile_result = await self.user_management_service.update_user_profile(
            user.id, profile_update
        )
        
        if profile_result:
            logger.info("User profile updated successfully")
        else:
            logger.error("Profile update failed")
        
        # Update user preferences
        preferences_update = UserPreferencesUpdate(
            email_notifications=True,
            push_notifications=False,
            theme='dark',
            language='en'
        )
        
        preferences_result = await self.user_management_service.update_user_preferences(
            user.id, preferences_update
        )
        
        if preferences_result:
            logger.info("User preferences updated successfully")
        else:
            logger.error("Preferences update failed")
        
        # Get user statistics
        stats = await self.user_management_service.get_user_stats()
        logger.info(f"User statistics: {stats}")
        
        return profile_result and preferences_result
    
    async def demonstrate_security_monitoring(self, user):
        """Demonstrate security monitoring capabilities."""
        logger.info("=== Security Monitoring Demo ===")
        
        # Log some security events
        await self.security_monitoring_service.log_security_event(
            SecurityEventType.LOGIN_SUCCESS,
            user_id=user.id,
            ip_address='192.168.1.100',
            user_agent='Demo Client 1.0',
            event_data={'session_id': 'demo_session_123'}
        )
        
        await self.security_monitoring_service.log_security_event(
            SecurityEventType.PASSWORD_CHANGE,
            user_id=user.id,
            ip_address='192.168.1.100',
            user_agent='Demo Client 1.0',
            event_data={'method': 'manual'}
        )
        
        # Analyze login attempt
        login_events = await self.security_monitoring_service.analyze_login_attempt(
            user.id, '192.168.1.100', 'Demo Client 1.0', True
        )
        
        logger.info(f"Login analysis generated {len(login_events)} events")
        
        # Get security metrics
        metrics = await self.security_monitoring_service.get_security_metrics()
        logger.info(f"Security metrics: {metrics}")
        
        return True
    
    async def demonstrate_rate_limiting(self):
        """Demonstrate rate limiting capabilities."""
        logger.info("=== Rate Limiting Demo ===")
        
        # Test rate limiting
        client_ip = '192.168.1.100'
        endpoint = '/api/test'
        
        # Make several requests
        for i in range(7):
            allowed = await self.rate_limiter.check_rate_limit(client_ip, endpoint, 'api')
            logger.info(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")
        
        # Get rate limit status
        status = await self.rate_limiter.get_rate_limit_status(client_ip, endpoint, 'api')
        logger.info(f"Rate limit status: {status}")
        
        return True
    
    async def demonstrate_metrics_collection(self):
        """Demonstrate metrics collection."""
        logger.info("=== Metrics Collection Demo ===")
        
        # Record some metrics
        await self.metrics_collector.increment_counter('demo_counter', 1.0, {'type': 'test'})
        await self.metrics_collector.set_gauge('demo_gauge', 42.0, {'status': 'active'})
        await self.metrics_collector.observe_histogram('demo_histogram', 0.5, {'endpoint': '/test'})
        
        # Get metrics
        metrics = await self.metrics_collector.get_metrics()
        logger.info(f"Collected {len(metrics['counters'])} counter metrics")
        logger.info(f"Collected {len(metrics['gauges'])} gauge metrics")
        logger.info(f"Collected {len(metrics['histograms'])} histogram metrics")
        
        # Get summary
        summary = await self.metrics_collector.get_summary()
        logger.info(f"Metrics summary: {summary}")
        
        return True
    
    async def demonstrate_password_change(self, user):
        """Demonstrate password change process."""
        logger.info("=== Password Change Demo ===")
        
        # Change password
        password_change = PasswordChangeRequest(
            current_password='SecurePass123!',
            new_password='NewSecurePass456!'
        )
        
        result = await self.auth_service.change_password(user.id, password_change)
        
        if result:
            logger.info("Password changed successfully")
            logger.info("All user sessions have been invalidated")
        else:
            logger.error("Password change failed")
        
        return result
    
    async def run_complete_demo(self):
        """Run the complete authentication system demonstration."""
        logger.info("Starting Autonomous Agent Authentication System Demo")
        logger.info("=" * 60)
        
        try:
            # 1. User Registration
            user = await self.demonstrate_user_registration()
            if not user:
                logger.error("Demo cannot continue without user registration")
                return
            
            # 2. User Login
            tokens = await self.demonstrate_user_login(user)
            if not tokens:
                logger.warning("Login demo failed, but continuing with other demos")
            
            # 3. MFA Setup
            await self.demonstrate_mfa_setup(user)
            
            # 4. OAuth Flow
            await self.demonstrate_oauth_flow()
            
            # 5. User Management
            await self.demonstrate_user_management(user)
            
            # 6. Security Monitoring
            await self.demonstrate_security_monitoring(user)
            
            # 7. Rate Limiting
            await self.demonstrate_rate_limiting()
            
            # 8. Metrics Collection
            await self.demonstrate_metrics_collection()
            
            # 9. Password Change
            await self.demonstrate_password_change(user)
            
            logger.info("=" * 60)
            logger.info("Demo completed successfully!")
            logger.info("Authentication system is fully functional")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {str(e)}")
            raise


async def main():
    """Main demo function."""
    demo = AuthenticationDemo()
    await demo.run_complete_demo()


if __name__ == '__main__':
    asyncio.run(main())