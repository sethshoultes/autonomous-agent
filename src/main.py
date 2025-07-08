"""
Main entry point for the Autonomous Agent System.

This module serves as the primary entry point for the autonomous agent system,
following SOLID principles and providing a clean, extensible architecture.
"""

import asyncio
import logging
import secrets
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import redis.asyncio as redis

from .services.auth_service import UserAuthService
from .services.mfa_service import MFAService
from .services.oauth_service import OAuthService
from .services.user_management_service import UserManagementService
from .services.security_monitoring_service import SecurityMonitoringService
from .api.auth_routes import create_auth_router
from .security.middleware import create_security_middleware_stack
from .security.rate_limiting import RateLimiter
from .monitoring.metrics import MetricsCollector


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application settings
    app_name: str = "Autonomous Agent System"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Database settings
    database_url: str = "postgresql://agent:password@localhost/autonomous_agent"
    redis_url: str = "redis://localhost:6379"

    # Security settings
    jwt_secret: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    max_login_attempts: int = 5
    lockout_duration: int = 300
    
    # MFA settings
    totp_issuer: str = "Autonomous Agent System"
    sms_enabled: bool = False
    email_enabled: bool = True
    
    # OAuth settings
    oauth_enabled: bool = True
    oauth_providers: dict = {}
    
    # Rate limiting settings
    rate_limiting_enabled: bool = True
    
    # Security monitoring
    security_monitoring_enabled: bool = True
    geoip_db_path: Optional[str] = None
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: list = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    app_name: str
    version: str
    auth_enabled: bool
    mfa_enabled: bool
    oauth_enabled: bool


def configure_logging(log_level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


async def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings instance. If None, creates default settings.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
    )

    # Initialize Redis client
    redis_client = None
    if settings.redis_url:
        try:
            redis_client = redis.from_url(settings.redis_url)
            await redis_client.ping()
            logging.info("Redis connection established")
        except Exception as e:
            logging.warning(f"Redis connection failed: {e}")
            redis_client = None

    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Initialize rate limiter
    rate_limiter = RateLimiter(
        redis_url=settings.redis_url,
        config={
            'enabled': settings.rate_limiting_enabled,
            'rules': {
                'auth': {'requests': 5, 'window': 300, 'burst': 2},
                'api': {'requests': 100, 'window': 60, 'burst': 10},
                'global': {'requests': 1000, 'window': 3600, 'burst': 100}
            }
        }
    )

    # Initialize authentication services
    auth_config = {
        'jwt_secret': settings.jwt_secret,
        'jwt_algorithm': settings.jwt_algorithm,
        'access_token_expire_minutes': settings.access_token_expire_minutes,
        'refresh_token_expire_days': settings.refresh_token_expire_days,
        'max_login_attempts': settings.max_login_attempts,
        'lockout_duration': settings.lockout_duration,
        'redis_url': settings.redis_url,
        'database_url': settings.database_url,
        'rate_limiting': {
            'enabled': settings.rate_limiting_enabled
        }
    }
    
    auth_service = UserAuthService(
        config=auth_config,
        redis_client=redis_client,
        metrics_collector=metrics_collector
    )

    # Initialize MFA service
    mfa_config = {
        'totp_issuer': settings.totp_issuer,
        'sms_enabled': settings.sms_enabled,
        'email_enabled': settings.email_enabled,
        'database_url': settings.database_url
    }
    
    mfa_service = MFAService(config=mfa_config)

    # Initialize OAuth service
    oauth_service = None
    if settings.oauth_enabled:
        oauth_config = {
            'oauth_providers': settings.oauth_providers,
            'database_url': settings.database_url
        }
        oauth_service = OAuthService(config=oauth_config)

    # Initialize user management service
    user_management_service = UserManagementService(
        config={'database_url': settings.database_url},
        metrics_collector=metrics_collector
    )

    # Initialize security monitoring service
    security_monitoring_service = None
    if settings.security_monitoring_enabled:
        security_config = {
            'database_url': settings.database_url,
            'geoip_db_path': settings.geoip_db_path,
            'max_login_attempts': settings.max_login_attempts,
            'rate_limit_threshold': 100,
            'alert_retention_days': 90
        }
        security_monitoring_service = SecurityMonitoringService(
            config=security_config,
            metrics_collector=metrics_collector
        )

    # Add security middleware
    security_middleware_config = {
        'security': {
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'require_https': not settings.debug,
            'blocked_user_agents': [],
            'blocked_ips': []
        },
        'cors': {
            'enabled': settings.cors_enabled,
            'allowed_origins': settings.cors_origins,
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allowed_headers': ['*']
        },
        'validation': {
            'validate_json': True,
            'sanitize_inputs': True,
            'max_json_size': 1024 * 1024  # 1MB
        },
        'csp': {
            'csp_policy': {
                'default-src': "'self'",
                'script-src': "'self' 'unsafe-inline'",
                'style-src': "'self' 'unsafe-inline'",
                'img-src': "'self' data:",
                'font-src': "'self'",
                'connect-src': "'self'",
                'frame-src': "'none'",
                'object-src': "'none'"
            }
        },
        'logging': {
            'log_requests': True,
            'log_responses': True,
            'log_body': settings.debug,
            'sensitive_headers': ['authorization', 'cookie', 'x-api-key']
        }
    }
    
    create_security_middleware_stack(app, security_middleware_config)

    # Add authentication routes
    auth_router = create_auth_router(
        auth_service=auth_service,
        mfa_service=mfa_service,
        oauth_service=oauth_service,
        rate_limiter=rate_limiter,
        metrics_collector=metrics_collector
    )
    app.include_router(auth_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            version=settings.app_version,
            auth_enabled=True,
            mfa_enabled=settings.sms_enabled or settings.email_enabled,
            oauth_enabled=settings.oauth_enabled
        )

    # Root endpoint
    @app.get("/")
    async def root() -> dict:
        """Root endpoint."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "features": {
                "authentication": True,
                "multi_factor_auth": settings.sms_enabled or settings.email_enabled,
                "oauth": settings.oauth_enabled,
                "rate_limiting": settings.rate_limiting_enabled,
                "security_monitoring": settings.security_monitoring_enabled
            }
        }

    # Metrics endpoint (if debug mode)
    if settings.debug:
        @app.get("/metrics")
        async def get_metrics():
            """Get application metrics."""
            return await metrics_collector.get_metrics()

    # Store services in app state for access in other modules
    app.state.auth_service = auth_service
    app.state.mfa_service = mfa_service
    app.state.oauth_service = oauth_service
    app.state.user_management_service = user_management_service
    app.state.security_monitoring_service = security_monitoring_service
    app.state.rate_limiter = rate_limiter
    app.state.metrics_collector = metrics_collector
    app.state.redis_client = redis_client

    return app


async def main() -> None:
    """
    Main application entry point.

    This function initializes the application, sets up logging,
    and starts the web server.
    """
    settings = Settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Authentication enabled: True")
    logger.info(f"MFA enabled: {settings.sms_enabled or settings.email_enabled}")
    logger.info(f"OAuth enabled: {settings.oauth_enabled}")
    logger.info(f"Rate limiting enabled: {settings.rate_limiting_enabled}")
    logger.info(f"Security monitoring enabled: {settings.security_monitoring_enabled}")

    app = await create_app(settings)

    try:
        import uvicorn

        config = uvicorn.Config(
            app=app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        
        # Clean up services
        if hasattr(app.state, 'redis_client') and app.state.redis_client:
            await app.state.redis_client.close()
        if hasattr(app.state, 'oauth_service') and app.state.oauth_service:
            await app.state.oauth_service.close()
            
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
