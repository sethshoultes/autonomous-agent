"""
Security middleware for the Autonomous Agent System.

This module provides middleware for security functions including:
- Input validation and sanitization
- Security headers
- Request filtering
- CORS handling
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware

from .validation import InputValidator, SecurityValidator, ValidationError
from .rate_limiting import RateLimiter


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request processing."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter()
        
        # Security configuration
        self.max_request_size = self.config.get('max_request_size', 10 * 1024 * 1024)  # 10MB
        self.allowed_origins = self.config.get('allowed_origins', ['http://localhost:3000'])
        self.blocked_user_agents = self.config.get('blocked_user_agents', [])
        self.blocked_ips = self.config.get('blocked_ips', [])
        self.require_https = self.config.get('require_https', False)
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through security middleware."""
        start_time = time.time()
        
        try:
            # Pre-request security checks
            await self._pre_request_checks(request)
            
            # Process request
            response = await call_next(request)
            
            # Post-request security headers
            response = await self._add_security_headers(response)
            
            # Log request
            processing_time = time.time() - start_time
            await self._log_request(request, response, processing_time)
            
            return response
            
        except HTTPException as e:
            # Handle HTTP exceptions
            logger.warning(f"HTTP exception in security middleware: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "type": "security_error"}
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in security middleware: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error", "type": "server_error"}
            )
    
    async def _pre_request_checks(self, request: Request) -> None:
        """Perform pre-request security checks."""
        # Check HTTPS requirement
        if self.require_https and request.url.scheme != 'https':
            raise HTTPException(
                status_code=status.HTTP_426_UPGRADE_REQUIRED,
                detail="HTTPS required"
            )
        
        # Check request size
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        # Check user agent
        user_agent = request.headers.get('user-agent', '').lower()
        for blocked_ua in self.blocked_user_agents:
            if blocked_ua.lower() in user_agent:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Blocked user agent"
                )
        
        # Check IP address
        client_ip = self._get_client_ip(request)
        if client_ip in self.blocked_ips:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Blocked IP address"
            )
        
        # Rate limiting
        if not await self.rate_limiter.check_rate_limit(client_ip, request.url.path):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Validate URL
        if not self._validate_url(request.url.path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid URL"
            )
        
        # Check for common attack patterns in URL
        if self._detect_attack_patterns(request.url.path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malicious request detected"
            )
    
    async def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        # Security headers
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else '127.0.0.1'
    
    def _validate_url(self, path: str) -> bool:
        """Validate URL path."""
        # Check for path traversal
        if SecurityValidator.detect_path_traversal(path):
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r', '\n']
        for char in dangerous_chars:
            if char in path:
                return False
        
        return True
    
    def _detect_attack_patterns(self, path: str) -> bool:
        """Detect common attack patterns in URL."""
        # SQL injection patterns
        if SecurityValidator.detect_sql_injection(path):
            return True
        
        # XSS patterns
        if SecurityValidator.detect_xss(path):
            return True
        
        # Command injection patterns
        if SecurityValidator.detect_command_injection(path):
            return True
        
        return False
    
    async def _log_request(self, request: Request, response: Response, processing_time: float) -> None:
        """Log request details."""
        log_data = {
            'method': request.method,
            'url': str(request.url),
            'status_code': response.status_code,
            'processing_time': processing_time,
            'client_ip': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent', ''),
            'content_length': request.headers.get('content-length', '0'),
        }
        
        if response.status_code >= 400:
            logger.warning(f"Request failed: {log_data}")
        else:
            logger.info(f"Request processed: {log_data}")


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response validation."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.validator = InputValidator()
        
        # Validation configuration
        self.validate_json = self.config.get('validate_json', True)
        self.sanitize_inputs = self.config.get('sanitize_inputs', True)
        self.max_json_size = self.config.get('max_json_size', 1024 * 1024)  # 1MB
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through validation middleware."""
        try:
            # Validate request
            if self.validate_json and await self._is_json_request(request):
                await self._validate_json_request(request)
            
            # Process request
            response = await call_next(request)
            
            return response
            
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": str(e), "type": "validation_error"}
            )
        except Exception as e:
            logger.error(f"Unexpected error in validation middleware: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error", "type": "server_error"}
            )
    
    async def _is_json_request(self, request: Request) -> bool:
        """Check if request contains JSON data."""
        content_type = request.headers.get('content-type', '')
        return 'application/json' in content_type
    
    async def _validate_json_request(self, request: Request) -> None:
        """Validate JSON request body."""
        try:
            # Get request body
            body = await request.body()
            
            # Check size
            if len(body) > self.max_json_size:
                raise ValidationError("JSON payload too large")
            
            # Parse and validate JSON
            if body:
                try:
                    json_data = json.loads(body.decode('utf-8'))
                    
                    # Validate with our validator
                    validated_data = self.validator.validate_json(json_data)
                    
                    # Store validated data for use in request
                    request.state.validated_json = validated_data
                    
                except json.JSONDecodeError as e:
                    raise ValidationError(f"Invalid JSON: {str(e)}")
                except ValueError as e:
                    raise ValidationError(f"Validation error: {str(e)}")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Failed to validate request: {str(e)}")


class CORSSecurityMiddleware:
    """CORS middleware with security enhancements."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # CORS configuration
        allowed_origins = self.config.get('allowed_origins', ['http://localhost:3000'])
        allowed_methods = self.config.get('allowed_methods', ['GET', 'POST', 'PUT', 'DELETE'])
        allowed_headers = self.config.get('allowed_headers', ['*'])
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=allowed_methods,
            allow_headers=allowed_headers,
            expose_headers=['X-Total-Count', 'X-Page-Count']
        )


class CSPMiddleware(BaseHTTPMiddleware):
    """Content Security Policy middleware."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        
        # CSP configuration
        self.csp_policy = self.config.get('csp_policy', {
            'default-src': "'self'",
            'script-src': "'self' 'unsafe-inline'",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data:",
            'font-src': "'self'",
            'connect-src': "'self'",
            'frame-src': "'none'",
            'object-src': "'none'",
            'base-uri': "'self'",
            'form-action': "'self'",
            'frame-ancestors': "'none'",
            'block-all-mixed-content': '',
            'upgrade-insecure-requests': ''
        })
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add CSP headers to response."""
        response = await call_next(request)
        
        # Build CSP policy string
        csp_parts = []
        for directive, value in self.csp_policy.items():
            if value:
                csp_parts.append(f"{directive} {value}")
            else:
                csp_parts.append(directive)
        
        csp_policy = "; ".join(csp_parts)
        response.headers["Content-Security-Policy"] = csp_policy
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request logging."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Logging configuration
        self.log_requests = self.config.get('log_requests', True)
        self.log_responses = self.config.get('log_responses', True)
        self.log_body = self.config.get('log_body', False)
        self.sensitive_headers = self.config.get('sensitive_headers', [
            'authorization', 'cookie', 'x-api-key', 'x-auth-token'
        ])
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log request and response details."""
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request)
        
        # Process request
        response = await call_next(request)
        
        # Log response
        processing_time = time.time() - start_time
        if self.log_responses:
            await self._log_response(request, response, processing_time)
        
        return response
    
    async def _log_request(self, request: Request) -> None:
        """Log request details."""
        # Filter sensitive headers
        headers = dict(request.headers)
        for header in self.sensitive_headers:
            if header in headers:
                headers[header] = '[REDACTED]'
        
        log_data = {
            'type': 'request',
            'method': request.method,
            'url': str(request.url),
            'headers': headers,
            'client_ip': request.client.host if request.client else None,
        }
        
        # Log body if enabled and safe
        if self.log_body and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body and len(body) < 1024:  # Only log small bodies
                    log_data['body'] = body.decode('utf-8')
            except Exception:
                pass
        
        logger.info(f"Request: {log_data}")
    
    async def _log_response(self, request: Request, response: Response, processing_time: float) -> None:
        """Log response details."""
        log_data = {
            'type': 'response',
            'method': request.method,
            'url': str(request.url),
            'status_code': response.status_code,
            'processing_time': processing_time,
            'response_size': len(response.body) if hasattr(response, 'body') else 0,
        }
        
        # Log level based on status code
        if response.status_code >= 500:
            logger.error(f"Response: {log_data}")
        elif response.status_code >= 400:
            logger.warning(f"Response: {log_data}")
        else:
            logger.info(f"Response: {log_data}")


def create_security_middleware_stack(app, config: Optional[Dict[str, Any]] = None) -> None:
    """Create a complete security middleware stack."""
    if config is None:
        config = {}
    
    # Add middleware in reverse order (last added = first executed)
    
    # Request logging (outermost)
    app.add_middleware(RequestLoggingMiddleware, config=config.get('logging', {}))
    
    # Content Security Policy
    app.add_middleware(CSPMiddleware, config=config.get('csp', {}))
    
    # CORS (if enabled)
    if config.get('cors', {}).get('enabled', False):
        CORSSecurityMiddleware(app, config=config.get('cors', {}))
    
    # Validation middleware
    app.add_middleware(ValidationMiddleware, config=config.get('validation', {}))
    
    # Security middleware (innermost)
    app.add_middleware(SecurityMiddleware, config=config.get('security', {}))