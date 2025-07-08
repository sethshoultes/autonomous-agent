"""
Input validation and sanitization utilities for the Autonomous Agent System.

This module provides comprehensive input validation and sanitization
to prevent security vulnerabilities such as injection attacks, XSS,
and data corruption.
"""

import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from email_validator import validate_email, EmailNotValidError
from pydantic import BaseModel, Field, validator
from pydantic.error_wrappers import ValidationError


logger = logging.getLogger(__name__)


class SecurityValidator:
    """Security-focused validation utilities."""
    
    # Regular expressions for common validation patterns
    PATTERNS = {
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
        'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
        'jwt': re.compile(r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$'),
        'sql_injection': re.compile(
            r'(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript|vbscript|onload|onerror)',
            re.IGNORECASE
        ),
        'xss': re.compile(r'<script[^>]*>.*?</script>|javascript:|data:text/html|vbscript:', re.IGNORECASE),
        'path_traversal': re.compile(r'\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c', re.IGNORECASE),
        'command_injection': re.compile(r'[;&|`$()]', re.IGNORECASE),
    }
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize a string input by removing dangerous characters."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Truncate if too long
        if len(value) > max_length:
            logger.warning(f"String truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # HTML escape
        value = html.escape(value)
        
        # Remove potentially dangerous sequences
        value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE)
        value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
        value = re.sub(r'data:text/html', '', value, flags=re.IGNORECASE)
        
        return value.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format."""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    @staticmethod
    def validate_url(url: str, allowed_schemes: List[str] = None) -> bool:
        """Validate URL format and scheme."""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in allowed_schemes and
                parsed.netloc and
                not SecurityValidator.PATTERNS['xss'].search(url)
            )
        except Exception:
            return False
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> bool:
        """Validate UUID format."""
        return bool(SecurityValidator.PATTERNS['uuid'].match(uuid_str.lower()))
    
    @staticmethod
    def validate_jwt(token: str) -> bool:
        """Validate JWT token format."""
        return bool(SecurityValidator.PATTERNS['jwt'].match(token))
    
    @staticmethod
    def detect_sql_injection(value: str) -> bool:
        """Detect potential SQL injection attempts."""
        return bool(SecurityValidator.PATTERNS['sql_injection'].search(value))
    
    @staticmethod
    def detect_xss(value: str) -> bool:
        """Detect potential XSS attempts."""
        return bool(SecurityValidator.PATTERNS['xss'].search(value))
    
    @staticmethod
    def detect_path_traversal(value: str) -> bool:
        """Detect path traversal attempts."""
        return bool(SecurityValidator.PATTERNS['path_traversal'].search(value))
    
    @staticmethod
    def detect_command_injection(value: str) -> bool:
        """Detect command injection attempts."""
        return bool(SecurityValidator.PATTERNS['command_injection'].search(value))
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Check if filename is safe."""
        if not filename or filename in ['.', '..']:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*', '\x00']
        for char in dangerous_chars:
            if char in filename:
                return False
        
        # Check for reserved Windows filenames
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        if filename.upper() in reserved_names:
            return False
        
        return True


class InputValidator:
    """Main input validation class."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
    
    def validate_json(self, data: Union[str, Dict[str, Any]], schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate and parse JSON data."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        
        if not isinstance(data, dict):
            raise ValueError("JSON data must be an object")
        
        # Recursively sanitize string values
        data = self._sanitize_dict(data)
        
        # Additional security checks
        json_str = json.dumps(data)
        if len(json_str) > 1048576:  # 1MB limit
            raise ValueError("JSON data too large")
        
        # Check for dangerous patterns
        if self.security_validator.detect_sql_injection(json_str):
            raise ValueError("Potential SQL injection detected")
        
        if self.security_validator.detect_xss(json_str):
            raise ValueError("Potential XSS detected")
        
        return data
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        sanitized = {}
        for key, value in data.items():
            # Sanitize keys
            if not isinstance(key, str):
                key = str(key)
            key = self.security_validator.sanitize_string(key, max_length=100)
            
            # Sanitize values
            if isinstance(value, str):
                value = self.security_validator.sanitize_string(value)
            elif isinstance(value, dict):
                value = self._sanitize_dict(value)
            elif isinstance(value, list):
                value = self._sanitize_list(value)
            
            sanitized[key] = value
        
        return sanitized
    
    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """Recursively sanitize list values."""
        sanitized = []
        for item in data:
            if isinstance(item, str):
                item = self.security_validator.sanitize_string(item)
            elif isinstance(item, dict):
                item = self._sanitize_dict(item)
            elif isinstance(item, list):
                item = self._sanitize_list(item)
            
            sanitized.append(item)
        
        return sanitized
    
    def validate_file_upload(self, filename: str, content: bytes, 
                           allowed_types: List[str] = None, 
                           max_size: int = 10485760) -> bool:  # 10MB default
        """Validate file upload."""
        if not self.security_validator.is_safe_filename(filename):
            raise ValueError("Invalid filename")
        
        if len(content) > max_size:
            raise ValueError(f"File too large (max {max_size} bytes)")
        
        if allowed_types:
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if file_ext not in allowed_types:
                raise ValueError(f"File type not allowed: {file_ext}")
        
        # Check for executable content
        dangerous_signatures = [
            b'MZ',  # Windows executable
            b'\x7fELF',  # Linux executable
            b'#!/',  # Script shebang
            b'<?php',  # PHP script
            b'<script',  # JavaScript
        ]
        
        for sig in dangerous_signatures:
            if content.startswith(sig):
                raise ValueError("Executable content detected")
        
        return True
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Check length (typical API keys are 32-64 characters)
        if len(api_key) < 32 or len(api_key) > 128:
            return False
        
        # Check for dangerous patterns
        if (self.security_validator.detect_sql_injection(api_key) or
            self.security_validator.detect_xss(api_key) or
            self.security_validator.detect_command_injection(api_key)):
            return False
        
        return True
    
    def validate_agent_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent configuration."""
        # Define required fields
        required_fields = ['name', 'type']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate specific fields
        if not isinstance(config['name'], str) or len(config['name']) > 100:
            raise ValueError("Invalid agent name")
        
        if not isinstance(config['type'], str) or len(config['type']) > 50:
            raise ValueError("Invalid agent type")
        
        # Sanitize configuration
        config = self._sanitize_dict(config)
        
        return config


class ValidationError(Exception):
    """Custom validation error."""
    pass


# Pydantic models for request validation
class AgentConfigModel(BaseModel):
    """Pydantic model for agent configuration validation."""
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    configuration: Optional[Dict[str, Any]] = None
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Name must be a non-empty string')
        return SecurityValidator.sanitize_string(v, max_length=100)
    
    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['email', 'github', 'research', 'chat', 'monitoring']
        if v not in allowed_types:
            raise ValueError(f'Type must be one of: {allowed_types}')
        return v
    
    @validator('configuration')
    def validate_configuration(cls, v):
        if v is not None:
            # Ensure it's a dictionary
            if not isinstance(v, dict):
                raise ValueError('Configuration must be a dictionary')
            
            # Check for dangerous patterns
            config_str = json.dumps(v)
            if SecurityValidator.detect_sql_injection(config_str):
                raise ValueError('Configuration contains dangerous patterns')
            
            if SecurityValidator.detect_xss(config_str):
                raise ValueError('Configuration contains XSS patterns')
        
        return v


class MessageModel(BaseModel):
    """Pydantic model for message validation."""
    content: str = Field(..., min_length=1, max_length=10000)
    message_type: str = Field(..., min_length=1, max_length=50)
    recipient: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('content')
    def validate_content(cls, v):
        return SecurityValidator.sanitize_string(v, max_length=10000)
    
    @validator('message_type')
    def validate_message_type(cls, v):
        allowed_types = ['email', 'notification', 'command', 'response']
        if v not in allowed_types:
            raise ValueError(f'Message type must be one of: {allowed_types}')
        return v
    
    @validator('recipient')
    def validate_recipient(cls, v):
        if v is not None:
            return SecurityValidator.sanitize_string(v, max_length=100)
        return v


class TaskModel(BaseModel):
    """Pydantic model for task validation."""
    task_type: str = Field(..., min_length=1, max_length=50)
    task_data: Dict[str, Any] = Field(...)
    priority: int = Field(default=5, ge=1, le=10)
    
    @validator('task_type')
    def validate_task_type(cls, v):
        allowed_types = ['email_process', 'github_sync', 'research', 'monitoring']
        if v not in allowed_types:
            raise ValueError(f'Task type must be one of: {allowed_types}')
        return v
    
    @validator('task_data')
    def validate_task_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Task data must be a dictionary')
        
        # Check for dangerous patterns
        data_str = json.dumps(v)
        if SecurityValidator.detect_sql_injection(data_str):
            raise ValueError('Task data contains dangerous patterns')
        
        if SecurityValidator.detect_xss(data_str):
            raise ValueError('Task data contains XSS patterns')
        
        return v