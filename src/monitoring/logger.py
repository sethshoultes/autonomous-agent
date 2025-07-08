"""
Advanced logging system for the Autonomous Agent System.

This module provides structured logging, log aggregation, and log management
with support for different output formats and destinations.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager

import structlog
from structlog.stdlib import LoggerFactory
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer, TimeStamper, add_log_level, CallsiteParameterAdder


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger_name": self.logger_name,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
            **self.extra
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    def __init__(self):
        super().__init__()
        self.sensitive_keys = {
            'password', 'token', 'secret', 'key', 'credential', 'auth',
            'jwt', 'api_key', 'access_token', 'refresh_token', 'private_key'
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Check if message contains sensitive patterns
            msg_lower = record.msg.lower()
            for sensitive_key in self.sensitive_keys:
                if sensitive_key in msg_lower:
                    # Replace with placeholder
                    record.msg = record.msg.replace(
                        record.msg[record.msg.lower().find(sensitive_key):],
                        f"[REDACTED-{sensitive_key.upper()}]"
                    )
        
        # Filter sensitive information from record attributes
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, dict):
                    filtered_args.append(self._filter_dict(arg))
                elif isinstance(arg, str):
                    filtered_args.append(self._filter_string(arg))
                else:
                    filtered_args.append(arg)
            record.args = tuple(filtered_args)
        
        return True
    
    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive keys from dictionary."""
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_keys):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_dict(value)
            else:
                filtered[key] = value
        return filtered
    
    def _filter_string(self, text: str) -> str:
        """Filter sensitive patterns from string."""
        text_lower = text.lower()
        for sensitive_key in self.sensitive_keys:
            if sensitive_key in text_lower:
                return f"[REDACTED-{sensitive_key.upper()}]"
        return text


class StructuredLogger:
    """Structured logger with enhanced features."""
    
    def __init__(self, 
                 name: str,
                 level: LogLevel = LogLevel.INFO,
                 format_json: bool = False,
                 include_caller: bool = False):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            level: Log level
            format_json: Whether to format as JSON
            include_caller: Whether to include caller information
        """
        self.name = name
        self.level = level
        self.format_json = format_json
        self.include_caller = include_caller
        
        # Configure structlog
        processors = [
            TimeStamper(fmt="iso"),
            add_log_level,
        ]
        
        if include_caller:
            processors.append(CallsiteParameterAdder())
        
        if format_json:
            processors.append(JSONRenderer())
        else:
            processors.append(ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(name)
        
        # Set log level
        logging.getLogger(name).setLevel(level.value)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding context to logs."""
        bound_logger = self.logger.bind(**kwargs)
        old_logger = self.logger
        self.logger = bound_logger
        try:
            yield self
        finally:
            self.logger = old_logger


class LogManager:
    """Centralized log management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize log manager.
        
        Args:
            config: Logging configuration
        """
        self.config = config or {}
        self.loggers: Dict[str, StructuredLogger] = {}
        self.handlers: List[logging.Handler] = []
        
        # Configuration
        self.log_level = LogLevel(self.config.get('level', 'INFO'))
        self.log_format = self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_file = self.config.get('file', '/app/logs/app.log')
        self.max_file_size = self.config.get('max_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = self.config.get('backup_count', 5)
        self.json_format = self.config.get('json_format', False)
        self.structured_logging = self.config.get('structured_logging', True)
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create log directory
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level.value)
        
        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level.value)
        
        # Create formatter
        if self.json_format:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(self.log_format)
        
        # Set formatters
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add security filter
        security_filter = SecurityFilter()
        console_handler.addFilter(security_filter)
        file_handler.addFilter(security_filter)
        
        # Add handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        self.handlers = [console_handler, file_handler]
        
        # Configure structlog if enabled
        if self.structured_logging:
            self._configure_structlog()
    
    def _configure_structlog(self):
        """Configure structlog for structured logging."""
        processors = [
            TimeStamper(fmt="iso"),
            add_log_level,
            CallsiteParameterAdder(),
        ]
        
        if self.json_format:
            processors.append(JSONRenderer())
        else:
            processors.append(ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(
                name=name,
                level=self.log_level,
                format_json=self.json_format,
                include_caller=True
            )
        return self.loggers[name]
    
    def set_level(self, level: LogLevel):
        """Set log level for all loggers."""
        self.log_level = level
        
        # Update root logger
        logging.getLogger().setLevel(level.value)
        
        # Update all handlers
        for handler in self.handlers:
            handler.setLevel(level.value)
        
        # Update structured loggers
        for logger in self.loggers.values():
            logger.level = level
            logging.getLogger(logger.name).setLevel(level.value)
    
    def add_handler(self, handler: logging.Handler):
        """Add a new log handler."""
        handler.setLevel(self.log_level.value)
        if self.json_format:
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(self.log_format))
        
        handler.addFilter(SecurityFilter())
        logging.getLogger().addHandler(handler)
        self.handlers.append(handler)
    
    def remove_handler(self, handler: logging.Handler):
        """Remove a log handler."""
        if handler in self.handlers:
            logging.getLogger().removeHandler(handler)
            self.handlers.remove(handler)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            "level": self.log_level.value,
            "handlers": len(self.handlers),
            "loggers": len(self.loggers),
            "log_file": self.log_file,
            "json_format": self.json_format,
            "structured_logging": self.structured_logging
        }
        
        # File size information
        if os.path.exists(self.log_file):
            stats["log_file_size"] = os.path.getsize(self.log_file)
        
        return stats
    
    def rotate_logs(self):
        """Manually rotate log files."""
        for handler in self.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.doRollover()
    
    def clear_logs(self):
        """Clear all log files."""
        for handler in self.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                # Close and reopen the file
                handler.close()
                if os.path.exists(handler.baseFilename):
                    os.remove(handler.baseFilename)
                handler.stream = handler._open()


class JsonFormatter(logging.Formatter):
    """JSON formatter for log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message'}:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""
    
    def __init__(self):
        super().__init__()
        self._correlation_id = None
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        if self._correlation_id:
            record.correlation_id = self._correlation_id
        return True
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current context."""
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self):
        """Clear correlation ID."""
        self._correlation_id = None


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self, log_manager: LogManager):
        """
        Initialize audit logger.
        
        Args:
            log_manager: Log manager instance
        """
        self.logger = log_manager.get_logger("audit")
        
    def log_authentication(self, user_id: str, success: bool, ip_address: str, user_agent: str):
        """Log authentication event."""
        self.logger.info(
            "Authentication attempt",
            event_type="authentication",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_api_access(self, user_id: str, endpoint: str, method: str, status_code: int, ip_address: str):
        """Log API access event."""
        self.logger.info(
            "API access",
            event_type="api_access",
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            ip_address=ip_address
        )
    
    def log_data_access(self, user_id: str, resource_type: str, resource_id: str, action: str):
        """Log data access event."""
        self.logger.info(
            "Data access",
            event_type="data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action
        )
    
    def log_configuration_change(self, user_id: str, config_key: str, old_value: Any, new_value: Any):
        """Log configuration change event."""
        self.logger.info(
            "Configuration change",
            event_type="configuration_change",
            user_id=user_id,
            config_key=config_key,
            old_value=str(old_value),
            new_value=str(new_value)
        )
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security event."""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            user_id=user_id,
            **details
        )


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, log_manager: LogManager):
        """
        Initialize performance logger.
        
        Args:
            log_manager: Log manager instance
        """
        self.logger = log_manager.get_logger("performance")
    
    def log_request_performance(self, 
                              endpoint: str, 
                              method: str, 
                              duration: float, 
                              status_code: int,
                              request_size: int = 0,
                              response_size: int = 0):
        """Log request performance metrics."""
        self.logger.info(
            "Request performance",
            metric_type="request_performance",
            endpoint=endpoint,
            method=method,
            duration=duration,
            status_code=status_code,
            request_size=request_size,
            response_size=response_size
        )
    
    def log_database_performance(self, 
                               query_type: str, 
                               duration: float, 
                               rows_affected: int = 0):
        """Log database performance metrics."""
        self.logger.info(
            "Database performance",
            metric_type="database_performance",
            query_type=query_type,
            duration=duration,
            rows_affected=rows_affected
        )
    
    def log_agent_performance(self, 
                            agent_id: str, 
                            task_type: str, 
                            duration: float, 
                            success: bool):
        """Log agent performance metrics."""
        self.logger.info(
            "Agent performance",
            metric_type="agent_performance",
            agent_id=agent_id,
            task_type=task_type,
            duration=duration,
            success=success
        )


# Global log manager instance
_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get the global log manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return get_log_manager().get_logger(name)


def configure_logging(config: Dict[str, Any]):
    """Configure global logging settings."""
    global _log_manager
    _log_manager = LogManager(config)


# Convenience functions
def debug(message: str, **kwargs):
    """Log debug message to default logger."""
    get_logger("app").debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message to default logger."""
    get_logger("app").info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message to default logger."""
    get_logger("app").warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message to default logger."""
    get_logger("app").error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message to default logger."""
    get_logger("app").critical(message, **kwargs)


def exception(message: str, **kwargs):
    """Log exception to default logger."""
    get_logger("app").exception(message, **kwargs)