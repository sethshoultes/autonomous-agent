"""
Logging management system for the autonomous agent framework.
"""

from contextlib import contextmanager, suppress
import logging
import logging.handlers
import re
import threading
import time
from typing import Any, ClassVar, Dict, Optional, Union


class LogFormatter(logging.Formatter):
    """
    Enhanced log formatter with support for colors and context.

    Provides structured, readable log output with optional color coding
    and contextual information for debugging and monitoring.
    """

    # Color codes for different log levels
    COLORS: ClassVar[Dict[str, str]] = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def __init__(
        self,
        format_string: Optional[str] = None,
        date_format: Optional[str] = None,
        use_colors: bool = False,
        include_context: bool = False,
        include_timestamp: bool = True
    ):
        """
        Initialize the log formatter.

        Args:
            format_string: Custom format string
            date_format: Date format string
            use_colors: Whether to use colored output
            include_context: Whether to include contextual information
            include_timestamp: Whether to include timestamp
        """
        if format_string is None:
            if include_timestamp:
                format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            else:
                format_string = "%(name)s - %(levelname)s - %(message)s"

            if include_context:
                format_string += " [%(filename)s:%(lineno)d]"

        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"

        super().__init__(format_string, date_format)

        self.use_colors = use_colors
        self.include_context = include_context
        self.format_string = format_string

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record.

        Args:
            record: Log record to format

        Returns:
            Formatted log message
        """
        # Add contextual information if available
        if self.include_context:
            if hasattr(record, 'agent_id'):
                record.msg = f"[Agent: {record.agent_id}] {record.msg}"

            if hasattr(record, 'operation'):
                record.msg = f"[Op: {record.operation}] {record.msg}"

        # Format the record
        formatted = super().format(record)

        # Add colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"

        return formatted

    def formatException(self, ei: Any) -> str:
        """
        Format exception information with enhanced details.

        Args:
            ei: Exception information tuple

        Returns:
            Formatted exception string
        """
        result = super().formatException(ei)

        # Add additional context if available
        if ei and ei[1]:
            exception = ei[1]
            if hasattr(exception, 'context') and exception.context:
                result += f"\nContext: {exception.context}"

            if hasattr(exception, 'cause') and exception.cause:
                result += f"\nCaused by: {exception.cause}"

        return result


class LogFilter(logging.Filter):
    """
    Enhanced log filter with multiple filtering criteria.

    Provides filtering based on log level, logger name patterns,
    and contextual information.
    """

    def __init__(
        self,
        name: str = '',
        level: Optional[int] = None,
        name_pattern: Optional[str] = None,
        context_filters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the log filter.

        Args:
            name: Logger name to filter (exact match)
            level: Minimum log level to accept
            name_pattern: Regex pattern for logger names
            context_filters: Dictionary of context key-value pairs to filter
        """
        super().__init__(name)
        self.level = level
        self.name_pattern = re.compile(name_pattern) if name_pattern else None
        self.context_filters = context_filters or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter a log record.

        Args:
            record: Log record to filter

        Returns:
            True if record should be processed, False otherwise
        """
        # Check base filter
        if not super().filter(record):
            return False

        # Check level filter
        if self.level is not None and record.levelno < self.level:
            return False

        # Check name pattern filter
        if self.name_pattern and not self.name_pattern.match(record.name):
            return False

        # Check context filters
        for key, expected_value in self.context_filters.items():
            if not hasattr(record, key) or getattr(record, key) != expected_value:
                return False

        return True


class LoggingManager:
    """
    Centralized logging management system.

    Provides comprehensive logging configuration, handler management,
    and advanced features like structured logging and metrics collection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logging manager.

        Args:
            config: Logging configuration dictionary
        """
        self.config = config.get("logging", {})
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.filters: Dict[str, LogFilter] = {}
        self.metrics_enabled = False
        self.metrics: Dict[str, int] = {}
        self._lock = threading.Lock()

        # Configure root logger
        self._configure_root_logger()

    def _configure_root_logger(self) -> None:
        """Configure the root logger with basic settings."""
        root_logger = logging.getLogger()

        # Set root level
        level = self.config.get("level", "INFO")
        root_logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add default console handler if no handlers configured
        if not self.config.get("handlers"):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(LogFormatter())
            root_logger.addHandler(console_handler)

    def get_logger(
        self,
        name: str,
        level: Union[str, int, None] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.

        Args:
            name: Logger name
            level: Optional log level override
            context: Optional context dictionary to attach to logger

        Returns:
            Configured logger instance
        """
        with self._lock:
            if name not in self.loggers:
                logger = logging.getLogger(name)

                # Set level if specified
                if level:
                    if isinstance(level, str):
                        level = getattr(logging, level.upper())
                    logger.setLevel(level)

                # Add context if provided
                if context:
                    logger.context = context

                # Configure handlers for this logger
                self._configure_logger_handlers(logger, name)

                self.loggers[name] = logger

            return self.loggers[name]

    def _configure_logger_handlers(self, logger: logging.Logger, logger_name: str) -> None:
        """Configure handlers for a specific logger."""
        logger_config = self.config.get("loggers", {}).get(logger_name, {})
        handler_names = logger_config.get("handlers", [])

        for handler_name in handler_names:
            if handler_name in self.handlers:
                logger.addHandler(self.handlers[handler_name])

    def configure_handlers(self) -> None:
        """Configure all handlers from the configuration."""
        handlers_config = self.config.get("handlers", {})

        for handler_name, handler_config in handlers_config.items():
            handler = self._create_handler(handler_name, handler_config)
            if handler:
                self.handlers[handler_name] = handler

    def _create_handler(self, name: str, config: Dict[str, Any]) -> Optional[logging.Handler]:
        """
        Create a handler from configuration.

        Args:
            name: Handler name
            config: Handler configuration

        Returns:
            Configured handler instance or None if creation fails
        """
        handler_type = config.get("type", "console")

        try:
            if handler_type == "console":
                handler = logging.StreamHandler()

            elif handler_type == "file":
                filename = config.get("filename", f"logs/{name}.log")
                handler = logging.FileHandler(filename)

            elif handler_type == "rotating_file":
                filename = config.get("filename", f"logs/{name}.log")
                max_bytes = config.get("max_bytes", 10 * 1024 * 1024)
                backup_count = config.get("backup_count", 5)
                handler = logging.handlers.RotatingFileHandler(
                    filename, maxBytes=max_bytes, backupCount=backup_count
                )

            elif handler_type == "syslog":
                address = config.get("address", ("localhost", 514))
                facility = config.get("facility", "user")
                handler = logging.handlers.SyslogHandler(address, facility)

            else:
                logging.warning(f"Unknown handler type: {handler_type}")
                return None

            # Configure handler
            level = config.get("level", "INFO")
            handler.setLevel(getattr(logging, level.upper()))

            # Set formatter
            formatter_config = config.get("formatter", {})
            formatter = LogFormatter(**formatter_config)
            handler.setFormatter(formatter)

            return handler

        except Exception as e:
            logging.error(f"Failed to create handler {name}: {e}")
            return None

    def add_handler(self, name: str, handler: logging.Handler) -> None:
        """
        Add a handler to the manager.

        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler

    def remove_handler(self, name: str) -> None:
        """
        Remove a handler from the manager.

        Args:
            name: Handler name to remove
        """
        if name in self.handlers:
            handler = self.handlers.pop(name)

            # Remove from all loggers
            for logger in self.loggers.values():
                if handler in logger.handlers:
                    logger.removeHandler(handler)

            # Close the handler
            handler.close()

    def add_filter(self, name: str, filter_obj: LogFilter) -> None:
        """
        Add a filter to the manager.

        Args:
            name: Filter name
            filter_obj: Filter instance
        """
        self.filters[name] = filter_obj

    def apply_filter(self, logger_name: str, filter_name: str) -> None:
        """
        Apply a filter to a logger.

        Args:
            logger_name: Logger name
            filter_name: Filter name
        """
        if logger_name in self.loggers and filter_name in self.filters:
            logger = self.loggers[logger_name]
            filter_obj = self.filters[filter_name]
            logger.addFilter(filter_obj)

    def set_log_level(self, logger_name: str, level: Union[str, int]) -> None:
        """
        Set the log level for a specific logger.

        Args:
            logger_name: Logger name
            level: Log level (string or integer)
        """
        if logger_name in self.loggers:
            logger = self.loggers[logger_name]

            if isinstance(level, str):
                level = getattr(logging, level.upper())

            logger.setLevel(level)

    def add_context(self, logger_name: str, context: Dict[str, Any]) -> None:
        """
        Add context to a logger.

        Args:
            logger_name: Logger name
            context: Context dictionary
        """
        if logger_name in self.loggers:
            logger = self.loggers[logger_name]

            if not hasattr(logger, 'context'):
                logger.context = {}

            logger.context.update(context)

    def log_structured(
        self,
        logger_name: str,
        level: str,
        message: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Log a structured message with additional data.

        Args:
            logger_name: Logger name
            level: Log level
            message: Log message
            data: Additional structured data
        """
        logger = self.get_logger(logger_name)

        # Create a custom log record with structured data
        log_level = getattr(logging, level.upper())

        # Add structured data to the record
        extra = {f"data_{k}": v for k, v in data.items()}

        logger.log(log_level, message, extra=extra)

    def log_exception(self, logger_name: str, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an exception with full context.

        Args:
            logger_name: Logger name
            exception: Exception to log
            context: Optional additional context
        """
        logger = self.get_logger(logger_name)

        extra = {}
        if context:
            extra.update({f"ctx_{k}": v for k, v in context.items()})

        logger.exception(f"Exception occurred: {exception}", extra=extra)

    @contextmanager
    def log_performance(self, logger_name: str, operation: str) -> Any:
        """
        Context manager for logging operation performance.

        Args:
            logger_name: Logger name
            operation: Operation name
        """
        logger = self.get_logger(logger_name)
        start_time = time.time()

        logger.info(f"Starting operation: {operation}")

        try:
            yield
            duration = time.time() - start_time
            logger.info(f"Completed operation: {operation} in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed operation: {operation} after {duration:.3f}s - {e}")
            raise

    def enable_metrics(self) -> None:
        """Enable metrics collection from log events."""
        self.metrics_enabled = True

        # Add metrics handler to all loggers
        for logger in self.loggers.values():
            metrics_handler = self._create_metrics_handler()
            logger.addHandler(metrics_handler)

    def disable_metrics(self) -> None:
        """Disable metrics collection."""
        self.metrics_enabled = False

    def _create_metrics_handler(self) -> logging.Handler:
        """Create a handler for metrics collection."""
        class MetricsCollector(logging.Handler):
            def __init__(self, manager: Any) -> None:
                super().__init__()
                self.manager = manager

            def emit(self, record: Any) -> None:
                with self.manager._lock:
                    metric_key = f"log_counts.{record.levelname}"
                    self.manager.metrics[metric_key] = self.manager.metrics.get(metric_key, 0) + 1

        return MetricsCollector(self)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collected logging metrics.

        Returns:
            Dictionary containing logging metrics
        """
        with self._lock:
            return {
                "log_counts": {
                    level: self.metrics.get(f"log_counts.{level}", 0)
                    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                },
                "metrics_enabled": self.metrics_enabled,
                "active_loggers": len(self.loggers),
                "active_handlers": len(self.handlers)
            }

    def shutdown(self) -> None:
        """Shutdown the logging manager and close all handlers."""
        with self._lock:
            # Close all handlers
            for handler in self.handlers.values():
                with suppress(Exception):
                    handler.close()

            # Clear collections
            self.handlers.clear()
            self.loggers.clear()
            self.filters.clear()

            # Shutdown logging
            logging.shutdown()
