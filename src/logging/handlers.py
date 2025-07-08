"""
Custom logging handlers for the autonomous agent system.
"""

import logging
from logging.handlers import RotatingFileHandler as BaseRotatingFileHandler
import os
from pathlib import Path
import time
from typing import Any, Optional, Union

from .manager import LogFormatter


class FileHandler(logging.FileHandler):
    """
    Enhanced file handler with automatic directory creation.

    Extends the standard FileHandler with additional features
    for the autonomous agent system.
    """

    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, delay: bool = False):
        """
        Initialize the file handler.

        Args:
            filename: Path to the log file
            mode: File open mode
            encoding: File encoding
            delay: Whether to delay file opening
        """
        # Ensure directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(filename, mode, encoding, delay)

        # Set default formatter
        self.setFormatter(LogFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record with enhanced error handling.

        Args:
            record: Log record to emit
        """
        try:
            super().emit(record)
        except Exception:
            # If we can't write to the file, try to handle gracefully
            self.handleError(record)


class RotatingFileHandler(BaseRotatingFileHandler):
    """
    Enhanced rotating file handler with automatic directory creation.

    Extends the standard RotatingFileHandler with additional features
    for the autonomous agent system.
    """

    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5,
        encoding: Optional[str] = None,
        delay: bool = False
    ):
        """
        Initialize the rotating file handler.

        Args:
            filename: Path to the log file
            mode: File open mode
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            encoding: File encoding
            delay: Whether to delay file opening
        """
        # Ensure directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(filename, mode, max_bytes, backup_count, encoding, delay)

        # Set default formatter
        self.setFormatter(LogFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record with enhanced error handling.

        Args:
            record: Log record to emit
        """
        try:
            super().emit(record)
        except Exception:
            # If we can't write to the file, try to handle gracefully
            self.handleError(record)


class DatabaseHandler(logging.Handler):
    """
    Database handler for storing log records in a database.

    Provides structured logging to a database backend with
    automatic table creation and error handling.
    """

    def __init__(self, database: Any, table_name: str = "logs"):
        """
        Initialize the database handler.

        Args:
            database: Database connection or interface
            table_name: Name of the log table
        """
        super().__init__()
        self.database = database
        self.table_name = table_name

        # Set default formatter
        self.setFormatter(LogFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the database.

        Args:
            record: Log record to emit
        """
        try:
            # Convert log record to dictionary
            log_data = {
                "timestamp": time.time(),
                "level": record.levelname,
                "logger_name": record.name,
                "message": record.getMessage(),
                "module": getattr(record, 'module', ''),
                "function": getattr(record, 'funcName', ''),
                "line_number": getattr(record, 'lineno', 0),
                "thread_id": getattr(record, 'thread', 0),
                "process_id": getattr(record, 'process', 0),
            }

            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.format(record)

            # Add custom fields from record
            for key, value in record.__dict__.items():
                if (key not in log_data and not key.startswith('_')
                    and isinstance(value, (str, int, float, bool, type(None)))):
                    log_data[f"custom_{key}"] = value

            # Insert into database
            self.database.insert(log_data)

        except Exception:
            # If we can't write to the database, handle gracefully
            self.handleError(record)

    def close(self) -> None:
        """Close the database handler."""
        try:
            if hasattr(self.database, 'close'):
                self.database.close()
        except Exception:
            pass
        super().close()


class MetricsHandler(logging.Handler):
    """
    Metrics handler for collecting log-based metrics.

    Sends log events to a metrics collection system for
    monitoring and alerting purposes.
    """

    def __init__(self, metrics_client: Any):
        """
        Initialize the metrics handler.

        Args:
            metrics_client: Metrics client interface
        """
        super().__init__()
        self.metrics_client = metrics_client

        # Only process warning and above by default
        self.setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record as metrics.

        Args:
            record: Log record to emit
        """
        try:
            # Generate metrics based on log level
            metric_name = f"log.{record.levelname.lower()}"
            tags = {
                "logger": record.name,
                "level": record.levelname
            }

            # Add custom tags from record
            if hasattr(record, 'agent_id'):
                tags["agent_id"] = record.agent_id

            if hasattr(record, 'operation'):
                tags["operation"] = record.operation

            # Increment counter for this log event
            self.metrics_client.increment(metric_name, tags=tags)

            # For errors, also record additional metrics
            if record.levelno >= logging.ERROR:
                self.metrics_client.increment("log.error.total", tags=tags)

                if record.exc_info:
                    exception_type = record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"
                    error_tags = {**tags, "exception_type": exception_type}
                    self.metrics_client.increment("log.exception", tags=error_tags)

        except Exception:
            # If metrics client fails, don't break logging
            self.handleError(record)


class AlertHandler(logging.Handler):
    """
    Alert handler for sending alerts based on log events.

    Sends alerts to external systems when critical log events occur.
    """

    def __init__(self, alert_client: Any, alert_level: int = logging.ERROR):
        """
        Initialize the alert handler.

        Args:
            alert_client: Alert client interface
            alert_level: Minimum log level to trigger alerts
        """
        super().__init__()
        self.alert_client = alert_client
        self.alert_level = alert_level

        # Set the level
        self.setLevel(alert_level)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record as an alert.

        Args:
            record: Log record to emit
        """
        try:
            # Only send alerts for specified level and above
            if record.levelno < self.alert_level:
                return

            # Prepare alert data
            alert_data = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "timestamp": time.time(),
                "source": "autonomous_agent_system"
            }

            # Add context from record
            if hasattr(record, 'agent_id'):
                alert_data["agent_id"] = record.agent_id

            if hasattr(record, 'operation'):
                alert_data["operation"] = record.operation

            # Add exception info if present
            if record.exc_info:
                alert_data["exception"] = self.format(record)
                alert_data["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Unknown"

            # Send alert
            self.alert_client.send_alert(**alert_data)

        except Exception:
            # If alert client fails, don't break logging
            self.handleError(record)


class StreamHandler(logging.StreamHandler):
    """
    Enhanced stream handler with better formatting.

    Provides improved console output with colors and structured formatting.
    """

    def __init__(self, stream: Any = None, use_colors: bool = True) -> None:
        """
        Initialize the stream handler.

        Args:
            stream: Output stream (defaults to sys.stderr)
            use_colors: Whether to use colored output
        """
        super().__init__(stream)
        self.use_colors = use_colors and self._supports_color()

        # Set default formatter
        if self.use_colors:
            self.setFormatter(LogFormatter(use_colors=True))
        else:
            self.setFormatter(LogFormatter())

    def _supports_color(self) -> bool:
        """Check if the terminal supports color output."""
        try:
            import sys
            return (
                hasattr(sys.stderr, "isatty") and
                sys.stderr.isatty() and
                os.environ.get("TERM") != "dumb"
            )
        except Exception:
            return False

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record with enhanced formatting.

        Args:
            record: Log record to emit
        """
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


class SyslogHandler(logging.handlers.SysLogHandler):
    """
    Enhanced syslog handler for system logging.

    Provides structured logging to system syslog with proper facility
    and priority mapping.
    """

    def __init__(self, address: Any = ('localhost', 514), facility: str = 'user') -> None:
        """
        Initialize the syslog handler.

        Args:
            address: Syslog server address
            facility: Syslog facility
        """
        super().__init__(address, facility)

        # Set default formatter
        self.setFormatter(LogFormatter(include_timestamp=False))  # Syslog adds its own timestamp

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to syslog.

        Args:
            record: Log record to emit
        """
        try:
            # Add system information to record
            record.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            record.app_name = 'autonomous_agent'

            super().emit(record)
        except Exception:
            self.handleError(record)
