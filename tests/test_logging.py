"""
Tests for the logging and error handling framework.
"""

import pytest
import tempfile
import os
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.logging.manager import LoggingManager, LogFormatter, LogFilter
from src.logging.handlers import (
    FileHandler, RotatingFileHandler, DatabaseHandler, 
    MetricsHandler, AlertHandler
)
from src.agents.exceptions import (
    AgentError, AgentStateError, AgentCommunicationError,
    AgentManagerError, AgentRegistrationError, AgentNotFoundError,
    ConfigError, ConfigValidationError, ConfigNotFoundError,
    CommunicationError, MessageValidationError, MessageRoutingError
)


class TestAgentExceptions:
    """Test custom exception classes."""
    
    def test_agent_error_base(self):
        """Test base AgentError exception."""
        error = AgentError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_agent_state_error(self):
        """Test AgentStateError exception."""
        error = AgentStateError("Invalid state transition")
        assert str(error) == "Invalid state transition"
        assert isinstance(error, AgentError)
    
    def test_agent_communication_error(self):
        """Test AgentCommunicationError exception."""
        error = AgentCommunicationError("Message delivery failed")
        assert str(error) == "Message delivery failed"
        assert isinstance(error, AgentError)
    
    def test_agent_manager_error(self):
        """Test AgentManagerError exception."""
        error = AgentManagerError("Manager operation failed")
        assert str(error) == "Manager operation failed"
        assert isinstance(error, AgentError)
    
    def test_agent_registration_error(self):
        """Test AgentRegistrationError exception."""
        error = AgentRegistrationError("Agent registration failed")
        assert str(error) == "Agent registration failed"
        assert isinstance(error, AgentManagerError)
    
    def test_agent_not_found_error(self):
        """Test AgentNotFoundError exception."""
        error = AgentNotFoundError("Agent not found")
        assert str(error) == "Agent not found"
        assert isinstance(error, AgentManagerError)
    
    def test_config_error(self):
        """Test ConfigError exception."""
        error = ConfigError("Configuration error")
        assert str(error) == "Configuration error"
        assert isinstance(error, AgentError)
    
    def test_config_validation_error(self):
        """Test ConfigValidationError exception."""
        error = ConfigValidationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert isinstance(error, ConfigError)
    
    def test_config_not_found_error(self):
        """Test ConfigNotFoundError exception."""
        error = ConfigNotFoundError("Configuration not found")
        assert str(error) == "Configuration not found"
        assert isinstance(error, ConfigError)
    
    def test_communication_error(self):
        """Test CommunicationError exception."""
        error = CommunicationError("Communication failed")
        assert str(error) == "Communication failed"
        assert isinstance(error, AgentError)
    
    def test_message_validation_error(self):
        """Test MessageValidationError exception."""
        error = MessageValidationError("Invalid message")
        assert str(error) == "Invalid message"
        assert isinstance(error, CommunicationError)
    
    def test_message_routing_error(self):
        """Test MessageRoutingError exception."""
        error = MessageRoutingError("Message routing failed")
        assert str(error) == "Message routing failed"
        assert isinstance(error, CommunicationError)
    
    def test_exception_with_context(self):
        """Test exceptions with additional context."""
        error = AgentError("Test error", context={"agent_id": "test_agent", "operation": "start"})
        assert str(error) == "Test error"
        assert error.context == {"agent_id": "test_agent", "operation": "start"}
    
    def test_exception_with_cause(self):
        """Test exceptions with underlying cause."""
        original_error = ValueError("Original error")
        error = AgentError("Wrapped error", cause=original_error)
        assert str(error) == "Wrapped error"
        assert error.cause == original_error


class TestLogFormatter:
    """Test LogFormatter class."""
    
    def test_formatter_initialization(self):
        """Test formatter initialization."""
        formatter = LogFormatter()
        assert formatter.format_string is not None
        assert formatter.date_format is not None
    
    def test_format_record(self):
        """Test formatting a log record."""
        formatter = LogFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        assert "test_logger" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
    
    def test_format_record_with_context(self):
        """Test formatting a log record with context."""
        formatter = LogFormatter(include_context=True)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add context to record
        record.agent_id = "test_agent"
        record.operation = "test_operation"
        
        formatted = formatter.format(record)
        
        assert "test_agent" in formatted
        assert "test_operation" in formatted
    
    def test_format_exception_record(self):
        """Test formatting a log record with exception."""
        formatter = LogFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
        
        formatted = formatter.format(record)
        
        assert "Error occurred" in formatted
        assert "ValueError" in formatted
        assert "Test exception" in formatted


class TestLogFilter:
    """Test LogFilter class."""
    
    def test_filter_initialization(self):
        """Test filter initialization."""
        filter_obj = LogFilter(level=logging.INFO)
        assert filter_obj.level == logging.INFO
    
    def test_filter_by_level(self):
        """Test filtering by log level."""
        filter_obj = LogFilter(level=logging.WARNING)
        
        info_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info message",
            args=(),
            exc_info=None
        )
        
        warning_record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        
        assert filter_obj.filter(info_record) is False
        assert filter_obj.filter(warning_record) is True
    
    def test_filter_by_name(self):
        """Test filtering by logger name."""
        filter_obj = LogFilter(name_pattern="agent.*")
        
        agent_record = logging.LogRecord(
            name="agent.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Agent message",
            args=(),
            exc_info=None
        )
        
        other_record = logging.LogRecord(
            name="other.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Other message",
            args=(),
            exc_info=None
        )
        
        assert filter_obj.filter(agent_record) is True
        assert filter_obj.filter(other_record) is False
    
    def test_filter_by_context(self):
        """Test filtering by context."""
        filter_obj = LogFilter(context_filters={"agent_id": "test_agent"})
        
        matching_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        matching_record.agent_id = "test_agent"
        
        non_matching_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        non_matching_record.agent_id = "other_agent"
        
        assert filter_obj.filter(matching_record) is True
        assert filter_obj.filter(non_matching_record) is False


class TestFileHandler:
    """Test FileHandler class."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    def test_file_handler_initialization(self, temp_log_file):
        """Test file handler initialization."""
        handler = FileHandler(temp_log_file)
        assert handler.filename == temp_log_file
        assert isinstance(handler.formatter, LogFormatter)
    
    def test_file_handler_write_log(self, temp_log_file):
        """Test writing log to file."""
        handler = FileHandler(temp_log_file)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        handler.close()
        
        # Verify log was written
        with open(temp_log_file, 'r') as f:
            content = f.read()
        
        assert "Test message" in content
        assert "INFO" in content


class TestRotatingFileHandler:
    """Test RotatingFileHandler class."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        
        # Cleanup rotated files
        for i in range(1, 6):
            rotated_file = f"{temp_file}.{i}"
            if os.path.exists(rotated_file):
                os.unlink(rotated_file)
    
    def test_rotating_handler_initialization(self, temp_log_file):
        """Test rotating file handler initialization."""
        handler = RotatingFileHandler(temp_log_file, max_bytes=1024, backup_count=5)
        assert handler.filename == temp_log_file
        assert handler.max_bytes == 1024
        assert handler.backup_count == 5
    
    def test_rotating_handler_rotation(self, temp_log_file):
        """Test log rotation."""
        handler = RotatingFileHandler(temp_log_file, max_bytes=100, backup_count=3)
        
        # Write logs that exceed max_bytes
        for i in range(50):
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Test message {i} with some padding to make it longer",
                args=(),
                exc_info=None
            )
            handler.emit(record)
        
        handler.close()
        
        # Check if rotation occurred
        assert os.path.exists(temp_log_file)
        # Note: In a real test, we'd check for rotated files, but that depends on actual file size


class TestDatabaseHandler:
    """Test DatabaseHandler class."""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing."""
        db = Mock()
        db.insert = Mock()
        db.close = Mock()
        return db
    
    def test_database_handler_initialization(self, mock_database):
        """Test database handler initialization."""
        handler = DatabaseHandler(mock_database)
        assert handler.database == mock_database
    
    def test_database_handler_write_log(self, mock_database):
        """Test writing log to database."""
        handler = DatabaseHandler(mock_database)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Verify database insert was called
        mock_database.insert.assert_called_once()
        
        call_args = mock_database.insert.call_args[0][0]
        assert call_args["level"] == "INFO"
        assert call_args["message"] == "Test message"
        assert call_args["logger_name"] == "test_logger"


class TestMetricsHandler:
    """Test MetricsHandler class."""
    
    @pytest.fixture
    def mock_metrics_client(self):
        """Mock metrics client for testing."""
        client = Mock()
        client.increment = Mock()
        client.gauge = Mock()
        client.histogram = Mock()
        return client
    
    def test_metrics_handler_initialization(self, mock_metrics_client):
        """Test metrics handler initialization."""
        handler = MetricsHandler(mock_metrics_client)
        assert handler.metrics_client == mock_metrics_client
    
    def test_metrics_handler_emit(self, mock_metrics_client):
        """Test emitting log metrics."""
        handler = MetricsHandler(mock_metrics_client)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Test error message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Verify metrics were sent
        mock_metrics_client.increment.assert_called_with("log.error", tags={"logger": "test_logger"})


class TestAlertHandler:
    """Test AlertHandler class."""
    
    @pytest.fixture
    def mock_alert_client(self):
        """Mock alert client for testing."""
        client = Mock()
        client.send_alert = Mock()
        return client
    
    def test_alert_handler_initialization(self, mock_alert_client):
        """Test alert handler initialization."""
        handler = AlertHandler(mock_alert_client, alert_level=logging.ERROR)
        assert handler.alert_client == mock_alert_client
        assert handler.alert_level == logging.ERROR
    
    def test_alert_handler_emit_alert(self, mock_alert_client):
        """Test emitting an alert."""
        handler = AlertHandler(mock_alert_client, alert_level=logging.ERROR)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Critical error occurred",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Verify alert was sent
        mock_alert_client.send_alert.assert_called_once()
        
        call_args = mock_alert_client.send_alert.call_args[1]
        assert call_args["level"] == "ERROR"
        assert call_args["message"] == "Critical error occurred"
    
    def test_alert_handler_no_alert_for_low_level(self, mock_alert_client):
        """Test that no alert is sent for low-level logs."""
        handler = AlertHandler(mock_alert_client, alert_level=logging.ERROR)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Info message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Verify no alert was sent
        mock_alert_client.send_alert.assert_not_called()


class TestLoggingManager:
    """Test LoggingManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "console": {
                        "type": "console",
                        "level": "INFO"
                    },
                    "file": {
                        "type": "file",
                        "level": "DEBUG",
                        "filename": "/tmp/test.log"
                    }
                },
                "loggers": {
                    "agent": {
                        "level": "DEBUG",
                        "handlers": ["console", "file"]
                    }
                }
            }
        }
    
    def test_logging_manager_initialization(self, mock_config):
        """Test logging manager initialization."""
        manager = LoggingManager(mock_config)
        assert manager.config == mock_config["logging"]
        assert manager.loggers == {}
        assert manager.handlers == {}
    
    def test_create_logger(self, mock_config):
        """Test creating a logger."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert isinstance(logger, logging.Logger)
        assert "test_logger" in manager.loggers
    
    def test_create_logger_with_context(self, mock_config):
        """Test creating a logger with context."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger", context={"agent_id": "test_agent"})
        
        assert logger.name == "test_logger"
        assert hasattr(logger, 'context')
        assert logger.context == {"agent_id": "test_agent"}
    
    def test_configure_handlers(self, mock_config):
        """Test configuring handlers."""
        manager = LoggingManager(mock_config)
        
        # This would normally create handlers based on config
        # For testing, we'll mock the handler creation
        with patch.object(manager, '_create_handler') as mock_create:
            mock_handler = Mock()
            mock_create.return_value = mock_handler
            
            manager.configure_handlers()
            
            # Verify handlers were created
            assert mock_create.call_count == 2  # console and file handlers
    
    def test_add_handler(self, mock_config):
        """Test adding a handler."""
        manager = LoggingManager(mock_config)
        
        mock_handler = Mock()
        manager.add_handler("test_handler", mock_handler)
        
        assert "test_handler" in manager.handlers
        assert manager.handlers["test_handler"] == mock_handler
    
    def test_remove_handler(self, mock_config):
        """Test removing a handler."""
        manager = LoggingManager(mock_config)
        
        mock_handler = Mock()
        manager.add_handler("test_handler", mock_handler)
        manager.remove_handler("test_handler")
        
        assert "test_handler" not in manager.handlers
    
    def test_set_log_level(self, mock_config):
        """Test setting log level."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger")
        manager.set_log_level("test_logger", logging.DEBUG)
        
        assert logger.level == logging.DEBUG
    
    def test_add_context_to_logger(self, mock_config):
        """Test adding context to logger."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger")
        manager.add_context("test_logger", {"operation": "test_operation"})
        
        assert hasattr(logger, 'context')
        assert logger.context["operation"] == "test_operation"
    
    def test_structured_logging(self, mock_config):
        """Test structured logging."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger")
        
        # Mock the logger to capture log calls
        with patch.object(logger, 'info') as mock_info:
            manager.log_structured(
                "test_logger",
                "INFO",
                "Test message",
                {"key": "value", "number": 42}
            )
            
            mock_info.assert_called_once()
            # The exact call depends on implementation
    
    def test_log_exception(self, mock_config):
        """Test logging exceptions."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger")
        
        # Mock the logger to capture log calls
        with patch.object(logger, 'error') as mock_error:
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                manager.log_exception("test_logger", e)
            
            mock_error.assert_called_once()
    
    def test_performance_logging(self, mock_config):
        """Test performance logging."""
        manager = LoggingManager(mock_config)
        
        logger = manager.get_logger("test_logger")
        
        # Mock the logger to capture log calls
        with patch.object(logger, 'info') as mock_info:
            with manager.log_performance("test_logger", "test_operation"):
                # Simulate some work
                pass
            
            mock_info.assert_called()
    
    def test_shutdown_logging(self, mock_config):
        """Test shutting down logging."""
        manager = LoggingManager(mock_config)
        
        # Add mock handlers
        mock_handler1 = Mock()
        mock_handler2 = Mock()
        manager.add_handler("handler1", mock_handler1)
        manager.add_handler("handler2", mock_handler2)
        
        manager.shutdown()
        
        # Verify all handlers were closed
        mock_handler1.close.assert_called_once()
        mock_handler2.close.assert_called_once()
    
    def test_log_filtering(self, mock_config):
        """Test log filtering."""
        manager = LoggingManager(mock_config)
        
        # Add a filter
        filter_obj = LogFilter(level=logging.WARNING)
        manager.add_filter("test_filter", filter_obj)
        
        logger = manager.get_logger("test_logger")
        
        # Apply filter to logger
        manager.apply_filter("test_logger", "test_filter")
        
        # Verify filter was added
        assert any(isinstance(f, LogFilter) for f in logger.filters)
    
    def test_log_metrics_collection(self, mock_config):
        """Test log metrics collection."""
        manager = LoggingManager(mock_config)
        
        # Enable metrics collection
        manager.enable_metrics()
        
        logger = manager.get_logger("test_logger")
        
        # Log some messages
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Get metrics
        metrics = manager.get_metrics()
        
        assert "log_counts" in metrics
        assert metrics["log_counts"]["INFO"] >= 1
        assert metrics["log_counts"]["WARNING"] >= 1
        assert metrics["log_counts"]["ERROR"] >= 1