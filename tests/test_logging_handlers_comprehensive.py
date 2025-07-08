"""
Comprehensive tests for logging handlers module.
"""

import pytest
import tempfile
import os
import json
import logging
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from src.logging.handlers import (
    FileHandler, RotatingFileHandler, DatabaseHandler, 
    MetricsHandler, AlertHandler, StreamHandler, SyslogHandler
)
from src.logging.manager import LogFormatter


class TestFileHandlerComprehensive:
    """Comprehensive tests for FileHandler class."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_file_handler_directory_creation(self, temp_dir):
        """Test automatic directory creation."""
        nested_path = os.path.join(temp_dir, "logs", "app", "test.log")
        handler = FileHandler(nested_path)
        
        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_path))
        handler.close()
    
    def test_file_handler_different_modes(self, temp_log_file):
        """Test file handler with different modes."""
        # Test append mode
        handler1 = FileHandler(temp_log_file, mode='a')
        record = logging.LogRecord("test", logging.INFO, "test.py", 10, "Message 1", (), None)
        handler1.emit(record)
        handler1.close()
        
        # Test write mode (should overwrite)
        handler2 = FileHandler(temp_log_file, mode='w')
        record = logging.LogRecord("test", logging.INFO, "test.py", 10, "Message 2", (), None)
        handler2.emit(record)
        handler2.close()
        
        with open(temp_log_file, 'r') as f:
            content = f.read()
        
        # Should only contain Message 2 (write mode overwrites)
        assert "Message 2" in content
        assert "Message 1" not in content
    
    def test_file_handler_encoding(self, temp_dir):
        """Test file handler with different encodings."""
        log_file = os.path.join(temp_dir, "test_encoding.log")
        handler = FileHandler(log_file, encoding='utf-8')
        
        # Test with unicode content
        record = logging.LogRecord("test", logging.INFO, "test.py", 10, "Test 中文 message", (), None)
        handler.emit(record)
        handler.close()
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "中文" in content
    
    def test_file_handler_emit_error_handling(self, temp_log_file):
        """Test error handling in emit method."""
        handler = FileHandler(temp_log_file)
        
        # Close the file stream to force an error
        handler.stream.close()
        
        record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
        
        # Should not raise an exception
        with patch.object(handler, 'handleError') as mock_handle_error:
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)
    
    def test_file_handler_formatter_assignment(self, temp_log_file):
        """Test that LogFormatter is automatically assigned."""
        handler = FileHandler(temp_log_file)
        assert isinstance(handler.formatter, LogFormatter)
        handler.close()
    
    def test_file_handler_delay_parameter(self, temp_log_file):
        """Test file handler with delay parameter."""
        # Remove the temp file first
        os.unlink(temp_log_file)
        
        handler = FileHandler(temp_log_file, delay=True)
        
        # File should not exist yet (delay=True)
        assert not os.path.exists(temp_log_file)
        
        # After emitting a record, file should be created
        record = logging.LogRecord("test", logging.INFO, "test.py", 10, "Test message", (), None)
        handler.emit(record)
        
        assert os.path.exists(temp_log_file)
        handler.close()


class TestRotatingFileHandlerComprehensive:
    """Comprehensive tests for RotatingFileHandler class."""
    
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
        for i in range(1, 10):
            rotated_file = f"{temp_file}.{i}"
            if os.path.exists(rotated_file):
                os.unlink(rotated_file)
    
    def test_rotating_handler_parameters(self, temp_log_file):
        """Test rotating handler initialization with various parameters."""
        handler = RotatingFileHandler(
            temp_log_file, 
            mode='a',
            max_bytes=1024,
            backup_count=3,
            encoding='utf-8',
            delay=True
        )
        
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3
        assert isinstance(handler.formatter, LogFormatter)
        handler.close()
    
    def test_rotating_handler_directory_creation(self, temp_log_file):
        """Test automatic directory creation for rotating handler."""
        nested_path = temp_log_file.replace('.log', '/nested/test.log')
        handler = RotatingFileHandler(nested_path, max_bytes=100, backup_count=2)
        
        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_path))
        handler.close()
    
    def test_rotating_handler_emit_error_handling(self, temp_log_file):
        """Test error handling in rotating handler emit method."""
        handler = RotatingFileHandler(temp_log_file, max_bytes=100, backup_count=2)
        
        # Close the file stream to force an error
        handler.stream.close()
        
        record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
        
        # Should not raise an exception
        with patch.object(handler, 'handleError') as mock_handle_error:
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)
    
    def test_rotating_handler_file_rotation_simulation(self, temp_log_file):
        """Test file rotation behavior simulation."""
        # Use a very small max_bytes to trigger rotation
        handler = RotatingFileHandler(temp_log_file, max_bytes=50, backup_count=2)
        
        # Write multiple large messages
        for i in range(5):
            long_message = f"This is a long test message number {i} " * 10
            record = logging.LogRecord("test", logging.INFO, "test.py", 10, long_message, (), None)
            handler.emit(record)
        
        handler.close()
        
        # Original file should exist
        assert os.path.exists(temp_log_file)
        
        # Note: Actual rotation depends on the base class implementation
        # We're mainly testing that it doesn't crash


class TestDatabaseHandlerComprehensive:
    """Comprehensive tests for DatabaseHandler class."""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing."""
        db = Mock()
        db.insert = Mock()
        db.close = Mock()
        return db
    
    def test_database_handler_initialization(self, mock_database):
        """Test database handler initialization with custom table name."""
        handler = DatabaseHandler(mock_database, table_name="custom_logs")
        
        assert handler.database == mock_database
        assert handler.table_name == "custom_logs"
        assert isinstance(handler.formatter, LogFormatter)
    
    def test_database_handler_emit_basic_record(self, mock_database):
        """Test emitting a basic log record."""
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
        assert call_args["line_number"] == 10
        assert "timestamp" in call_args
    
    def test_database_handler_emit_with_exception(self, mock_database):
        """Test emitting a log record with exception info."""
        handler = DatabaseHandler(mock_database)
        
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
        
        handler.emit(record)
        
        call_args = mock_database.insert.call_args[0][0]
        assert call_args["level"] == "ERROR"
        assert "exception" in call_args
        assert "ValueError" in call_args["exception"]
    
    def test_database_handler_emit_with_custom_fields(self, mock_database):
        """Test emitting a log record with custom fields."""
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
        
        # Add custom fields
        record.agent_id = "test_agent"
        record.operation = "test_operation"
        record.custom_data = {"key": "value"}  # This should be filtered out (not a simple type)
        record._private = "private"  # This should be filtered out (starts with _)
        
        handler.emit(record)
        
        call_args = mock_database.insert.call_args[0][0]
        assert call_args["custom_agent_id"] == "test_agent"
        assert call_args["custom_operation"] == "test_operation"
        assert "custom_custom_data" not in call_args  # Complex type filtered out
        assert "custom__private" not in call_args  # Private field filtered out
    
    def test_database_handler_emit_error_handling(self, mock_database):
        """Test error handling when database insert fails."""
        mock_database.insert.side_effect = Exception("Database error")
        
        handler = DatabaseHandler(mock_database)
        
        record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
        
        # Should not raise an exception
        with patch.object(handler, 'handleError') as mock_handle_error:
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)
    
    def test_database_handler_close(self, mock_database):
        """Test database handler close method."""
        handler = DatabaseHandler(mock_database)
        handler.close()
        
        mock_database.close.assert_called_once()
    
    def test_database_handler_close_without_close_method(self):
        """Test database handler close with database that doesn't have close method."""
        mock_database = Mock()
        del mock_database.close  # Remove close method
        
        handler = DatabaseHandler(mock_database)
        
        # Should not raise an exception
        handler.close()


class TestMetricsHandlerComprehensive:
    """Comprehensive tests for MetricsHandler class."""
    
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
        assert handler.level == logging.WARNING  # Default level
    
    def test_metrics_handler_emit_warning(self, mock_metrics_client):
        """Test emitting a warning level log."""
        handler = MetricsHandler(mock_metrics_client)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Verify metrics were sent
        mock_metrics_client.increment.assert_any_call(
            "log.warning", 
            tags={"logger": "test_logger", "level": "WARNING"}
        )
    
    def test_metrics_handler_emit_error_with_context(self, mock_metrics_client):
        """Test emitting an error with custom context."""
        handler = MetricsHandler(mock_metrics_client)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        # Add custom context
        record.agent_id = "test_agent"
        record.operation = "test_operation"
        
        handler.emit(record)
        
        expected_tags = {
            "logger": "test_logger", 
            "level": "ERROR",
            "agent_id": "test_agent",
            "operation": "test_operation"
        }
        
        # Verify metrics were sent
        mock_metrics_client.increment.assert_any_call("log.error", tags=expected_tags)
        mock_metrics_client.increment.assert_any_call("log.error.total", tags=expected_tags)
    
    def test_metrics_handler_emit_with_exception(self, mock_metrics_client):
        """Test emitting a log with exception info."""
        handler = MetricsHandler(mock_metrics_client)
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error with exception",
                args=(),
                exc_info=True
            )
        
        handler.emit(record)
        
        # Verify exception metrics were sent
        expected_tags = {
            "logger": "test_logger", 
            "level": "ERROR",
            "exception_type": "ValueError"
        }
        
        mock_metrics_client.increment.assert_any_call("log.exception", tags=expected_tags)
    
    def test_metrics_handler_below_level_threshold(self, mock_metrics_client):
        """Test that logs below threshold are not processed."""
        handler = MetricsHandler(mock_metrics_client)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,  # Below WARNING threshold
            pathname="test.py",
            lineno=10,
            msg="Info message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # No metrics should be sent for INFO level
        mock_metrics_client.increment.assert_not_called()
    
    def test_metrics_handler_emit_error_handling(self, mock_metrics_client):
        """Test error handling when metrics client fails."""
        mock_metrics_client.increment.side_effect = Exception("Metrics error")
        
        handler = MetricsHandler(mock_metrics_client)
        
        record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
        
        # Should not raise an exception
        with patch.object(handler, 'handleError') as mock_handle_error:
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)


class TestAlertHandlerComprehensive:
    """Comprehensive tests for AlertHandler class."""
    
    @pytest.fixture
    def mock_alert_client(self):
        """Mock alert client for testing."""
        client = Mock()
        client.send_alert = Mock()
        return client
    
    def test_alert_handler_initialization_default_level(self, mock_alert_client):
        """Test alert handler initialization with default level."""
        handler = AlertHandler(mock_alert_client)
        
        assert handler.alert_client == mock_alert_client
        assert handler.alert_level == logging.ERROR
        assert handler.level == logging.ERROR
    
    def test_alert_handler_initialization_custom_level(self, mock_alert_client):
        """Test alert handler initialization with custom level."""
        handler = AlertHandler(mock_alert_client, alert_level=logging.CRITICAL)
        
        assert handler.alert_level == logging.CRITICAL
        assert handler.level == logging.CRITICAL
    
    def test_alert_handler_emit_error_alert(self, mock_alert_client):
        """Test emitting an error level alert."""
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
        
        call_kwargs = mock_alert_client.send_alert.call_args[1]
        assert call_kwargs["level"] == "ERROR"
        assert call_kwargs["message"] == "Critical error occurred"
        assert call_kwargs["logger"] == "test_logger"
        assert call_kwargs["source"] == "autonomous_agent_system"
        assert "timestamp" in call_kwargs
    
    def test_alert_handler_emit_with_context(self, mock_alert_client):
        """Test emitting an alert with custom context."""
        handler = AlertHandler(mock_alert_client)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error with context",
            args=(),
            exc_info=None
        )
        
        # Add custom context
        record.agent_id = "test_agent"
        record.operation = "test_operation"
        
        handler.emit(record)
        
        call_kwargs = mock_alert_client.send_alert.call_args[1]
        assert call_kwargs["agent_id"] == "test_agent"
        assert call_kwargs["operation"] == "test_operation"
    
    def test_alert_handler_emit_with_exception(self, mock_alert_client):
        """Test emitting an alert with exception info."""
        handler = AlertHandler(mock_alert_client)
        
        try:
            raise RuntimeError("Test runtime error")
        except RuntimeError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error with exception",
                args=(),
                exc_info=True
            )
        
        handler.emit(record)
        
        call_kwargs = mock_alert_client.send_alert.call_args[1]
        assert "exception" in call_kwargs
        assert "exception_type" in call_kwargs
        assert call_kwargs["exception_type"] == "RuntimeError"
        assert "RuntimeError" in call_kwargs["exception"]
    
    def test_alert_handler_below_threshold(self, mock_alert_client):
        """Test that logs below alert threshold don't trigger alerts."""
        handler = AlertHandler(mock_alert_client, alert_level=logging.ERROR)
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,  # Below ERROR threshold
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # No alert should be sent
        mock_alert_client.send_alert.assert_not_called()
    
    def test_alert_handler_emit_error_handling(self, mock_alert_client):
        """Test error handling when alert client fails."""
        mock_alert_client.send_alert.side_effect = Exception("Alert system error")
        
        handler = AlertHandler(mock_alert_client)
        
        record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
        
        # Should not raise an exception
        with patch.object(handler, 'handleError') as mock_handle_error:
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)


class TestStreamHandlerComprehensive:
    """Comprehensive tests for StreamHandler class."""
    
    def test_stream_handler_initialization_default(self):
        """Test stream handler initialization with defaults."""
        with patch('src.logging.handlers.StreamHandler._supports_color', return_value=True):
            handler = StreamHandler()
            
            assert handler.use_colors is True
            assert isinstance(handler.formatter, LogFormatter)
    
    def test_stream_handler_initialization_no_colors(self):
        """Test stream handler initialization without colors."""
        handler = StreamHandler(use_colors=False)
        
        assert handler.use_colors is False
        assert isinstance(handler.formatter, LogFormatter)
    
    def test_stream_handler_custom_stream(self):
        """Test stream handler with custom stream."""
        custom_stream = StringIO()
        handler = StreamHandler(stream=custom_stream, use_colors=False)
        
        assert handler.stream == custom_stream
    
    def test_stream_handler_supports_color_detection(self):
        """Test color support detection."""
        handler = StreamHandler()
        
        # Test the color detection method
        with patch('sys.stderr.isatty', return_value=True):
            with patch.dict(os.environ, {'TERM': 'xterm'}):
                assert handler._supports_color() is True
        
        with patch('sys.stderr.isatty', return_value=False):
            assert handler._supports_color() is False
        
        with patch.dict(os.environ, {'TERM': 'dumb'}):
            assert handler._supports_color() is False
    
    def test_stream_handler_emit(self):
        """Test stream handler emit method."""
        custom_stream = StringIO()
        handler = StreamHandler(stream=custom_stream, use_colors=False)
        
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
        
        output = custom_stream.getvalue()
        assert "Test message" in output
        assert "INFO" in output
    
    def test_stream_handler_emit_error_handling(self):
        """Test stream handler error handling."""
        handler = StreamHandler()
        
        # Close the stream to force an error
        handler.stream.close()
        
        record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
        
        # Should not raise an exception
        with patch.object(handler, 'handleError') as mock_handle_error:
            handler.emit(record)
            mock_handle_error.assert_called_once_with(record)


class TestSyslogHandlerComprehensive:
    """Comprehensive tests for SyslogHandler class."""
    
    def test_syslog_handler_initialization_default(self):
        """Test syslog handler initialization with defaults."""
        with patch('logging.handlers.SysLogHandler.__init__') as mock_init:
            mock_init.return_value = None
            
            handler = SyslogHandler()
            
            mock_init.assert_called_once_with(('localhost', 514), 'user')
    
    def test_syslog_handler_initialization_custom(self):
        """Test syslog handler initialization with custom parameters."""
        with patch('logging.handlers.SysLogHandler.__init__') as mock_init:
            mock_init.return_value = None
            
            handler = SyslogHandler(address=('syslog.example.com', 514), facility='local0')
            
            mock_init.assert_called_once_with(('syslog.example.com', 514), 'local0')
    
    def test_syslog_handler_emit(self):
        """Test syslog handler emit method."""
        with patch('logging.handlers.SysLogHandler.__init__') as mock_init:
            with patch('logging.handlers.SysLogHandler.emit') as mock_emit:
                with patch('os.uname') as mock_uname:
                    mock_init.return_value = None
                    mock_uname.return_value = type('obj', (object,), {'nodename': 'test-host'})()
                    
                    handler = SyslogHandler()
                    handler.setFormatter = Mock()  # Mock the setFormatter call
                    
                    record = logging.LogRecord(
                        name="test_logger",
                        level=logging.INFO,
                        pathname="test.py",
                        lineno=10,
                        msg="Test syslog message",
                        args=(),
                        exc_info=None
                    )
                    
                    handler.emit(record)
                    
                    # Verify that hostname and app_name were added
                    assert record.hostname == 'test-host'
                    assert record.app_name == 'autonomous_agent'
                    
                    # Verify parent emit was called
                    mock_emit.assert_called_once_with(record)
    
    def test_syslog_handler_emit_no_uname(self):
        """Test syslog handler emit when os.uname is not available."""
        with patch('logging.handlers.SysLogHandler.__init__') as mock_init:
            with patch('logging.handlers.SysLogHandler.emit') as mock_emit:
                with patch.object(os, 'uname', side_effect=AttributeError):
                    mock_init.return_value = None
                    
                    handler = SyslogHandler()
                    handler.setFormatter = Mock()
                    
                    record = logging.LogRecord(
                        name="test_logger",
                        level=logging.INFO,
                        pathname="test.py",
                        lineno=10,
                        msg="Test syslog message",
                        args=(),
                        exc_info=None
                    )
                    
                    handler.emit(record)
                    
                    # Should use 'unknown' when uname is not available
                    assert record.hostname == 'unknown'
                    assert record.app_name == 'autonomous_agent'
    
    def test_syslog_handler_emit_error_handling(self):
        """Test syslog handler error handling."""
        with patch('logging.handlers.SysLogHandler.__init__') as mock_init:
            with patch('logging.handlers.SysLogHandler.emit', side_effect=Exception("Syslog error")):
                mock_init.return_value = None
                
                handler = SyslogHandler()
                handler.setFormatter = Mock()
                
                record = logging.LogRecord("test", logging.ERROR, "test.py", 10, "Error message", (), None)
                
                # Should not raise an exception
                with patch.object(handler, 'handleError') as mock_handle_error:
                    handler.emit(record)
                    mock_handle_error.assert_called_once_with(record)


class TestHandlersIntegration:
    """Integration tests for multiple handlers working together."""
    
    def test_multiple_handlers_same_record(self):
        """Test that the same record can be processed by multiple handlers."""
        # Create mock clients
        mock_database = Mock()
        mock_metrics = Mock()
        mock_alerts = Mock()
        
        # Create handlers
        db_handler = DatabaseHandler(mock_database)
        metrics_handler = MetricsHandler(mock_metrics)
        alert_handler = AlertHandler(mock_alerts, alert_level=logging.ERROR)
        
        # Create an error record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Critical system error",
            args=(),
            exc_info=None
        )
        
        # Emit to all handlers
        db_handler.emit(record)
        metrics_handler.emit(record)
        alert_handler.emit(record)
        
        # Verify all handlers processed the record
        mock_database.insert.assert_called_once()
        mock_metrics.increment.assert_called()
        mock_alerts.send_alert.assert_called_once()
    
    def test_handler_level_filtering(self):
        """Test that handlers respect their level settings."""
        mock_metrics = Mock()
        mock_alerts = Mock()
        
        # Set different levels
        metrics_handler = MetricsHandler(mock_metrics)  # WARNING and above
        alert_handler = AlertHandler(mock_alerts, alert_level=logging.CRITICAL)  # CRITICAL only
        
        # Test with WARNING level record
        warning_record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        
        metrics_handler.emit(warning_record)
        alert_handler.emit(warning_record)
        
        # Metrics should process WARNING, alerts should not
        mock_metrics.increment.assert_called()
        mock_alerts.send_alert.assert_not_called()
        
        # Reset mocks
        mock_metrics.reset_mock()
        mock_alerts.reset_mock()
        
        # Test with CRITICAL level record
        critical_record = logging.LogRecord(
            name="test_logger",
            level=logging.CRITICAL,
            pathname="test.py",
            lineno=10,
            msg="Critical message",
            args=(),
            exc_info=None
        )
        
        metrics_handler.emit(critical_record)
        alert_handler.emit(critical_record)
        
        # Both should process CRITICAL
        mock_metrics.increment.assert_called()
        mock_alerts.send_alert.assert_called_once()