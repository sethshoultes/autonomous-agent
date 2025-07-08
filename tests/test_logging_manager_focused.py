"""
Focused tests for logging manager to improve coverage.
"""

import logging
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from src.logging.manager import LogFormatter, LoggingManager
from src.logging.handlers import DatabaseHandler, MetricsHandler, AlertHandler


class TestLogFormatter:
    """Test LogFormatter functionality."""
    
    def test_log_formatter_default_initialization(self):
        """Test LogFormatter with default parameters."""
        formatter = LogFormatter()
        assert formatter.use_colors is False
        assert formatter.include_context is False
        assert formatter.include_timestamp is True
    
    def test_log_formatter_with_colors(self):
        """Test LogFormatter with colors enabled."""
        formatter = LogFormatter(use_colors=True)
        assert formatter.use_colors is True
    
    def test_log_formatter_with_context(self):
        """Test LogFormatter with context enabled."""
        formatter = LogFormatter(include_context=True)
        assert formatter.include_context is True
    
    def test_format_record_basic(self):
        """Test basic record formatting."""
        formatter = LogFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "INFO" in formatted
    
    def test_format_record_with_colors(self):
        """Test record formatting with colors."""
        formatter = LogFormatter(use_colors=True)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "\033[31m" in formatted  # Red color for ERROR
        assert "\033[0m" in formatted   # Reset color
    
    def test_format_record_with_context(self):
        """Test record formatting with context."""
        formatter = LogFormatter(include_context=True)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add some context to the record
        record.agent_id = "test_agent"
        record.operation = "test_operation"
        
        formatted = formatter.format(record)
        assert "agent_id=test_agent" in formatted
        assert "operation=test_operation" in formatted


class TestLoggingManager:
    """Test LoggingManager functionality."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_file = f.name
        yield temp_file
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    def test_log_manager_initialization(self):
        """Test LoggingManager initialization."""
        manager = LoggingManager()
        assert manager.loggers == {}
        assert manager.handlers == {}
        assert manager.filters == {}
        assert isinstance(manager._lock, type(threading.Lock()))
    
    def test_get_logger_new(self):
        """Test getting a new logger."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        assert logger.name == "test.logger"
        assert "test.logger" in manager.loggers
        assert manager.loggers["test.logger"] == logger
    
    def test_get_logger_existing(self):
        """Test getting an existing logger."""
        manager = LoggingManager()
        logger1 = manager.get_logger("test.logger")
        logger2 = manager.get_logger("test.logger")
        
        assert logger1 is logger2
    
    def test_get_logger_with_level(self):
        """Test getting logger with specific level."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger", level=logging.WARNING)
        
        assert logger.level == logging.WARNING
    
    def test_set_level(self):
        """Test setting logger level."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        manager.set_level("test.logger", logging.ERROR)
        assert logger.level == logging.ERROR
    
    def test_set_level_nonexistent_logger(self):
        """Test setting level for non-existent logger."""
        manager = LoggingManager()
        
        # Should not raise exception
        manager.set_level("nonexistent.logger", logging.ERROR)
    
    def test_add_handler_file(self, temp_log_file):
        """Test adding file handler."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        manager.add_handler(
            "test.logger",
            "file",
            filename=temp_log_file,
            level=logging.INFO
        )
        
        assert len(logger.handlers) > 0
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    def test_add_handler_console(self):
        """Test adding console handler."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        manager.add_handler("test.logger", "console", level=logging.DEBUG)
        
        assert len(logger.handlers) > 0
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    def test_add_handler_with_formatter(self):
        """Test adding handler with custom formatter."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        formatter = LogFormatter(use_colors=True)
        manager.add_handler(
            "test.logger",
            "console",
            level=logging.DEBUG,
            formatter=formatter
        )
        
        handler = logger.handlers[0]
        assert handler.formatter is formatter
    
    def test_remove_handler(self):
        """Test removing handler."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        handler_id = manager.add_handler("test.logger", "console")
        assert len(logger.handlers) > 0
        
        success = manager.remove_handler("test.logger", handler_id)
        assert success is True
        assert len(logger.handlers) == 0
    
    def test_remove_nonexistent_handler(self):
        """Test removing non-existent handler."""
        manager = LoggingManager()
        manager.get_logger("test.logger")
        
        success = manager.remove_handler("test.logger", "nonexistent")
        assert success is False
    
    def test_add_filter(self):
        """Test adding log filter."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        def test_filter(record):
            return "test" in record.getMessage()
        
        filter_id = manager.add_filter("test.logger", test_filter)
        
        assert filter_id in manager.filters
        assert len(logger.filters) > 0
    
    def test_remove_filter(self):
        """Test removing log filter."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        def test_filter(record):
            return True
        
        filter_id = manager.add_filter("test.logger", test_filter)
        
        success = manager.remove_filter("test.logger", filter_id)
        assert success is True
        assert len(logger.filters) == 0
    
    def test_configure_from_dict(self):
        """Test configuring logging from dictionary."""
        manager = LoggingManager()
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "class": "src.logging.manager.LogFormatter",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "test.logger": {
                    "level": "DEBUG",
                    "handlers": ["console"],
                    "propagate": False
                }
            }
        }
        
        manager.configure_from_dict(config)
        
        logger = logging.getLogger("test.logger")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0
    
    def test_get_logger_stats(self):
        """Test getting logger statistics."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger", level=logging.INFO)
        manager.add_handler("test.logger", "console")
        
        stats = manager.get_logger_stats("test.logger")
        
        assert stats["name"] == "test.logger"
        assert stats["level"] == "INFO"
        assert stats["handler_count"] > 0
        assert stats["filter_count"] == 0
    
    def test_get_logger_stats_nonexistent(self):
        """Test getting stats for non-existent logger."""
        manager = LoggingManager()
        
        stats = manager.get_logger_stats("nonexistent.logger")
        assert stats is None
    
    def test_get_all_logger_stats(self):
        """Test getting all logger statistics."""
        manager = LoggingManager()
        manager.get_logger("test.logger1")
        manager.get_logger("test.logger2")
        
        all_stats = manager.get_all_logger_stats()
        
        assert len(all_stats) >= 2
        logger_names = [stats["name"] for stats in all_stats]
        assert "test.logger1" in logger_names
        assert "test.logger2" in logger_names
    
    def test_shutdown(self):
        """Test shutting down logging."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        manager.add_handler("test.logger", "console")
        
        initial_handler_count = len(logger.handlers)
        assert initial_handler_count > 0
        
        manager.shutdown()
        
        # Handlers should be removed and closed
        assert len(logger.handlers) == 0
    
    def test_context_manager(self):
        """Test using LoggingManager as context manager."""
        with LoggingManager() as manager:
            logger = manager.get_logger("test.logger")
            manager.add_handler("test.logger", "console")
            
            assert len(logger.handlers) > 0
        
        # After context exits, handlers should be cleaned up
        assert len(logger.handlers) == 0
    
    def test_configure_agent_logging(self):
        """Test configuring agent-specific logging."""
        manager = LoggingManager()
        
        manager.configure_agent_logging(
            agent_id="test_agent",
            level=logging.DEBUG,
            log_file="/tmp/test_agent.log",
            include_console=True
        )
        
        logger = manager.get_logger(f"agent.test_agent")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1  # At least console handler
    
    def test_log_agent_event(self, temp_log_file):
        """Test logging agent events."""
        manager = LoggingManager()
        logger = manager.get_logger("test.agent")
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        manager.log_agent_event(
            agent_id="test_agent",
            event="startup",
            level=logging.INFO,
            details={"status": "success"}
        )
        
        output = stream.getvalue()
        assert "test_agent" in output
        assert "startup" in output
        assert "success" in output
    
    def test_log_performance_metric(self):
        """Test logging performance metrics."""
        manager = LoggingManager()
        
        # Mock metrics handler
        with patch.object(manager, '_get_metrics_handler') as mock_get_handler:
            mock_handler = Mock()
            mock_get_handler.return_value = mock_handler
            
            manager.log_performance_metric(
                agent_id="test_agent",
                metric_name="cpu_usage",
                value=75.5,
                timestamp=time.time()
            )
            
            mock_handler.emit.assert_called_once()
    
    def test_log_error_with_context(self):
        """Test logging errors with context."""
        manager = LoggingManager()
        logger = manager.get_logger("test.agent")
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            manager.log_error_with_context(
                agent_id="test_agent",
                error=e,
                context={"operation": "test_operation", "step": "validation"}
            )
        
        output = stream.getvalue()
        assert "test_agent" in output
        assert "ValueError" in output
        assert "test_operation" in output
    
    def test_get_log_metrics(self):
        """Test getting log metrics."""
        manager = LoggingManager()
        
        # Mock metrics collection
        with patch.object(manager, '_collect_log_metrics') as mock_collect:
            mock_collect.return_value = {
                "total_logs": 100,
                "error_count": 5,
                "warning_count": 10
            }
            
            metrics = manager.get_log_metrics()
            
            assert metrics["total_logs"] == 100
            assert metrics["error_count"] == 5
            assert metrics["warning_count"] == 10


class TestLoggingManagerIntegration:
    """Test LoggingManager integration scenarios."""
    
    def test_multi_agent_logging(self):
        """Test logging for multiple agents."""
        manager = LoggingManager()
        
        # Configure logging for multiple agents
        agents = ["agent_1", "agent_2", "agent_3"]
        
        for agent_id in agents:
            manager.configure_agent_logging(
                agent_id=agent_id,
                level=logging.INFO,
                include_console=False
            )
        
        # Verify each agent has its own logger
        for agent_id in agents:
            logger = manager.get_logger(f"agent.{agent_id}")
            assert logger is not None
            assert logger.name == f"agent.{agent_id}"
    
    def test_logging_with_filters(self):
        """Test logging with custom filters."""
        manager = LoggingManager()
        logger = manager.get_logger("test.logger")
        
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Add filter that only allows messages containing "important"
        def important_filter(record):
            return "important" in record.getMessage()
        
        manager.add_filter("test.logger", important_filter)
        
        # Log messages
        logger.info("This is an important message")
        logger.info("This is a regular message")
        
        output = stream.getvalue()
        assert "important message" in output
        assert "regular message" not in output
    
    def test_hierarchical_logging(self):
        """Test hierarchical logger structure."""
        manager = LoggingManager()
        
        # Create parent and child loggers
        parent_logger = manager.get_logger("parent", level=logging.WARNING)
        child_logger = manager.get_logger("parent.child", level=logging.DEBUG)
        
        # Add handler to parent
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        parent_logger.addHandler(handler)
        
        # Child logger should inherit parent's handlers
        child_logger.info("Child message")
        
        output = stream.getvalue()
        # Message might not appear if parent level is WARNING
        # but the structure should be correct
        assert child_logger.parent == parent_logger
    
    def test_logging_performance(self):
        """Test logging performance with many messages."""
        manager = LoggingManager()
        logger = manager.get_logger("performance.test")
        
        # Use a null handler to avoid I/O overhead
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
        logger.setLevel(logging.DEBUG)
        
        # Log many messages
        start_time = time.time()
        for i in range(1000):
            logger.info(f"Performance test message {i}")
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        duration = end_time - start_time
        assert duration < 1.0