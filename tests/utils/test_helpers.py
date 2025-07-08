"""
Test helper functions and utilities.

This module provides common utilities for test setup, data generation,
and test validation following TDD principles and SOLID design patterns.
"""

import asyncio
import json
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_uuid() -> str:
        """Generate a mock UUID for testing."""
        import uuid
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_timestamp() -> str:
        """Generate a mock timestamp."""
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def generate_email_address(username: str = "test", domain: str = "example.com") -> str:
        """Generate a test email address."""
        return f"{username}@{domain}"
    
    @staticmethod
    def generate_url(protocol: str = "https", domain: str = "example.com", path: str = "") -> str:
        """Generate a test URL."""
        return f"{protocol}://{domain}{path}"
    
    @staticmethod
    def generate_agent_config(agent_id: str = None, agent_type: str = "test") -> Dict[str, Any]:
        """Generate a test agent configuration."""
        return {
            "agent_id": agent_id or TestDataGenerator.generate_uuid(),
            "agent_type": agent_type,
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 30.0,
            "heartbeat_interval": 10.0,
            "log_level": "DEBUG",
            "metrics_enabled": True,
        }
    
    @staticmethod
    def generate_email_data(from_addr: str = None, to_addr: str = None) -> Dict[str, Any]:
        """Generate test email data."""
        return {
            "from": from_addr or TestDataGenerator.generate_email_address("sender"),
            "to": to_addr or TestDataGenerator.generate_email_address("recipient"),
            "subject": "Test Email Subject",
            "body": "This is a test email body with some content.",
            "date": TestDataGenerator.generate_timestamp(),
            "message_id": TestDataGenerator.generate_uuid(),
            "thread_id": TestDataGenerator.generate_uuid(),
            "labels": ["INBOX", "UNREAD"],
            "attachments": [],
        }
    
    @staticmethod
    def generate_github_data(repo_name: str = "test-repo", owner: str = "test-user") -> Dict[str, Any]:
        """Generate test GitHub repository data."""
        return {
            "repository": {
                "id": 123456789,
                "name": repo_name,
                "full_name": f"{owner}/{repo_name}",
                "owner": {"login": owner, "id": 12345},
                "private": False,
                "html_url": f"https://github.com/{owner}/{repo_name}",
                "description": "A test repository",
            },
            "sender": {"login": owner, "id": 12345},
        }
    
    @staticmethod
    def generate_ollama_data(model: str = "llama3.1:8b") -> Dict[str, Any]:
        """Generate test Ollama response data."""
        return {
            "model": model,
            "created_at": TestDataGenerator.generate_timestamp(),
            "response": "This is a mock response from Ollama",
            "done": True,
            "context": [1, 2, 3, 4, 5],
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "eval_count": 20,
            "eval_duration": 3000000000,
        }


class AsyncTestHelper:
    """Helper functions for async testing."""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        return False
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run a coroutine with a timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            pytest.fail(f"Operation timed out after {timeout} seconds")
    
    @staticmethod
    @asynccontextmanager
    async def async_mock_context(mock_obj: AsyncMock):
        """Context manager for async mock objects."""
        try:
            yield mock_obj
        finally:
            mock_obj.reset_mock()
    
    @staticmethod
    def create_async_mock_with_return(return_value: Any) -> AsyncMock:
        """Create an async mock that returns a specific value."""
        mock = AsyncMock()
        mock.return_value = return_value
        return mock
    
    @staticmethod
    def create_async_mock_with_side_effect(side_effect: Any) -> AsyncMock:
        """Create an async mock with a side effect."""
        mock = AsyncMock()
        mock.side_effect = side_effect
        return mock


class FileTestHelper:
    """Helper functions for file-related testing."""
    
    @staticmethod
    @contextmanager
    def temp_file(content: str = "", suffix: str = ".txt"):
        """Create a temporary file with content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            yield temp_path
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @staticmethod
    @contextmanager
    def temp_directory():
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @staticmethod
    def create_config_file(config_data: Dict[str, Any], temp_dir: Path) -> Path:
        """Create a temporary configuration file."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path
    
    @staticmethod
    def create_json_file(data: Dict[str, Any], temp_dir: Path, filename: str = "data.json") -> Path:
        """Create a temporary JSON file."""
        json_path = temp_dir / filename
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return json_path
    
    @staticmethod
    def read_file_content(file_path: Path) -> str:
        """Read content from a file."""
        with open(file_path, "r") as f:
            return f.read()
    
    @staticmethod
    def assert_file_exists(file_path: Path) -> None:
        """Assert that a file exists."""
        assert file_path.exists(), f"File {file_path} does not exist"
    
    @staticmethod
    def assert_file_contains(file_path: Path, content: str) -> None:
        """Assert that a file contains specific content."""
        FileTestHelper.assert_file_exists(file_path)
        file_content = FileTestHelper.read_file_content(file_path)
        assert content in file_content, f"File {file_path} does not contain '{content}'"


class MockTestHelper:
    """Helper functions for working with mocks."""
    
    @staticmethod
    def create_mock_with_spec(spec_class: Type) -> MagicMock:
        """Create a mock with a specific spec."""
        return MagicMock(spec=spec_class)
    
    @staticmethod
    def create_async_mock_with_spec(spec_class: Type) -> AsyncMock:
        """Create an async mock with a specific spec."""
        return AsyncMock(spec=spec_class)
    
    @staticmethod
    def assert_mock_called_with_partial(mock_obj: MagicMock, **expected_kwargs) -> None:
        """Assert that a mock was called with specific keyword arguments."""
        mock_obj.assert_called()
        call_args = mock_obj.call_args
        if call_args is None:
            pytest.fail("Mock was not called")
        
        _, actual_kwargs = call_args
        for key, expected_value in expected_kwargs.items():
            assert key in actual_kwargs, f"Expected keyword argument '{key}' not found"
            assert actual_kwargs[key] == expected_value, f"Expected {key}={expected_value}, got {actual_kwargs[key]}"
    
    @staticmethod
    def assert_async_mock_awaited_with_partial(mock_obj: AsyncMock, **expected_kwargs) -> None:
        """Assert that an async mock was awaited with specific keyword arguments."""
        mock_obj.assert_awaited()
        call_args = mock_obj.await_args
        if call_args is None:
            pytest.fail("Async mock was not awaited")
        
        _, actual_kwargs = call_args
        for key, expected_value in expected_kwargs.items():
            assert key in actual_kwargs, f"Expected keyword argument '{key}' not found"
            assert actual_kwargs[key] == expected_value, f"Expected {key}={expected_value}, got {actual_kwargs[key]}"
    
    @staticmethod
    def create_mock_context_manager(return_value: Any = None) -> MagicMock:
        """Create a mock context manager."""
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = return_value
        mock_cm.__exit__.return_value = False
        return mock_cm
    
    @staticmethod
    def create_async_mock_context_manager(return_value: Any = None) -> AsyncMock:
        """Create an async mock context manager."""
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = return_value
        mock_cm.__aexit__.return_value = False
        return mock_cm


class ValidationTestHelper:
    """Helper functions for validation testing."""
    
    @staticmethod
    def assert_valid_uuid(uuid_string: str) -> None:
        """Assert that a string is a valid UUID."""
        import uuid
        try:
            uuid.UUID(uuid_string)
        except ValueError:
            pytest.fail(f"'{uuid_string}' is not a valid UUID")
    
    @staticmethod
    def assert_valid_email(email: str) -> None:
        """Assert that a string is a valid email address."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert re.match(email_pattern, email), f"'{email}' is not a valid email address"
    
    @staticmethod
    def assert_valid_url(url: str) -> None:
        """Assert that a string is a valid URL."""
        import re
        url_pattern = r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/.*)?$'
        assert re.match(url_pattern, url), f"'{url}' is not a valid URL"
    
    @staticmethod
    def assert_valid_timestamp(timestamp: str) -> None:
        """Assert that a string is a valid ISO timestamp."""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"'{timestamp}' is not a valid ISO timestamp")
    
    @staticmethod
    def assert_dict_contains_keys(data: Dict[str, Any], required_keys: List[str]) -> None:
        """Assert that a dictionary contains all required keys."""
        missing_keys = [key for key in required_keys if key not in data]
        assert not missing_keys, f"Dictionary missing required keys: {missing_keys}"
    
    @staticmethod
    def assert_dict_values_not_none(data: Dict[str, Any], keys: List[str]) -> None:
        """Assert that specific dictionary values are not None."""
        none_values = [key for key in keys if data.get(key) is None]
        assert not none_values, f"Dictionary has None values for keys: {none_values}"
    
    @staticmethod
    def assert_list_not_empty(data: List[Any], name: str = "list") -> None:
        """Assert that a list is not empty."""
        assert data, f"{name} should not be empty"
    
    @staticmethod
    def assert_all_items_have_type(data: List[Any], expected_type: Type, name: str = "list") -> None:
        """Assert that all items in a list have the expected type."""
        for i, item in enumerate(data):
            assert isinstance(item, expected_type), f"Item {i} in {name} has type {type(item)}, expected {expected_type}"


class PerformanceTestHelper:
    """Helper functions for performance testing."""
    
    @staticmethod
    @contextmanager
    def measure_time():
        """Context manager to measure execution time."""
        start_time = time.time()
        yield lambda: time.time() - start_time
    
    @staticmethod
    async def measure_async_time(coro):
        """Measure execution time of an async operation."""
        start_time = time.time()
        result = await coro
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def assert_execution_time_under(seconds: float):
        """Assert that execution time is under a threshold."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    assert execution_time < seconds, f"Execution took {execution_time:.3f}s, expected under {seconds}s"
                    return result
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    assert execution_time < seconds, f"Execution took {execution_time:.3f}s, expected under {seconds}s"
                    return result
                return sync_wrapper
        return decorator
    
    @staticmethod
    def create_memory_profiler():
        """Create a simple memory profiler for testing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        class MemoryProfiler:
            def __init__(self):
                self.start_memory = None
                self.peak_memory = None
            
            def start(self):
                self.start_memory = process.memory_info().rss
                self.peak_memory = self.start_memory
            
            def update(self):
                current_memory = process.memory_info().rss
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
            
            def get_memory_usage(self):
                return {
                    "start_mb": self.start_memory / 1024 / 1024,
                    "peak_mb": self.peak_memory / 1024 / 1024,
                    "increase_mb": (self.peak_memory - self.start_memory) / 1024 / 1024,
                }
        
        return MemoryProfiler()


class DatabaseTestHelper:
    """Helper functions for database testing."""
    
    @staticmethod
    def create_mock_database_connection() -> AsyncMock:
        """Create a mock database connection."""
        db_mock = AsyncMock()
        db_mock.execute.return_value = MagicMock()
        db_mock.fetch.return_value = []
        db_mock.fetchone.return_value = None
        db_mock.fetchval.return_value = None
        db_mock.close.return_value = None
        return db_mock
    
    @staticmethod
    def create_mock_transaction() -> AsyncMock:
        """Create a mock database transaction."""
        transaction_mock = AsyncMock()
        transaction_mock.start.return_value = None
        transaction_mock.commit.return_value = None
        transaction_mock.rollback.return_value = None
        return transaction_mock
    
    @staticmethod
    def assert_sql_contains(actual_sql: str, expected_fragments: List[str]) -> None:
        """Assert that SQL contains expected fragments."""
        actual_sql_lower = actual_sql.lower()
        for fragment in expected_fragments:
            assert fragment.lower() in actual_sql_lower, f"SQL does not contain '{fragment}'"


class ConfigTestHelper:
    """Helper functions for configuration testing."""
    
    @staticmethod
    def create_test_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a test configuration with optional overrides."""
        base_config = {
            "agent_manager": {
                "max_workers": 4,
                "heartbeat_interval": 30,
                "shutdown_timeout": 60,
            },
            "logging": {
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }
        
        if overrides:
            ConfigTestHelper._deep_update(base_config, overrides)
        
        return base_config
    
    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update a dictionary with another dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigTestHelper._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @staticmethod
    def assert_config_valid(config: Dict[str, Any], required_sections: List[str]) -> None:
        """Assert that a configuration is valid."""
        for section in required_sections:
            assert section in config, f"Configuration missing required section: {section}"
    
    @staticmethod
    def mock_environment_variables(env_vars: Dict[str, str]):
        """Mock environment variables for testing."""
        return patch.dict("os.environ", env_vars)


# Export all helper classes for easy importing
__all__ = [
    "TestDataGenerator",
    "AsyncTestHelper",
    "FileTestHelper",
    "MockTestHelper",
    "ValidationTestHelper",
    "PerformanceTestHelper",
    "DatabaseTestHelper",
    "ConfigTestHelper",
]