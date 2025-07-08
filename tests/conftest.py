"""
Pytest configuration and shared fixtures for the test suite.

This module provides common test fixtures and configuration that can be
used across all test modules.
"""

import pytest
from fastapi.testclient import TestClient
from typing import Generator, Dict, Any

from src.main import create_app, Settings


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with appropriate values for testing."""
    return Settings(
        app_name="Test Autonomous Agent System",
        app_version="0.1.0-test",
        debug=True,
        log_level="DEBUG",
        host="127.0.0.1",
        port=8888,
    )


@pytest.fixture
def test_app(test_settings: Settings) -> Generator[TestClient, None, None]:
    """Create a test application instance."""
    app = create_app(test_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        "timeout": 30,
        "retry_count": 3,
        "debug": True,
        "batch_size": 10,
    }


@pytest.fixture
def sample_agent_name() -> str:
    """Provide a sample agent name for testing."""
    return "TestAgent"


@pytest.fixture
def sample_service_name() -> str:
    """Provide a sample service name for testing."""
    return "TestService"


# Async fixtures
@pytest.fixture
async def async_sample_data() -> Dict[str, Any]:
    """Provide sample data for async tests."""
    return {
        "id": "test-123",
        "name": "Test Data",
        "value": 42,
        "active": True,
    }