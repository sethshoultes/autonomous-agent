"""
Integration test configuration and fixtures.

This module provides fixtures and configurations specifically for integration tests,
which test the interaction between multiple components of the autonomous agent system.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from tests.mocks.gmail_mocks import MockGmailService
from tests.mocks.github_mocks import MockGitHubClient
from tests.mocks.ollama_mocks import MockOllamaClient


# ============================================================================
# Integration test markers
# ============================================================================

# Mark tests that require external services (even if mocked)
pytestmark = pytest.mark.integration


# ============================================================================
# Integration test configuration
# ============================================================================

@pytest.fixture(scope="session")
def integration_config() -> Dict[str, Any]:
    """Provide configuration for integration tests."""
    return {
        "test_timeout": 30.0,
        "max_retries": 3,
        "retry_delay": 1.0,
        "mock_external_services": True,
        "enable_logging": True,
        "log_level": "DEBUG",
    }


@pytest.fixture(scope="session")
def test_environment() -> Dict[str, str]:
    """Set up test environment variables."""
    return {
        "TESTING": "true",
        "ENVIRONMENT": "integration_test",
        "LOG_LEVEL": "DEBUG",
        "DISABLE_EXTERNAL_APIS": "true",
        "MOCK_SERVICES": "true",
        "TEST_MODE": "integration",
    }


# ============================================================================
# Service integration fixtures
# ============================================================================

@pytest.fixture
def integrated_gmail_service(mock_gmail_service: MockGmailService) -> MockGmailService:
    """Provide an integrated Gmail service with test data."""
    from tests.mocks.gmail_mocks import generate_sample_emails, generate_spam_emails
    
    # Add sample emails
    for email in generate_sample_emails(10):
        mock_gmail_service.add_message(email)
    
    # Add spam emails
    for email in generate_spam_emails(3):
        mock_gmail_service.add_message(email)
    
    return mock_gmail_service


@pytest.fixture
def integrated_github_client(mock_github_client: MockGitHubClient) -> MockGitHubClient:
    """Provide an integrated GitHub client with test data."""
    from tests.mocks.github_mocks import (
        generate_sample_repositories, 
        generate_sample_pull_requests,
        generate_sample_issues
    )
    
    # Add sample repositories
    for repo in generate_sample_repositories(5):
        mock_github_client.add_repository(repo)
    
    # Add sample pull requests
    for pr in generate_sample_pull_requests(3):
        mock_github_client.add_pull_request(pr)
    
    # Add sample issues
    for issue in generate_sample_issues(4):
        mock_github_client.add_issue(issue)
    
    return mock_github_client


@pytest.fixture
def integrated_ollama_client(mock_ollama_client: MockOllamaClient) -> MockOllamaClient:
    """Provide an integrated Ollama client with test models."""
    from tests.mocks.ollama_mocks import generate_sample_models
    
    # Add sample models
    for model in generate_sample_models(3):
        mock_ollama_client.add_model(model)
    
    return mock_ollama_client


# ============================================================================
# Multi-service integration fixtures
# ============================================================================

@pytest.fixture
def integrated_services(
    integrated_gmail_service: MockGmailService,
    integrated_github_client: MockGitHubClient,
    integrated_ollama_client: MockOllamaClient,
) -> Dict[str, Any]:
    """Provide all integrated services together."""
    return {
        "gmail": integrated_gmail_service,
        "github": integrated_github_client,
        "ollama": integrated_ollama_client,
    }


@pytest.fixture
def mock_service_registry() -> Dict[str, Any]:
    """Provide a mock service registry for integration tests."""
    registry = MagicMock()
    registry.get_service.side_effect = lambda name: {
        "gmail": MagicMock(),
        "github": MagicMock(),
        "ollama": MagicMock(),
        "redis": MagicMock(),
        "database": MagicMock(),
    }.get(name)
    
    registry.register_service.return_value = True
    registry.unregister_service.return_value = True
    registry.list_services.return_value = ["gmail", "github", "ollama", "redis", "database"]
    
    return registry


# ============================================================================
# Database integration fixtures
# ============================================================================

@pytest.fixture
async def integration_database():
    """Provide a test database for integration tests."""
    # In a real implementation, this might set up a test database
    # For now, we'll use a mock
    db_mock = AsyncMock()
    
    # Mock common database operations
    db_mock.execute.return_value = MagicMock()
    db_mock.fetch.return_value = []
    db_mock.fetchone.return_value = None
    db_mock.fetchval.return_value = None
    
    # Mock transaction support
    transaction_mock = AsyncMock()
    transaction_mock.start.return_value = None
    transaction_mock.commit.return_value = None
    transaction_mock.rollback.return_value = None
    db_mock.transaction.return_value = transaction_mock
    
    yield db_mock
    
    # Cleanup would go here in a real implementation
    await db_mock.close()


@pytest.fixture
async def integration_redis():
    """Provide a test Redis instance for integration tests."""
    # In a real implementation, this might set up a test Redis instance
    # For now, we'll use a mock
    redis_mock = AsyncMock()
    
    # Mock Redis operations
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.expire.return_value = True
    redis_mock.lpush.return_value = 1
    redis_mock.rpop.return_value = None
    redis_mock.llen.return_value = 0
    redis_mock.flushdb.return_value = True
    
    yield redis_mock
    
    # Cleanup would go here in a real implementation
    await redis_mock.close()


# ============================================================================
# Agent integration fixtures
# ============================================================================

@pytest.fixture
def mock_agent_manager():
    """Provide a mock agent manager for integration tests."""
    manager = AsyncMock()
    
    # Mock agent lifecycle methods
    manager.start.return_value = True
    manager.stop.return_value = True
    manager.restart.return_value = True
    manager.get_status.return_value = "running"
    
    # Mock agent registration
    manager.register_agent.return_value = True
    manager.unregister_agent.return_value = True
    manager.list_agents.return_value = ["gmail_agent", "github_agent", "ollama_agent"]
    
    # Mock message passing
    manager.send_message.return_value = True
    manager.broadcast_message.return_value = True
    
    return manager


@pytest.fixture
def mock_gmail_agent():
    """Provide a mock Gmail agent for integration tests."""
    agent = AsyncMock()
    
    # Mock agent lifecycle
    agent.start.return_value = True
    agent.stop.return_value = True
    agent.get_status.return_value = "running"
    
    # Mock Gmail operations
    agent.fetch_emails.return_value = []
    agent.process_email.return_value = {"status": "processed"}
    agent.send_email.return_value = {"status": "sent"}
    agent.classify_email.return_value = {"category": "inbox", "confidence": 0.9}
    
    return agent


@pytest.fixture
def mock_github_agent():
    """Provide a mock GitHub agent for integration tests."""
    agent = AsyncMock()
    
    # Mock agent lifecycle
    agent.start.return_value = True
    agent.stop.return_value = True
    agent.get_status.return_value = "running"
    
    # Mock GitHub operations
    agent.monitor_repository.return_value = {"status": "monitoring"}
    agent.process_pull_request.return_value = {"status": "reviewed"}
    agent.create_issue.return_value = {"status": "created", "issue_id": 123}
    agent.review_code.return_value = {"status": "reviewed", "comments": []}
    
    return agent


@pytest.fixture
def mock_ollama_agent():
    """Provide a mock Ollama agent for integration tests."""
    agent = AsyncMock()
    
    # Mock agent lifecycle
    agent.start.return_value = True
    agent.stop.return_value = True
    agent.get_status.return_value = "running"
    
    # Mock AI operations
    agent.generate_text.return_value = {"text": "Generated text response"}
    agent.chat.return_value = {"response": "Chat response"}
    agent.analyze_content.return_value = {"analysis": "Content analysis"}
    agent.summarize.return_value = {"summary": "Content summary"}
    
    return agent


@pytest.fixture
def integrated_agent_system(
    mock_agent_manager,
    mock_gmail_agent,
    mock_github_agent,
    mock_ollama_agent,
):
    """Provide a complete integrated agent system."""
    system = {
        "manager": mock_agent_manager,
        "agents": {
            "gmail": mock_gmail_agent,
            "github": mock_github_agent,
            "ollama": mock_ollama_agent,
        }
    }
    
    # Configure agent manager to return our mock agents
    mock_agent_manager.get_agent.side_effect = lambda name: system["agents"].get(name)
    
    return system


# ============================================================================
# Test data fixtures
# ============================================================================

@pytest.fixture
def integration_test_data() -> Dict[str, Any]:
    """Provide comprehensive test data for integration tests."""
    return {
        "emails": [
            {
                "from": "user1@example.com",
                "to": "agent@example.com",
                "subject": "Test Email 1",
                "body": "This is a test email for integration testing.",
            },
            {
                "from": "user2@example.com", 
                "to": "agent@example.com",
                "subject": "Test Email 2",
                "body": "Another test email with different content.",
            },
        ],
        "repositories": [
            {
                "name": "test-repo-1",
                "owner": "test-user",
                "description": "First test repository",
            },
            {
                "name": "test-repo-2",
                "owner": "test-user",
                "description": "Second test repository",
            },
        ],
        "pull_requests": [
            {
                "title": "Feature: Add new functionality",
                "body": "This PR adds new functionality to the system.",
                "head_ref": "feature-branch",
                "base_ref": "main",
            },
        ],
        "ai_prompts": [
            "Analyze this email and determine its priority",
            "Review this code and provide feedback",
            "Summarize the key points from this text",
        ],
    }


# ============================================================================
# Test lifecycle fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def setup_integration_test():
    """Set up each integration test."""
    # Setup code here
    yield
    # Cleanup code here
    pass


@pytest.fixture(scope="function")
async def isolated_test_environment(test_environment):
    """Provide an isolated environment for each test."""
    with patch.dict(os.environ, test_environment):
        yield


# ============================================================================
# Performance and monitoring fixtures
# ============================================================================

@pytest.fixture
def performance_monitor():
    """Provide a performance monitor for integration tests."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            
            return {
                "execution_time": end_time - self.start_time,
                "memory_used_mb": (end_memory - self.start_memory) / 1024 / 1024,
                "peak_memory_mb": self.process.memory_info().rss / 1024 / 1024,
            }
    
    return PerformanceMonitor()


@pytest.fixture
def test_timeout():
    """Provide a timeout for integration tests."""
    return 30.0  # 30 seconds timeout for integration tests


# ============================================================================
# Utility fixtures
# ============================================================================

@pytest.fixture
def temp_workspace():
    """Provide a temporary workspace for integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create common directories
        (workspace / "config").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "data").mkdir()
        (workspace / "temp").mkdir()
        
        yield workspace


@pytest.fixture
def integration_config_file(temp_workspace: Path, integration_config: Dict[str, Any]) -> Path:
    """Create a configuration file for integration tests."""
    config_path = temp_workspace / "config" / "integration_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(integration_config, f)
    return config_path


# ============================================================================
# Cleanup fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_integration_test():
    """Clean up after each integration test."""
    yield
    # Cleanup code would go here
    pass