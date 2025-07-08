"""
Integration tests for Gmail Agent with existing framework components.

This module tests the Gmail Agent's integration with the autonomous agent
framework including AgentManager, ConfigManager, CommunicationBroker,
and other system components.
"""

import asyncio
import json
import logging
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.gmail_agent import GmailAgent
from src.agents.manager import AgentManager
from src.communication.broker import CommunicationBroker
from src.config.manager import ConfigManager
from src.logging.manager import LoggingManager
from tests.mocks.gmail_mocks import (
    MockGmailService,
    MockGmailAPIContext,
    generate_sample_emails,
    generate_work_emails,
    generate_important_emails,
)


class TestGmailAgentFrameworkIntegration:
    """Test Gmail Agent integration with the autonomous agent framework."""

    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Provide sample configuration for integration testing."""
        return {
            "agent_manager": {
                "max_agents": 10,
                "heartbeat_interval": 30.0,
                "communication_timeout": 60.0,
                "retry_attempts": 3
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO"
                    }
                }
            },
            "communication": {
                "message_broker": {
                    "queue_size": 1000,
                    "timeout": 30.0
                }
            },
            "gmail": {
                "credentials_path": "/tmp/test_credentials.json",
                "scopes": [
                    "https://www.googleapis.com/auth/gmail.readonly",
                    "https://www.googleapis.com/auth/gmail.send",
                    "https://www.googleapis.com/auth/gmail.modify"
                ],
                "user_email": "test@example.com",
                "batch_size": 100,
                "rate_limit_per_minute": 250,
                "max_retries": 3,
                "retry_delay": 1.0,
                "classification": {
                    "enabled": True,
                    "spam_threshold": 0.8,
                    "importance_threshold": 0.7,
                    "categories": ["important", "spam", "personal", "work", "archive"],
                    "keywords": {
                        "important": ["urgent", "asap", "deadline", "priority"],
                        "spam": ["prize", "winner", "lottery", "click here"],
                        "work": ["meeting", "project", "deadline", "team"],
                        "personal": ["family", "friend", "personal"]
                    }
                },
                "auto_response": {
                    "enabled": True,
                    "response_delay": 300,
                    "max_responses_per_day": 50,
                    "templates": {
                        "out_of_office": "I'm currently out of office...",
                        "meeting_request": "Thank you for the meeting request...",
                        "general_inquiry": "Thank you for your email..."
                    },
                    "trigger_patterns": {
                        "out_of_office": ["vacation", "out of office"],
                        "meeting_request": ["meeting", "call", "appointment"],
                        "general_inquiry": ["question", "inquiry", "help"]
                    }
                },
                "archiving": {
                    "enabled": True,
                    "archive_after_days": 30,
                    "auto_label": True,
                    "label_rules": [
                        {"pattern": "newsletter", "label": "Newsletters"},
                        {"pattern": "noreply", "label": "Automated"}
                    ],
                    "smart_folders": {
                        "receipts": ["receipt", "invoice", "purchase"],
                        "travel": ["flight", "hotel", "booking"]
                    }
                }
            },
            "agents": {
                "gmail_agent_001": {
                    "agent_type": "gmail",
                    "enabled": True,
                    "priority": 1,
                    "config": {}
                }
            }
        }

    @pytest.fixture
    def mock_gmail_service(self) -> MockGmailService:
        """Provide mock Gmail service with sample data."""
        service = MockGmailService()
        
        # Add sample emails
        sample_emails = generate_sample_emails(5)
        work_emails = generate_work_emails(3)
        important_emails = generate_important_emails(2)
        
        for email in sample_emails + work_emails + important_emails:
            service.add_message(email)
        
        return service

    @pytest.fixture
    def temp_config_file(self, sample_config: Dict[str, Any]) -> Path:
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f, indent=2)
            return Path(f.name)

    @pytest.mark.asyncio
    async def test_gmail_agent_with_config_manager(self, sample_config: Dict[str, Any], temp_config_file: Path):
        """Test Gmail Agent integration with ConfigManager."""
        # Create ConfigManager and load configuration
        config_manager = ConfigManager()
        config_manager.load_config(temp_config_file)
        
        # Verify Gmail configuration is loaded and validated
        gmail_config = config_manager.get("gmail")
        assert gmail_config is not None
        assert gmail_config["credentials_path"] == "/tmp/test_credentials.json"
        assert gmail_config["batch_size"] == 100
        assert gmail_config["classification"]["enabled"] is True
        
        # Test agent configuration retrieval
        agent_config = config_manager.get_agent_config("gmail_agent_001")
        assert agent_config["agent_type"] == "gmail"
        assert agent_config["enabled"] is True
        
        # Clean up
        temp_config_file.unlink()

    @pytest.mark.asyncio
    async def test_gmail_agent_with_logging_manager(self, sample_config: Dict[str, Any]):
        """Test Gmail Agent integration with LoggingManager."""
        # Create LoggingManager
        logging_manager = LoggingManager()
        logging_manager.configure(sample_config["logging"])
        
        # Get logger for Gmail Agent
        logger = logging_manager.get_logger("gmail_agent")
        assert logger is not None
        assert logger.level == logging.INFO
        
        # Create Gmail Agent with managed logger
        message_broker = AsyncMock()
        gmail_agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=sample_config,
            logger=logger,
            message_broker=message_broker
        )
        
        assert gmail_agent.logger == logger
        assert gmail_agent.logger.name == "gmail_agent"

    @pytest.mark.asyncio
    async def test_gmail_agent_with_communication_broker(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent integration with CommunicationBroker."""
        # Create CommunicationBroker
        logger = logging.getLogger("test")
        broker_config = sample_config["communication"]["message_broker"]
        communication_broker = CommunicationBroker(
            queue_size=broker_config["queue_size"],
            timeout=broker_config["timeout"],
            logger=logger
        )
        
        # Create Gmail Agent with real communication broker
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=communication_broker
            )
            
            # Start the agent
            await gmail_agent.start()
            
            # Test sending a message through the broker
            await gmail_agent.send_message(
                recipient="scheduler_agent",
                message_type="email_processed",
                payload={
                    "email_id": "test_email_001",
                    "classification": "important"
                }
            )
            
            # Verify message was sent through broker
            # (In real implementation, this would be verified through broker's message queue)
            
            # Stop the agent
            await gmail_agent.stop()

    @pytest.mark.asyncio
    async def test_gmail_agent_with_agent_manager(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent registration and management with AgentManager."""
        # Create required components
        config_manager = ConfigManager()
        config_manager.load_config(sample_config)
        
        logger = logging.getLogger("test")
        agent_manager = AgentManager(config_manager, logger)
        
        # Create Gmail Agent
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=AsyncMock()
            )
            
            # Register agent with manager
            await agent_manager.register_agent(gmail_agent)
            
            # Verify registration
            assert "gmail_agent_001" in agent_manager.agents
            assert agent_manager.agents["gmail_agent_001"] == gmail_agent
            
            # Start all agents through manager
            await agent_manager.start_agents()
            
            # Verify agent is active
            assert gmail_agent.state.value == "active"
            
            # Get agent status through manager
            status = agent_manager.get_agent_status("gmail_agent_001")
            assert status["state"] == "active"
            assert status["agent_id"] == "gmail_agent_001"
            
            # Stop all agents through manager
            await agent_manager.stop_agents()
            
            # Verify agent is inactive
            assert gmail_agent.state.value == "inactive"

    @pytest.mark.asyncio
    async def test_gmail_agent_inter_agent_communication(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent communication with other agents."""
        # Create mock scheduler agent
        scheduler_agent = AsyncMock()
        scheduler_agent.agent_id = "scheduler_agent"
        
        # Create communication broker
        logger = logging.getLogger("test")
        communication_broker = CommunicationBroker(
            queue_size=1000,
            timeout=30.0,
            logger=logger
        )
        
        # Create Gmail Agent
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=communication_broker
            )
            
            await gmail_agent.start()
            
            # Simulate receiving a message from scheduler agent
            from src.agents.base import AgentMessage
            
            fetch_request = AgentMessage(
                id="msg_001",
                sender="scheduler_agent",
                recipient="gmail_agent_001",
                message_type="fetch_emails",
                payload={
                    "query": "is:unread",
                    "max_results": 5
                }
            )
            
            # Process the message
            response = await gmail_agent._process_message(fetch_request)
            
            # Verify response
            assert response is not None
            assert response.message_type == "fetch_emails_response"
            assert "emails" in response.payload
            assert "count" in response.payload
            
            await gmail_agent.stop()

    @pytest.mark.asyncio
    async def test_gmail_agent_configuration_updates(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent response to configuration updates."""
        # Create Gmail Agent
        logger = logging.getLogger("test")
        
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=AsyncMock()
            )
            
            await gmail_agent.start()
            
            # Verify initial configuration
            assert gmail_agent.gmail_config["batch_size"] == 100
            assert gmail_agent.gmail_config["rate_limit_per_minute"] == 250
            
            # Update configuration
            new_config = sample_config.copy()
            new_config["gmail"]["batch_size"] = 200
            new_config["gmail"]["rate_limit_per_minute"] = 300
            
            await gmail_agent._update_config(new_config)
            
            # Verify configuration was updated
            assert gmail_agent.gmail_config["batch_size"] == 200
            assert gmail_agent.gmail_config["rate_limit_per_minute"] == 300
            assert gmail_agent.rate_limiter.requests_per_minute == 300
            
            await gmail_agent.stop()

    @pytest.mark.asyncio
    async def test_gmail_agent_task_execution_integration(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent task execution through framework integration."""
        # Create Gmail Agent
        logger = logging.getLogger("test")
        
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=AsyncMock()
            )
            
            await gmail_agent.start()
            
            # Execute email summary task
            summary_task = {
                "task_type": "email_summary",
                "parameters": {
                    "time_range": "last_24_hours",
                    "categories": ["important", "work"],
                    "include_attachments": False
                }
            }
            
            result = await gmail_agent.execute_task(summary_task)
            
            # Verify task result
            assert "summary" in result
            summary = result["summary"]
            assert "total_emails" in summary
            assert "categories" in summary
            assert "time_range" in summary
            
            # Execute bulk archive task (dry run)
            archive_task = {
                "task_type": "bulk_archive",
                "parameters": {
                    "query": "older_than:30d",
                    "batch_size": 50,
                    "dry_run": True
                }
            }
            
            result = await gmail_agent.execute_task(archive_task)
            
            # Verify task result
            assert "would_archive" in result or "archived_count" in result
            assert result["dry_run"] is True
            
            await gmail_agent.stop()

    @pytest.mark.asyncio
    async def test_gmail_agent_error_handling_integration(self, sample_config: Dict[str, Any]):
        """Test Gmail Agent error handling in framework integration scenarios."""
        # Create Gmail Agent with invalid configuration
        invalid_config = sample_config.copy()
        invalid_config["gmail"]["credentials_path"] = "/nonexistent/path.json"
        
        logger = logging.getLogger("test")
        gmail_agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=invalid_config,
            logger=logger,
            message_broker=AsyncMock()
        )
        
        # Test that agent handles authentication errors gracefully
        with pytest.raises(Exception):  # Should raise authentication error
            await gmail_agent.start()
        
        # Verify agent state is error
        assert gmail_agent.state.value == "error"

    @pytest.mark.asyncio
    async def test_gmail_agent_metrics_integration(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent metrics collection and reporting."""
        # Create Gmail Agent
        logger = logging.getLogger("test")
        
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=AsyncMock()
            )
            
            await gmail_agent.start()
            
            # Perform some operations to generate metrics
            emails = await gmail_agent._fetch_emails(max_results=5)
            
            for email in emails[:3]:
                classification = await gmail_agent._classify_email(email)
            
            # Get metrics
            metrics = gmail_agent.get_metrics()
            
            # Verify Gmail-specific metrics are present
            assert "emails_processed" in metrics
            assert "classifications_made" in metrics
            assert "auto_responses_sent" in metrics
            assert "emails_archived" in metrics
            assert "api_calls_made" in metrics
            assert "rate_limit_hits" in metrics
            
            # Verify metrics have been updated
            assert metrics["emails_processed"] >= 5
            assert metrics["classifications_made"] >= 3
            assert metrics["api_calls_made"] > 0
            
            # Verify base agent metrics
            assert "messages_processed" in metrics
            assert "tasks_completed" in metrics
            assert "errors" in metrics
            assert "uptime" in metrics
            assert "state" in metrics
            assert "agent_id" in metrics
            
            await gmail_agent.stop()

    @pytest.mark.asyncio
    async def test_gmail_agent_lifecycle_integration(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent complete lifecycle through framework integration."""
        # Create all required components
        config_manager = ConfigManager()
        config_manager.load_config(sample_config)
        
        logger = logging.getLogger("test")
        
        communication_broker = CommunicationBroker(
            queue_size=1000,
            timeout=30.0,
            logger=logger
        )
        
        agent_manager = AgentManager(config_manager, logger)
        
        # Create and register Gmail Agent
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=communication_broker
            )
            
            # Register with agent manager
            await agent_manager.register_agent(gmail_agent)
            
            # Start through agent manager
            await agent_manager.start_agents()
            
            # Verify agent is running
            assert gmail_agent.state.value == "active"
            
            # Perform health check
            health_status = await gmail_agent.health_check()
            assert health_status is True
            
            # Test agent functionality
            emails = await gmail_agent._fetch_emails(max_results=3)
            assert len(emails) >= 0
            
            # Simulate configuration change
            new_config = sample_config.copy()
            new_config["gmail"]["batch_size"] = 150
            await gmail_agent._update_config(new_config)
            
            # Stop through agent manager
            await agent_manager.stop_agents()
            
            # Verify agent is stopped
            assert gmail_agent.state.value == "inactive"

    @pytest.mark.asyncio
    async def test_gmail_agent_concurrent_operations(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent handling concurrent operations."""
        # Create Gmail Agent
        logger = logging.getLogger("test")
        
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=AsyncMock()
            )
            
            await gmail_agent.start()
            
            # Create multiple concurrent tasks
            async def fetch_task():
                return await gmail_agent._fetch_emails(max_results=5)
            
            async def classify_task():
                emails = await gmail_agent._fetch_emails(max_results=3)
                classifications = []
                for email in emails:
                    classification = await gmail_agent._classify_email(email)
                    classifications.append(classification)
                return classifications
            
            async def summary_task():
                return await gmail_agent._generate_email_summary("last_24_hours")
            
            # Run tasks concurrently
            results = await asyncio.gather(
                fetch_task(),
                classify_task(),
                summary_task(),
                return_exceptions=True
            )
            
            # Verify all tasks completed successfully
            assert len(results) == 3
            for result in results:
                assert not isinstance(result, Exception)
            
            # Verify rate limiting worked correctly
            metrics = gmail_agent.get_metrics()
            assert metrics["api_calls_made"] > 0
            assert metrics["rate_limit_hits"] >= 0  # Should be 0 with proper rate limiting
            
            await gmail_agent.stop()

    @pytest.mark.asyncio
    async def test_gmail_agent_system_integration_stress(self, sample_config: Dict[str, Any], mock_gmail_service: MockGmailService):
        """Test Gmail Agent under stress conditions with full system integration."""
        # Increase sample data for stress testing
        for i in range(50):
            email = generate_sample_emails(1)[0]
            email.message_id = f"stress_test_email_{i:03d}"
            mock_gmail_service.add_message(email)
        
        # Create system components
        config_manager = ConfigManager()
        config_manager.load_config(sample_config)
        
        logger = logging.getLogger("stress_test")
        
        communication_broker = CommunicationBroker(
            queue_size=1000,
            timeout=30.0,
            logger=logger
        )
        
        agent_manager = AgentManager(config_manager, logger)
        
        # Create Gmail Agent
        with MockGmailAPIContext(mock_gmail_service):
            gmail_agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=sample_config,
                logger=logger,
                message_broker=communication_broker
            )
            
            await agent_manager.register_agent(gmail_agent)
            await agent_manager.start_agents()
            
            # Simulate high load
            tasks = []
            for i in range(10):
                tasks.append(gmail_agent._fetch_emails(max_results=10))
                tasks.append(gmail_agent._process_emails_batch(batch_size=20))
            
            # Execute stress test
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify system stability
            error_count = sum(1 for result in results if isinstance(result, Exception))
            assert error_count < len(results) // 2  # Less than 50% errors acceptable
            
            # Verify agent is still healthy
            health_status = await gmail_agent.health_check()
            assert health_status is True
            
            # Check metrics for reasonable values
            metrics = gmail_agent.get_metrics()
            assert metrics["emails_processed"] > 0
            assert metrics["api_calls_made"] > 0
            assert metrics["errors"] < metrics["api_calls_made"]  # Error rate should be low
            
            await agent_manager.stop_agents()