"""
Sample unit tests demonstrating TDD approach for agent components.

This module shows how to write tests following Test-Driven Development principles:
1. Write failing tests first (Red)
2. Write minimal code to make tests pass (Green)  
3. Refactor code while keeping tests passing (Refactor)

This example shows testing patterns for a hypothetical BaseAgent class.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from tests.utils.test_helpers import TestDataGenerator, AsyncTestHelper
from tests.utils.assertions import (
    assert_agent_config_valid,
    assert_valid_uuid,
    assert_positive_number,
    assert_string_not_empty,
)


# ============================================================================
# Test-Driven Development Example: BaseAgent
# ============================================================================

class TestBaseAgent:
    """
    Test suite for BaseAgent following TDD principles.
    
    This demonstrates the Red-Green-Refactor cycle:
    1. Write tests that describe the desired behavior
    2. Implement minimal code to pass tests
    3. Refactor implementation while keeping tests green
    """
    
    # RED: Write failing tests first
    
    @pytest.mark.unit
    def test_agent_creation_requires_valid_config(self):
        """Test that agent creation requires a valid configuration."""
        # ARRANGE
        valid_config = TestDataGenerator.generate_agent_config()
        
        # ACT & ASSERT
        assert_agent_config_valid(valid_config)
        
        # Test that invalid configs are rejected
        invalid_configs = [
            {},  # Empty config
            {"agent_id": "invalid"},  # Missing required fields
            {"agent_id": "123", "agent_type": "", "max_retries": -1, "timeout": 0},  # Invalid values
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(AssertionError):
                assert_agent_config_valid(invalid_config)
    
    @pytest.mark.unit
    def test_agent_has_unique_id_on_creation(self):
        """Test that each agent gets a unique ID when created."""
        # This test would fail initially, driving us to implement ID generation
        
        # ARRANGE
        config1 = TestDataGenerator.generate_agent_config()
        config2 = TestDataGenerator.generate_agent_config()
        
        # ACT
        agent_id1 = config1["agent_id"]
        agent_id2 = config2["agent_id"]
        
        # ASSERT
        assert_valid_uuid(agent_id1)
        assert_valid_uuid(agent_id2)
        assert agent_id1 != agent_id2, "Agent IDs should be unique"
    
    @pytest.mark.unit
    async def test_agent_starts_and_stops_correctly(self):
        """Test that agent can be started and stopped."""
        # RED: This test would fail until we implement start/stop methods
        
        # ARRANGE
        mock_agent = self._create_mock_agent()
        
        # ACT
        start_result = await mock_agent.start()
        status_after_start = await mock_agent.get_status()
        
        stop_result = await mock_agent.stop()
        status_after_stop = await mock_agent.get_status()
        
        # ASSERT
        assert start_result is True, "Agent should start successfully"
        assert status_after_start == "running", "Agent should be running after start"
        
        assert stop_result is True, "Agent should stop successfully"
        assert status_after_stop == "stopped", "Agent should be stopped after stop"
    
    @pytest.mark.unit
    async def test_agent_handles_heartbeat(self):
        """Test that agent responds to heartbeat requests."""
        # RED: Drives implementation of heartbeat mechanism
        
        # ARRANGE
        mock_agent = self._create_mock_agent()
        await mock_agent.start()
        
        # ACT
        heartbeat_response = await mock_agent.heartbeat()
        
        # ASSERT
        assert heartbeat_response is not None, "Heartbeat should return a response"
        assert "timestamp" in heartbeat_response, "Heartbeat should include timestamp"
        assert "status" in heartbeat_response, "Heartbeat should include status"
        assert heartbeat_response["status"] == "running", "Status should be running"
    
    @pytest.mark.unit
    async def test_agent_processes_messages(self):
        """Test that agent can process messages."""
        # RED: Drives implementation of message processing
        
        # ARRANGE
        mock_agent = self._create_mock_agent()
        await mock_agent.start()
        
        test_message = {
            "id": TestDataGenerator.generate_uuid(),
            "type": "test_message",
            "payload": {"data": "test data"},
            "timestamp": TestDataGenerator.generate_timestamp(),
        }
        
        # ACT
        result = await mock_agent.process_message(test_message)
        
        # ASSERT
        assert result is not None, "Message processing should return a result"
        assert "status" in result, "Result should include status"
        assert "message_id" in result, "Result should include message ID"
        assert result["message_id"] == test_message["id"], "Should reference original message"
    
    @pytest.mark.unit
    async def test_agent_handles_errors_gracefully(self):
        """Test that agent handles errors without crashing."""
        # RED: Drives implementation of error handling
        
        # ARRANGE
        mock_agent = self._create_mock_agent()
        await mock_agent.start()
        
        # Simulate an error condition
        invalid_message = {"invalid": "message"}
        
        # ACT
        result = await mock_agent.process_message(invalid_message)
        
        # ASSERT
        assert result is not None, "Should return result even for invalid messages"
        assert "error" in result, "Result should indicate error"
        assert "status" in result, "Result should include status"
        assert result["status"] == "error", "Status should be error"
        
        # Agent should still be running after error
        status = await mock_agent.get_status()
        assert status == "running", "Agent should still be running after error"
    
    @pytest.mark.unit
    async def test_agent_retries_failed_operations(self):
        """Test that agent retries operations according to configuration."""
        # RED: Drives implementation of retry logic
        
        # ARRANGE
        config = TestDataGenerator.generate_agent_config()
        config["max_retries"] = 3
        
        mock_agent = self._create_mock_agent_with_config(config)
        await mock_agent.start()
        
        # Mock an operation that fails twice then succeeds
        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            {"status": "success"}  # Success on third try
        ]
        
        # ACT
        result = await mock_agent._retry_operation(mock_operation)
        
        # ASSERT
        assert result["status"] == "success", "Should succeed after retries"
        assert mock_operation.call_count == 3, "Should retry according to config"
    
    @pytest.mark.unit
    async def test_agent_respects_timeout_configuration(self):
        """Test that agent respects timeout settings."""
        # RED: Drives implementation of timeout handling
        
        # ARRANGE
        config = TestDataGenerator.generate_agent_config()
        config["timeout"] = 1.0  # 1 second timeout
        
        mock_agent = self._create_mock_agent_with_config(config)
        await mock_agent.start()
        
        # Mock a slow operation
        async def slow_operation():
            await asyncio.sleep(2.0)  # Takes 2 seconds
            return {"status": "success"}
        
        # ACT & ASSERT
        with pytest.raises(asyncio.TimeoutError):
            await mock_agent._execute_with_timeout(slow_operation())
    
    # GREEN: Helper methods to create mocks (minimal implementation)
    
    def _create_mock_agent(self) -> AsyncMock:
        """Create a mock agent with basic behavior."""
        agent = AsyncMock()
        
        # Mock basic agent methods
        agent.start.return_value = True
        agent.stop.return_value = True
        agent.get_status.return_value = "running"
        
        # Mock heartbeat
        agent.heartbeat.return_value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "uptime": 0,
        }
        
        # Mock message processing
        async def mock_process_message(message):
            if "id" not in message:
                return {"status": "error", "error": "Invalid message format"}
            return {
                "status": "processed",
                "message_id": message["id"],
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        
        agent.process_message.side_effect = mock_process_message
        
        # Mock retry operation
        async def mock_retry_operation(operation):
            try:
                return await operation()
            except Exception:
                return await operation()  # Simple retry once
        
        agent._retry_operation.side_effect = mock_retry_operation
        
        # Mock timeout handling
        async def mock_execute_with_timeout(coro, timeout=30.0):
            return await asyncio.wait_for(coro, timeout=timeout)
        
        agent._execute_with_timeout.side_effect = mock_execute_with_timeout
        
        return agent
    
    def _create_mock_agent_with_config(self, config: Dict[str, Any]) -> AsyncMock:
        """Create a mock agent with specific configuration."""
        agent = self._create_mock_agent()
        agent.config = config
        return agent


# ============================================================================
# TDD Example: EmailProcessor Component
# ============================================================================

class TestEmailProcessor:
    """
    Example of TDD for an email processing component.
    
    This shows how to test email classification, filtering, and response generation.
    """
    
    @pytest.mark.unit
    def test_email_classification_requires_valid_email_data(self, sample_email_data):
        """Test that email classification validates input data."""
        # RED: Write test for validation first
        
        from tests.utils.assertions import assert_email_data_valid
        
        # ARRANGE - valid email data
        valid_email = sample_email_data
        
        # ACT & ASSERT
        assert_email_data_valid(valid_email)
        
        # Test invalid email data
        invalid_emails = [
            {},  # Empty
            {"from": "invalid-email", "to": "also-invalid"},  # Invalid email format
            {"from": "test@example.com", "to": "user@example.com", "subject": ""},  # Empty subject
        ]
        
        for invalid_email in invalid_emails:
            with pytest.raises(AssertionError):
                assert_email_data_valid(invalid_email)
    
    @pytest.mark.unit
    async def test_email_classification_returns_category_and_confidence(self, sample_email_data):
        """Test that email classification returns category and confidence score."""
        # RED: Define the interface we want
        
        # ARRANGE
        email_processor = self._create_mock_email_processor()
        email_data = sample_email_data
        
        # ACT
        classification = await email_processor.classify_email(email_data)
        
        # ASSERT
        assert "category" in classification, "Should return category"
        assert "confidence" in classification, "Should return confidence score"
        assert isinstance(classification["confidence"], float), "Confidence should be float"
        assert 0.0 <= classification["confidence"] <= 1.0, "Confidence should be between 0 and 1"
        
        valid_categories = ["inbox", "spam", "promotional", "social", "important"]
        assert classification["category"] in valid_categories, f"Category should be one of {valid_categories}"
    
    @pytest.mark.unit
    async def test_spam_detection_identifies_spam_emails(self):
        """Test that spam detection correctly identifies spam emails."""
        # RED: Drive spam detection implementation
        
        # ARRANGE
        email_processor = self._create_mock_email_processor()
        
        spam_email = {
            "from": "spam@suspicious.com",
            "to": "user@example.com",
            "subject": "URGENT: You've won $1,000,000!",
            "body": "Click here to claim your prize! Don't delay!",
            "date": TestDataGenerator.generate_timestamp(),
        }
        
        legitimate_email = {
            "from": "colleague@work.com",
            "to": "user@example.com",
            "subject": "Meeting tomorrow",
            "body": "Just wanted to confirm our meeting at 2 PM tomorrow.",
            "date": TestDataGenerator.generate_timestamp(),
        }
        
        # ACT
        spam_classification = await email_processor.classify_email(spam_email)
        legit_classification = await email_processor.classify_email(legitimate_email)
        
        # ASSERT
        assert spam_classification["category"] == "spam", "Should classify spam as spam"
        assert spam_classification["confidence"] > 0.7, "Should have high confidence for obvious spam"
        
        assert legit_classification["category"] != "spam", "Should not classify legitimate email as spam"
        assert legit_classification["confidence"] > 0.5, "Should have reasonable confidence"
    
    @pytest.mark.unit
    async def test_email_response_generation_creates_appropriate_responses(self, sample_email_data):
        """Test that response generation creates contextually appropriate responses."""
        # RED: Drive response generation implementation
        
        # ARRANGE
        email_processor = self._create_mock_email_processor()
        email_data = sample_email_data
        
        # ACT
        response = await email_processor.generate_response(email_data)
        
        # ASSERT
        assert "subject" in response, "Response should have subject"
        assert "body" in response, "Response should have body"
        assert "type" in response, "Response should have type"
        
        assert_string_not_empty(response["subject"], "Subject should not be empty")
        assert_string_not_empty(response["body"], "Body should not be empty")
        
        valid_response_types = ["acknowledgment", "auto_reply", "forward", "decline"]
        assert response["type"] in valid_response_types, f"Type should be one of {valid_response_types}"
    
    def _create_mock_email_processor(self) -> AsyncMock:
        """Create a mock email processor with realistic behavior."""
        processor = AsyncMock()
        
        # Mock email classification
        async def mock_classify_email(email_data):
            # Simple rule-based classification for testing
            subject = email_data.get("subject", "").lower()
            body = email_data.get("body", "").lower()
            
            if any(word in subject + body for word in ["urgent", "winner", "prize", "claim"]):
                return {"category": "spam", "confidence": 0.9}
            elif any(word in subject + body for word in ["meeting", "project", "work"]):
                return {"category": "important", "confidence": 0.8}
            else:
                return {"category": "inbox", "confidence": 0.6}
        
        processor.classify_email.side_effect = mock_classify_email
        
        # Mock response generation
        async def mock_generate_response(email_data):
            return {
                "subject": f"Re: {email_data.get('subject', 'No Subject')}",
                "body": "Thank you for your email. I will review it and respond accordingly.",
                "type": "acknowledgment",
            }
        
        processor.generate_response.side_effect = mock_generate_response
        
        return processor


# ============================================================================
# TDD Example: Configuration Manager
# ============================================================================

class TestConfigurationManager:
    """
    Example of TDD for configuration management.
    
    Shows testing of configuration loading, validation, and environment handling.
    """
    
    @pytest.mark.unit
    def test_config_manager_loads_from_file(self, temp_dir):
        """Test that configuration manager can load from YAML file."""
        # RED: Drive file loading implementation
        
        # ARRANGE
        config_data = TestDataGenerator.generate_agent_config()
        config_file = temp_dir / "test_config.yaml"
        
        import yaml
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config_manager = self._create_mock_config_manager()
        
        # ACT
        loaded_config = config_manager.load_from_file(str(config_file))
        
        # ASSERT
        assert loaded_config == config_data, "Should load exact configuration from file"
    
    @pytest.mark.unit
    def test_config_manager_validates_required_fields(self):
        """Test that configuration manager validates required fields."""
        # RED: Drive validation implementation
        
        # ARRANGE
        config_manager = self._create_mock_config_manager()
        
        valid_config = {
            "agent_manager": {"max_workers": 4},
            "logging": {"level": "INFO"},
        }
        
        invalid_config = {
            "logging": {"level": "INFO"}
            # Missing agent_manager section
        }
        
        # ACT & ASSERT
        assert config_manager.validate(valid_config) is True, "Should validate correct config"
        
        with pytest.raises(ValueError):
            config_manager.validate(invalid_config)
    
    @pytest.mark.unit
    def test_config_manager_handles_environment_overrides(self, monkeypatch):
        """Test that configuration manager handles environment variable overrides."""
        # RED: Drive environment override implementation
        
        # ARRANGE
        base_config = {
            "database": {"host": "localhost", "port": 5432},
            "logging": {"level": "INFO"},
        }
        
        # Set environment variables
        monkeypatch.setenv("DATABASE_HOST", "prod-db.example.com")
        monkeypatch.setenv("DATABASE_PORT", "5433")
        monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
        
        config_manager = self._create_mock_config_manager()
        
        # ACT
        final_config = config_manager.apply_environment_overrides(base_config)
        
        # ASSERT
        assert final_config["database"]["host"] == "prod-db.example.com"
        assert final_config["database"]["port"] == 5433  # Should be converted to int
        assert final_config["logging"]["level"] == "DEBUG"
    
    def _create_mock_config_manager(self) -> MagicMock:
        """Create a mock configuration manager."""
        manager = MagicMock()
        
        # Mock file loading
        def mock_load_from_file(file_path):
            import yaml
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        
        manager.load_from_file.side_effect = mock_load_from_file
        
        # Mock validation
        def mock_validate(config):
            required_sections = ["agent_manager", "logging"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            return True
        
        manager.validate.side_effect = mock_validate
        
        # Mock environment overrides
        def mock_apply_environment_overrides(config):
            import os
            result = config.copy()
            
            # Simple environment override logic
            env_mappings = {
                "DATABASE_HOST": ("database", "host"),
                "DATABASE_PORT": ("database", "port"),
                "LOGGING_LEVEL": ("logging", "level"),
            }
            
            for env_var, (section, key) in env_mappings.items():
                if env_var in os.environ:
                    value = os.environ[env_var]
                    # Convert port to int
                    if key == "port":
                        value = int(value)
                    
                    if section not in result:
                        result[section] = {}
                    result[section][key] = value
            
            return result
        
        manager.apply_environment_overrides.side_effect = mock_apply_environment_overrides
        
        return manager


# ============================================================================
# REFACTOR: Demonstrate test refactoring
# ============================================================================

class TestRefactoringExample:
    """
    Example showing how to refactor tests while keeping them green.
    
    This demonstrates the 'Refactor' phase of Red-Green-Refactor.
    """
    
    # Before refactoring: Repetitive test setup
    @pytest.mark.unit
    def test_agent_processing_before_refactor(self):
        """Example of test before refactoring - lots of duplication."""
        # ARRANGE - lots of repetitive setup
        agent_id = TestDataGenerator.generate_uuid()
        config = {
            "agent_id": agent_id,
            "agent_type": "test",
            "max_retries": 3,
            "timeout": 30.0,
        }
        
        mock_agent = AsyncMock()
        mock_agent.config = config
        mock_agent.process_message.return_value = {"status": "processed"}
        
        message = {
            "id": TestDataGenerator.generate_uuid(),
            "type": "test",
            "payload": {"data": "test"},
        }
        
        # ACT
        result = mock_agent.process_message(message)
        
        # ASSERT
        assert result["status"] == "processed"
    
    # After refactoring: Clean, reusable setup
    @pytest.mark.unit
    def test_agent_processing_after_refactor(self, configured_mock_agent, test_message):
        """Example of test after refactoring - clean and focused."""
        # ARRANGE - simplified with fixtures
        
        # ACT
        result = configured_mock_agent.process_message(test_message)
        
        # ASSERT
        assert result["status"] == "processed"
    
    # Fixtures created during refactoring
    @pytest.fixture
    def configured_mock_agent(self):
        """Fixture providing a properly configured mock agent."""
        config = TestDataGenerator.generate_agent_config()
        
        mock_agent = AsyncMock()
        mock_agent.config = config
        mock_agent.process_message.return_value = {"status": "processed"}
        
        return mock_agent
    
    @pytest.fixture
    def test_message(self):
        """Fixture providing a test message."""
        return {
            "id": TestDataGenerator.generate_uuid(),
            "type": "test",
            "payload": {"data": "test"},
            "timestamp": TestDataGenerator.generate_timestamp(),
        }