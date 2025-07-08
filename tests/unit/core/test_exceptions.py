"""
Tests for the core.exceptions module.

This module contains tests for custom exceptions used throughout the system.
"""

import pytest

from src.core.exceptions import (
    AgentError,
    ServiceError,
    ConfigurationError,
    ValidationError,
)


class TestAgentError:
    """Test cases for the AgentError exception."""

    def test_agent_error_with_message_only(self) -> None:
        """Test AgentError with message only."""
        error = AgentError("Test error message")
        
        assert error.message == "Test error message"
        assert error.agent_name == "Unknown"
        assert str(error) == "Agent 'Unknown': Test error message"

    def test_agent_error_with_message_and_agent_name(self) -> None:
        """Test AgentError with message and agent name."""
        error = AgentError("Test error message", "TestAgent")
        
        assert error.message == "Test error message"
        assert error.agent_name == "TestAgent"
        assert str(error) == "Agent 'TestAgent': Test error message"

    def test_agent_error_inheritance(self) -> None:
        """Test that AgentError inherits from Exception."""
        error = AgentError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, AgentError)

    def test_agent_error_can_be_raised(self) -> None:
        """Test that AgentError can be raised and caught."""
        with pytest.raises(AgentError) as exc_info:
            raise AgentError("Test error", "TestAgent")
        
        assert exc_info.value.message == "Test error"
        assert exc_info.value.agent_name == "TestAgent"


class TestServiceError:
    """Test cases for the ServiceError exception."""

    def test_service_error_with_message_only(self) -> None:
        """Test ServiceError with message only."""
        error = ServiceError("Test service error")
        
        assert error.message == "Test service error"
        assert error.service_name == "Unknown"
        assert str(error) == "Service 'Unknown': Test service error"

    def test_service_error_with_message_and_service_name(self) -> None:
        """Test ServiceError with message and service name."""
        error = ServiceError("Test service error", "TestService")
        
        assert error.message == "Test service error"
        assert error.service_name == "TestService"
        assert str(error) == "Service 'TestService': Test service error"

    def test_service_error_inheritance(self) -> None:
        """Test that ServiceError inherits from Exception."""
        error = ServiceError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ServiceError)

    def test_service_error_can_be_raised(self) -> None:
        """Test that ServiceError can be raised and caught."""
        with pytest.raises(ServiceError) as exc_info:
            raise ServiceError("Test error", "TestService")
        
        assert exc_info.value.message == "Test error"
        assert exc_info.value.service_name == "TestService"


class TestConfigurationError:
    """Test cases for the ConfigurationError exception."""

    def test_configuration_error_with_message_only(self) -> None:
        """Test ConfigurationError with message only."""
        error = ConfigurationError("Invalid configuration")
        
        assert error.message == "Invalid configuration"
        assert error.config_key == "Unknown"
        assert str(error) == "Configuration error for 'Unknown': Invalid configuration"

    def test_configuration_error_with_message_and_config_key(self) -> None:
        """Test ConfigurationError with message and config key."""
        error = ConfigurationError("Invalid value", "database.host")
        
        assert error.message == "Invalid value"
        assert error.config_key == "database.host"
        assert str(error) == "Configuration error for 'database.host': Invalid value"

    def test_configuration_error_inheritance(self) -> None:
        """Test that ConfigurationError inherits from Exception."""
        error = ConfigurationError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_can_be_raised(self) -> None:
        """Test that ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Test error", "test.key")
        
        assert exc_info.value.message == "Test error"
        assert exc_info.value.config_key == "test.key"


class TestValidationError:
    """Test cases for the ValidationError exception."""

    def test_validation_error_with_message_only(self) -> None:
        """Test ValidationError with message only."""
        error = ValidationError("Invalid data")
        
        assert error.message == "Invalid data"
        assert error.field_name == "Unknown"
        assert str(error) == "Validation error for 'Unknown': Invalid data"

    def test_validation_error_with_message_and_field_name(self) -> None:
        """Test ValidationError with message and field name."""
        error = ValidationError("Must be positive", "age")
        
        assert error.message == "Must be positive"
        assert error.field_name == "age"
        assert str(error) == "Validation error for 'age': Must be positive"

    def test_validation_error_inheritance(self) -> None:
        """Test that ValidationError inherits from Exception."""
        error = ValidationError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ValidationError)

    def test_validation_error_can_be_raised(self) -> None:
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test error", "test_field")
        
        assert exc_info.value.message == "Test error"
        assert exc_info.value.field_name == "test_field"


class TestExceptionInteraction:
    """Test cases for exception interactions and edge cases."""

    def test_exceptions_are_distinct(self) -> None:
        """Test that different exception types are distinct."""
        agent_error = AgentError("Test")
        service_error = ServiceError("Test")
        config_error = ConfigurationError("Test")
        validation_error = ValidationError("Test")
        
        assert type(agent_error) != type(service_error)
        assert type(service_error) != type(config_error)
        assert type(config_error) != type(validation_error)

    def test_exceptions_can_be_caught_separately(self) -> None:
        """Test that exceptions can be caught separately."""
        # Test AgentError
        with pytest.raises(AgentError):
            raise AgentError("Test")
        
        # Test ServiceError
        with pytest.raises(ServiceError):
            raise ServiceError("Test")
        
        # Test ConfigurationError
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test")
        
        # Test ValidationError
        with pytest.raises(ValidationError):
            raise ValidationError("Test")

    def test_exceptions_can_be_caught_as_base_exception(self) -> None:
        """Test that custom exceptions can be caught as base Exception."""
        with pytest.raises(Exception):
            raise AgentError("Test")
        
        with pytest.raises(Exception):
            raise ServiceError("Test")
        
        with pytest.raises(Exception):
            raise ConfigurationError("Test")
        
        with pytest.raises(Exception):
            raise ValidationError("Test")