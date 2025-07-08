"""
Tests for the main application module.

This module contains tests for the main application entry point,
demonstrating TDD principles and proper test structure.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import create_app, Settings, configure_logging, main


class TestSettings:
    """Test cases for the Settings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are properly initialized."""
        settings = Settings()
        
        assert settings.app_name == "Autonomous Agent System"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_settings_with_custom_values(self) -> None:
        """Test that settings can be customized."""
        settings = Settings(
            app_name="Custom App",
            debug=True,
            log_level="DEBUG",
            port=9000,
        )
        
        assert settings.app_name == "Custom App"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.port == 9000


class TestCreateApp:
    """Test cases for the create_app function."""

    def test_create_app_with_default_settings(self) -> None:
        """Test that create_app works with default settings."""
        app = create_app()
        
        assert app.title == "Autonomous Agent System"
        assert app.version == "0.1.0"
        assert app.debug is False

    def test_create_app_with_custom_settings(self) -> None:
        """Test that create_app works with custom settings."""
        settings = Settings(
            app_name="Test App",
            app_version="1.0.0",
            debug=True,
        )
        app = create_app(settings)
        
        assert app.title == "Test App"
        assert app.version == "1.0.0"
        assert app.debug is True


class TestEndpoints:
    """Test cases for the API endpoints."""

    def test_root_endpoint(self, test_app: TestClient) -> None:
        """Test the root endpoint returns expected response."""
        response = test_app.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "Test Autonomous Agent System" in data["message"]

    def test_health_endpoint(self, test_app: TestClient) -> None:
        """Test the health endpoint returns expected response."""
        response = test_app.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["app_name"] == "Test Autonomous Agent System"
        assert data["version"] == "0.1.0-test"


class TestConfigureLogging:
    """Test cases for the configure_logging function."""

    @patch('src.main.logging.basicConfig')
    def test_configure_logging_default(self, mock_basic_config) -> None:
        """Test that configure_logging works with default level."""
        configure_logging()
        
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs['level'] == 20  # INFO level

    @patch('src.main.logging.basicConfig')
    def test_configure_logging_custom_level(self, mock_basic_config) -> None:
        """Test that configure_logging works with custom level."""
        configure_logging("DEBUG")
        
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs['level'] == 10  # DEBUG level


class TestMain:
    """Test cases for the main function."""

    @pytest.mark.asyncio
    async def test_main_initialization(self) -> None:
        """Test that main function initializes properly."""
        with patch('src.main.uvicorn.Server') as mock_server:
            mock_server_instance = AsyncMock()
            mock_server.return_value = mock_server_instance
            
            with patch('src.main.Settings') as mock_settings:
                mock_settings.return_value = Settings()
                
                with patch('src.main.configure_logging') as mock_logging:
                    # Mock server.serve to raise KeyboardInterrupt for clean exit
                    mock_server_instance.serve.side_effect = KeyboardInterrupt()
                    
                    await main()
                    
                    mock_logging.assert_called_once()
                    mock_server.assert_called_once()
                    mock_server_instance.serve.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_handles_keyboard_interrupt(self) -> None:
        """Test that main function handles KeyboardInterrupt gracefully."""
        with patch('src.main.uvicorn.Server') as mock_server:
            mock_server_instance = AsyncMock()
            mock_server.return_value = mock_server_instance
            mock_server_instance.serve.side_effect = KeyboardInterrupt()
            
            with patch('src.main.configure_logging'):
                # Should not raise an exception
                await main()

    @pytest.mark.asyncio
    async def test_main_handles_unexpected_error(self) -> None:
        """Test that main function handles unexpected errors properly."""
        with patch('src.main.uvicorn.Server') as mock_server:
            mock_server_instance = AsyncMock()
            mock_server.return_value = mock_server_instance
            mock_server_instance.serve.side_effect = RuntimeError("Test error")
            
            with patch('src.main.configure_logging'):
                with pytest.raises(RuntimeError, match="Test error"):
                    await main()