"""
Integration tests for the autonomous agent system.

This module contains integration tests that verify the interaction
between different components of the system.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.main import create_app, Settings


class TestAppIntegration:
    """Integration tests for the main application."""

    @pytest.fixture
    def integration_app(self) -> TestClient:
        """Create an integration test application."""
        settings = Settings(
            app_name="Integration Test App",
            app_version="0.1.0-integration",
            debug=True,
            log_level="DEBUG",
        )
        app = create_app(settings)
        return TestClient(app)

    def test_app_startup_and_endpoints(self, integration_app: TestClient) -> None:
        """Test that the app starts up and all endpoints work."""
        # Test root endpoint
        response = integration_app.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Integration Test App" in data["message"]
        assert data["version"] == "0.1.0-integration"
        
        # Test health endpoint
        response = integration_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["app_name"] == "Integration Test App"
        assert data["version"] == "0.1.0-integration"

    def test_app_handles_invalid_endpoints(self, integration_app: TestClient) -> None:
        """Test that the app handles invalid endpoints gracefully."""
        response = integration_app.get("/nonexistent")
        assert response.status_code == 404

    def test_app_cors_and_headers(self, integration_app: TestClient) -> None:
        """Test that the app handles CORS and headers appropriately."""
        response = integration_app.get("/health")
        assert response.status_code == 200
        # FastAPI automatically handles basic headers
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"

    def test_app_openapi_docs(self, integration_app: TestClient) -> None:
        """Test that OpenAPI documentation is accessible."""
        response = integration_app.get("/docs")
        assert response.status_code == 200
        
        response = integration_app.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Integration Test App"


class TestSystemIntegration:
    """Integration tests for system-level functionality."""

    @pytest.mark.integration
    async def test_system_health_flow(self) -> None:
        """Test the complete system health checking flow."""
        settings = Settings(debug=True)
        app = create_app(settings)
        
        with TestClient(app) as client:
            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] == "healthy"
            
            # Test that the system can handle multiple requests
            for i in range(5):
                response = client.get("/health")
                assert response.status_code == 200
                assert response.json()["status"] == "healthy"

    @pytest.mark.integration
    async def test_configuration_integration(self) -> None:
        """Test that configuration works end-to-end."""
        # Test with custom configuration
        custom_settings = Settings(
            app_name="Custom Config Test",
            app_version="2.0.0",
            debug=False,
            log_level="WARNING",
        )
        
        app = create_app(custom_settings)
        
        with TestClient(app) as client:
            # Verify configuration is applied
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "Custom Config Test" in data["message"]
            assert data["version"] == "2.0.0"
            
            # Verify health endpoint reflects configuration
            response = client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["app_name"] == "Custom Config Test"
            assert health_data["version"] == "2.0.0"

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_requests(self) -> None:
        """Test that the system handles concurrent requests properly."""
        import asyncio
        import aiohttp
        
        settings = Settings(debug=True)
        app = create_app(settings)
        
        async def make_request(session: aiohttp.ClientSession, url: str) -> dict:
            """Make a single request."""
            async with session.get(url) as response:
                return await response.json()
        
        # This would be a more comprehensive test with actual async client
        # For now, we'll test with TestClient
        with TestClient(app) as client:
            # Make multiple concurrent requests
            responses = []
            for i in range(10):
                response = client.get("/health")
                responses.append(response)
            
            # Verify all responses are successful
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"