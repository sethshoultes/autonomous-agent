"""
Test suite for the Code Agent implementation.

This test suite follows TDD principles and tests all aspects of the Code Agent:
- GitHub API integration
- Code review automation
- Pull request analysis
- Repository monitoring
- AI-powered code analysis
- Development workflow automation
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

import pytest

from src.agents.base import AgentMessage, AgentState, BaseAgent
from src.agents.code_agent import CodeAgent
from src.agents.exceptions import AgentError, AgentStateError
from src.agents.manager import AgentConfig
from src.services.github_service import GitHubService, GitHubError
from src.services.ai_code_analyzer import AICodeAnalyzer
from tests.mocks.github_mocks import (
    MockGitHubClient,
    MockGitHubPullRequest,
    MockGitHubRepository,
    MockGitHubIssue,
    MockGitHubCommit,
    MockGitHubWebhook,
    generate_pr_webhook_payload,
    generate_issue_webhook_payload,
    mock_github_api,
)
from tests.mocks.ollama_mocks import MockOllamaService


class TestCodeAgent:
    """Test suite for the Code Agent."""
    
    @pytest.fixture
    def mock_logger(self):
        """Provide a mock logger."""
        return MagicMock(spec=logging.Logger)
    
    @pytest.fixture
    def mock_message_broker(self):
        """Provide a mock message broker."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        broker.subscribe = AsyncMock()
        broker.disconnect = AsyncMock()
        return broker
    
    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration."""
        return {
            "github": {
                "token": "test_token",
                "webhook_secret": "test_secret",
                "base_url": "https://api.github.com",
                "timeout": 30,
                "max_retries": 3,
            },
            "ai_analysis": {
                "enabled": True,
                "model": "codellama:7b",
                "temperature": 0.2,
                "max_context_length": 8192,
            },
            "repository_monitoring": {
                "enabled": True,
                "events": ["push", "pull_request", "issues"],
                "polling_interval": 300,
            },
            "code_review": {
                "auto_review": True,
                "min_confidence": 0.7,
                "review_categories": ["style", "security", "performance", "bugs"],
            },
            "workflow_automation": {
                "enabled": True,
                "auto_merge": False,
                "auto_deploy": False,
                "ci_integration": True,
            },
        }
    
    @pytest.fixture
    def mock_github_service(self):
        """Provide a mock GitHub service."""
        service = MagicMock(spec=GitHubService)
        service.connect = AsyncMock()
        service.disconnect = AsyncMock()
        service.get_repository = AsyncMock()
        service.get_pull_request = AsyncMock()
        service.get_pull_request_files = AsyncMock()
        service.create_pr_comment = AsyncMock()
        service.create_pr_review = AsyncMock()
        service.get_commits = AsyncMock()
        service.get_issues = AsyncMock()
        service.create_issue = AsyncMock()
        service.close_issue = AsyncMock()
        service.set_webhook = AsyncMock()
        service.health_check = AsyncMock(return_value={"status": "healthy"})
        return service
    
    @pytest.fixture
    def mock_ai_analyzer(self):
        """Provide a mock AI code analyzer."""
        analyzer = MagicMock(spec=AICodeAnalyzer)
        analyzer.analyze_code = AsyncMock()
        analyzer.review_pull_request = AsyncMock()
        analyzer.detect_vulnerabilities = AsyncMock()
        analyzer.suggest_improvements = AsyncMock()
        analyzer.generate_documentation = AsyncMock()
        return analyzer
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Provide a mock Ollama service."""
        return MockOllamaService()
    
    @pytest.fixture
    def code_agent(self, mock_config, mock_logger, mock_message_broker, 
                   mock_github_service, mock_ai_analyzer, mock_ollama_service):
        """Provide a Code Agent instance."""
        with patch('src.agents.code_agent.GitHubService', return_value=mock_github_service), \
             patch('src.agents.code_agent.AICodeAnalyzer', return_value=mock_ai_analyzer), \
             patch('src.agents.code_agent.OllamaService', return_value=mock_ollama_service):
            
            agent = CodeAgent(
                agent_id="test_code_agent",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker
            )
            
            # Inject mocks
            agent.github_service = mock_github_service
            agent.ai_analyzer = mock_ai_analyzer
            agent.ollama_service = mock_ollama_service
            
            return agent
    
    @pytest.mark.asyncio
    async def test_code_agent_initialization(self, code_agent):
        """Test Code Agent initialization."""
        assert code_agent.agent_id == "test_code_agent"
        assert code_agent.state == AgentState.INACTIVE
        assert code_agent.config["github"]["token"] == "test_token"
        assert code_agent.config["ai_analysis"]["enabled"] is True
        assert code_agent.monitored_repositories == []
        assert code_agent.active_reviews == {}
        assert code_agent.webhook_handlers is not None
    
    @pytest.mark.asyncio
    async def test_code_agent_start_success(self, code_agent):
        """Test successful Code Agent startup."""
        # Mock the initialization methods
        code_agent.github_service.connect = AsyncMock()
        code_agent.ollama_service.connect = AsyncMock()
        code_agent._load_monitored_repositories = AsyncMock()
        code_agent._setup_webhook_handlers = AsyncMock()
        code_agent._start_repository_monitoring = AsyncMock()
        
        await code_agent.start()
        
        assert code_agent.state == AgentState.ACTIVE
        code_agent.github_service.connect.assert_called_once()
        code_agent.ollama_service.connect.assert_called_once()
        code_agent._load_monitored_repositories.assert_called_once()
        code_agent._setup_webhook_handlers.assert_called_once()
        code_agent._start_repository_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_code_agent_start_failure(self, code_agent):
        """Test Code Agent startup failure."""
        # Mock a connection failure
        code_agent.github_service.connect = AsyncMock(side_effect=GitHubError("Connection failed"))
        
        with pytest.raises(AgentError, match="Failed to start agent"):
            await code_agent.start()
        
        assert code_agent.state == AgentState.ERROR
    
    @pytest.mark.asyncio
    async def test_code_agent_stop_success(self, code_agent):
        """Test successful Code Agent shutdown."""
        # Start the agent first
        code_agent.state = AgentState.ACTIVE
        
        # Mock cleanup methods
        code_agent._stop_repository_monitoring = AsyncMock()
        code_agent.github_service.disconnect = AsyncMock()
        code_agent.ollama_service.disconnect = AsyncMock()
        
        await code_agent.stop()
        
        assert code_agent.state == AgentState.INACTIVE
        code_agent._stop_repository_monitoring.assert_called_once()
        code_agent.github_service.disconnect.assert_called_once()
        code_agent.ollama_service.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, code_agent):
        """Test health check when agent is healthy."""
        code_agent.state = AgentState.ACTIVE
        code_agent.github_service.health_check = AsyncMock(return_value={"status": "healthy"})
        code_agent.ollama_service.health_check = AsyncMock(return_value={"status": "healthy"})
        
        result = await code_agent.health_check()
        
        assert result is True
        code_agent.github_service.health_check.assert_called_once()
        code_agent.ollama_service.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, code_agent):
        """Test health check when agent is unhealthy."""
        code_agent.state = AgentState.ACTIVE
        code_agent.github_service.health_check = AsyncMock(return_value={"status": "unhealthy"})
        
        result = await code_agent.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_process_pull_request_webhook(self, code_agent):
        """Test processing pull request webhook events."""
        # Setup test data
        payload = generate_pr_webhook_payload("opened")
        webhook_message = AgentMessage(
            id="test_msg_1",
            sender="webhook_handler",
            recipient="test_code_agent",
            message_type="github_webhook",
            payload={
                "event": "pull_request",
                "action": "opened",
                "data": payload
            }
        )
        
        # Mock methods
        code_agent.ai_analyzer.review_pull_request = AsyncMock(return_value={
            "review_id": "review_123",
            "comments": [
                {
                    "file": "test.py",
                    "line": 10,
                    "message": "Consider using a more descriptive variable name",
                    "category": "style",
                    "confidence": 0.8
                }
            ],
            "summary": "Code looks good with minor style improvements needed",
            "approve": True
        })
        
        code_agent.github_service.create_pr_review = AsyncMock()
        
        # Process the webhook
        response = await code_agent._process_message(webhook_message)
        
        # Verify processing
        code_agent.ai_analyzer.review_pull_request.assert_called_once()
        code_agent.github_service.create_pr_review.assert_called_once()
        assert response is not None
        assert response.message_type == "webhook_processed"
    
    @pytest.mark.asyncio
    async def test_process_issue_webhook(self, code_agent):
        """Test processing issue webhook events."""
        payload = generate_issue_webhook_payload("opened")
        webhook_message = AgentMessage(
            id="test_msg_2",
            sender="webhook_handler",
            recipient="test_code_agent",
            message_type="github_webhook",
            payload={
                "event": "issues",
                "action": "opened",
                "data": payload
            }
        )
        
        # Mock AI analysis
        code_agent.ai_analyzer.analyze_issue = AsyncMock(return_value={
            "category": "bug",
            "priority": "high",
            "tags": ["authentication", "security"],
            "suggested_assignees": ["dev-team-lead"],
            "estimated_complexity": "medium"
        })
        
        # Process the webhook
        response = await code_agent._process_message(webhook_message)
        
        # Verify processing
        code_agent.ai_analyzer.analyze_issue.assert_called_once()
        assert response is not None
        assert response.message_type == "webhook_processed"
    
    @pytest.mark.asyncio
    async def test_execute_code_review_task(self, code_agent):
        """Test executing a code review task."""
        task = {
            "type": "code_review",
            "repository": "test-user/test-repo",
            "pull_request": 1,
            "review_type": "full",
            "focus_areas": ["security", "performance"]
        }
        
        # Mock GitHub data
        mock_pr = MockGitHubPullRequest()
        mock_files = [
            {
                "filename": "test.py",
                "patch": "@@ -1,3 +1,4 @@\n def test():\n+    print('hello')\n     return True",
                "additions": 1,
                "deletions": 0
            }
        ]
        
        code_agent.github_service.get_pull_request = AsyncMock(return_value=mock_pr.to_github_format())
        code_agent.github_service.get_pull_request_files = AsyncMock(return_value=mock_files)
        
        # Mock AI analysis
        code_agent.ai_analyzer.review_pull_request = AsyncMock(return_value={
            "review_id": "review_456",
            "files_analyzed": 1,
            "issues_found": 2,
            "comments": [
                {
                    "file": "test.py",
                    "line": 2,
                    "message": "Consider using logging instead of print statements",
                    "category": "best_practice",
                    "confidence": 0.9
                }
            ],
            "security_issues": [],
            "performance_issues": [
                {
                    "file": "test.py",
                    "message": "Function could be optimized",
                    "suggestion": "Use more efficient algorithm"
                }
            ],
            "overall_assessment": "Good code with minor improvements needed",
            "approve": True
        })
        
        # Execute the task
        result = await code_agent.execute_task(task)
        
        # Verify execution
        assert result["success"] is True
        assert result["review_id"] == "review_456"
        assert result["files_analyzed"] == 1
        assert result["issues_found"] == 2
        assert "comments" in result
        assert "security_issues" in result
        assert "performance_issues" in result
        
        code_agent.github_service.get_pull_request.assert_called_once()
        code_agent.github_service.get_pull_request_files.assert_called_once()
        code_agent.ai_analyzer.review_pull_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_vulnerability_scan_task(self, code_agent):
        """Test executing a vulnerability scan task."""
        task = {
            "type": "vulnerability_scan",
            "repository": "test-user/test-repo",
            "branch": "main",
            "scan_type": "full",
            "include_dependencies": True
        }
        
        # Mock vulnerability scan results
        code_agent.ai_analyzer.detect_vulnerabilities = AsyncMock(return_value={
            "scan_id": "scan_789",
            "vulnerabilities": [
                {
                    "severity": "high",
                    "category": "sql_injection",
                    "file": "database.py",
                    "line": 45,
                    "description": "Potential SQL injection vulnerability",
                    "recommendation": "Use parameterized queries",
                    "cwe": "CWE-89"
                }
            ],
            "dependencies": {
                "vulnerable_packages": [
                    {
                        "package": "requests",
                        "version": "2.20.0",
                        "vulnerability": "CVE-2023-12345",
                        "severity": "medium"
                    }
                ]
            },
            "summary": {
                "total_vulnerabilities": 1,
                "high_severity": 1,
                "medium_severity": 0,
                "low_severity": 0
            }
        })
        
        # Execute the task
        result = await code_agent.execute_task(task)
        
        # Verify execution
        assert result["success"] is True
        assert result["scan_id"] == "scan_789"
        assert result["total_vulnerabilities"] == 1
        assert len(result["vulnerabilities"]) == 1
        assert result["vulnerabilities"][0]["severity"] == "high"
        assert "dependencies" in result
        
        code_agent.ai_analyzer.detect_vulnerabilities.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_documentation_generation_task(self, code_agent):
        """Test executing documentation generation task."""
        task = {
            "type": "generate_documentation",
            "repository": "test-user/test-repo",
            "files": ["src/main.py", "src/utils.py"],
            "doc_type": "api",
            "format": "markdown"
        }
        
        # Mock documentation generation
        code_agent.ai_analyzer.generate_documentation = AsyncMock(return_value={
            "doc_id": "doc_101",
            "files_processed": 2,
            "documentation": {
                "src/main.py": {
                    "summary": "Main application module",
                    "functions": [
                        {
                            "name": "main",
                            "description": "Entry point for the application",
                            "parameters": [],
                            "returns": "None"
                        }
                    ],
                    "classes": []
                },
                "src/utils.py": {
                    "summary": "Utility functions",
                    "functions": [
                        {
                            "name": "helper_function",
                            "description": "Performs utility operations",
                            "parameters": ["param1: str", "param2: int"],
                            "returns": "bool"
                        }
                    ]
                }
            },
            "markdown_output": "# API Documentation\n\n## Main Module\n...",
            "success": True
        })
        
        # Execute the task
        result = await code_agent.execute_task(task)
        
        # Verify execution
        assert result["success"] is True
        assert result["doc_id"] == "doc_101"
        assert result["files_processed"] == 2
        assert "documentation" in result
        assert "markdown_output" in result
        
        code_agent.ai_analyzer.generate_documentation.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_repository_monitoring_start(self, code_agent):
        """Test starting repository monitoring."""
        # Mock repositories to monitor
        code_agent.monitored_repositories = ["test-user/test-repo"]
        code_agent._repository_monitoring_task = None
        
        # Start monitoring
        await code_agent._start_repository_monitoring()
        
        # Verify monitoring started
        assert code_agent._repository_monitoring_task is not None
        assert not code_agent._repository_monitoring_task.done()
        
        # Cleanup
        code_agent._repository_monitoring_task.cancel()
        await asyncio.sleep(0.1)  # Let the task cleanup
    
    @pytest.mark.asyncio
    async def test_repository_monitoring_stop(self, code_agent):
        """Test stopping repository monitoring."""
        # Create a mock monitoring task
        code_agent._repository_monitoring_task = asyncio.create_task(asyncio.sleep(10))
        
        # Stop monitoring
        await code_agent._stop_repository_monitoring()
        
        # Verify monitoring stopped
        assert code_agent._repository_monitoring_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_webhook_handler_registration(self, code_agent):
        """Test webhook handler registration."""
        # Setup webhook handlers
        await code_agent._setup_webhook_handlers()
        
        # Verify handlers are registered
        assert "pull_request" in code_agent.webhook_handlers
        assert "issues" in code_agent.webhook_handlers
        assert "push" in code_agent.webhook_handlers
        assert callable(code_agent.webhook_handlers["pull_request"])
        assert callable(code_agent.webhook_handlers["issues"])
        assert callable(code_agent.webhook_handlers["push"])
    
    @pytest.mark.asyncio
    async def test_ai_code_analysis_integration(self, code_agent):
        """Test AI code analysis integration."""
        # Test data
        code_content = """
        def vulnerable_function(user_input):
            # This is a SQL injection vulnerability
            query = "SELECT * FROM users WHERE name = '" + user_input + "'"
            return execute_query(query)
        """
        
        # Mock AI analysis
        code_agent.ai_analyzer.analyze_code = AsyncMock(return_value={
            "analysis_id": "analysis_202",
            "code_quality": {
                "score": 6.5,
                "issues": [
                    {
                        "type": "security",
                        "severity": "high",
                        "line": 4,
                        "message": "SQL injection vulnerability detected",
                        "suggestion": "Use parameterized queries or ORM"
                    }
                ]
            },
            "style_issues": [
                {
                    "line": 3,
                    "message": "Line too long (exceeds 80 characters)",
                    "rule": "E501"
                }
            ],
            "performance_suggestions": [],
            "security_assessment": {
                "risk_level": "high",
                "vulnerabilities": 1,
                "recommendations": [
                    "Implement input validation",
                    "Use prepared statements"
                ]
            }
        })
        
        # Perform analysis
        result = await code_agent.ai_analyzer.analyze_code(code_content, "python")
        
        # Verify analysis
        assert result["analysis_id"] == "analysis_202"
        assert result["code_quality"]["score"] == 6.5
        assert len(result["code_quality"]["issues"]) == 1
        assert result["code_quality"]["issues"][0]["severity"] == "high"
        assert result["security_assessment"]["risk_level"] == "high"
        assert result["security_assessment"]["vulnerabilities"] == 1
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, code_agent):
        """Test getting agent metrics."""
        # Set up some test metrics
        code_agent.metrics.update({
            "pull_requests_reviewed": 15,
            "vulnerabilities_detected": 3,
            "documentation_generated": 8,
            "repositories_monitored": 5
        })
        
        # Get metrics
        metrics = code_agent.get_metrics()
        
        # Verify metrics
        assert metrics["agent_id"] == "test_code_agent"
        assert metrics["pull_requests_reviewed"] == 15
        assert metrics["vulnerabilities_detected"] == 3
        assert metrics["documentation_generated"] == 8
        assert metrics["repositories_monitored"] == 5
        assert "uptime" in metrics
        assert "state" in metrics
    
    @pytest.mark.asyncio
    async def test_error_handling_github_service_failure(self, code_agent):
        """Test error handling when GitHub service fails."""
        # Mock GitHub service failure
        code_agent.github_service.get_pull_request = AsyncMock(side_effect=GitHubError("API rate limit exceeded"))
        
        task = {
            "type": "code_review",
            "repository": "test-user/test-repo",
            "pull_request": 1
        }
        
        # Execute task and expect failure
        result = await code_agent.execute_task(task)
        
        # Verify error handling
        assert result["success"] is False
        assert "error" in result
        assert "API rate limit exceeded" in result["error"]
        assert code_agent.metrics["errors"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_ai_analysis_failure(self, code_agent):
        """Test error handling when AI analysis fails."""
        # Mock AI analysis failure
        code_agent.ai_analyzer.review_pull_request = AsyncMock(side_effect=Exception("AI model not available"))
        
        # Mock successful GitHub calls
        mock_pr = MockGitHubPullRequest()
        code_agent.github_service.get_pull_request = AsyncMock(return_value=mock_pr.to_github_format())
        code_agent.github_service.get_pull_request_files = AsyncMock(return_value=[])
        
        task = {
            "type": "code_review",
            "repository": "test-user/test-repo",
            "pull_request": 1
        }
        
        # Execute task and expect failure
        result = await code_agent.execute_task(task)
        
        # Verify error handling
        assert result["success"] is False
        assert "error" in result
        assert "AI model not available" in result["error"]
        assert code_agent.metrics["errors"] > 0
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, code_agent):
        """Test configuration validation."""
        # Test with invalid configuration
        invalid_config = {
            "github": {
                # Missing required 'token' field
                "base_url": "https://api.github.com"
            }
        }
        
        # Validate configuration
        is_valid = code_agent._validate_configuration(invalid_config)
        assert is_valid is False
        
        # Test with valid configuration
        valid_config = {
            "github": {
                "token": "test_token",
                "base_url": "https://api.github.com"
            },
            "ai_analysis": {
                "enabled": True,
                "model": "codellama:7b"
            }
        }
        
        is_valid = code_agent._validate_configuration(valid_config)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_message_handling_unknown_type(self, code_agent):
        """Test handling of unknown message types."""
        unknown_message = AgentMessage(
            id="test_msg_unknown",
            sender="test_sender",
            recipient="test_code_agent",
            message_type="unknown_type",
            payload={"data": "test"}
        )
        
        # Process unknown message
        response = await code_agent._process_message(unknown_message)
        
        # Verify response
        assert response is None or response.message_type == "error"
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, code_agent):
        """Test concurrent task execution."""
        # Create multiple tasks
        tasks = [
            {
                "type": "code_review",
                "repository": f"test-user/test-repo-{i}",
                "pull_request": i+1
            }
            for i in range(3)
        ]
        
        # Mock GitHub service calls
        mock_pr = MockGitHubPullRequest()
        code_agent.github_service.get_pull_request = AsyncMock(return_value=mock_pr.to_github_format())
        code_agent.github_service.get_pull_request_files = AsyncMock(return_value=[])
        
        # Mock AI analysis
        code_agent.ai_analyzer.review_pull_request = AsyncMock(return_value={
            "review_id": "concurrent_review",
            "comments": [],
            "approve": True
        })
        
        # Execute tasks concurrently
        results = await asyncio.gather(*[code_agent.execute_task(task) for task in tasks])
        
        # Verify all tasks completed
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert result["review_id"] == "concurrent_review"
    
    @pytest.mark.asyncio
    async def test_repository_monitoring_integration(self, code_agent):
        """Test repository monitoring integration."""
        # Mock repository data
        code_agent.monitored_repositories = ["test-user/test-repo"]
        
        # Mock GitHub service responses
        mock_commits = [MockGitHubCommit(sha=f"sha_{i}") for i in range(3)]
        code_agent.github_service.get_commits = AsyncMock(return_value=[c.to_github_format() for c in mock_commits])
        
        # Mock AI analysis for commits
        code_agent.ai_analyzer.analyze_commits = AsyncMock(return_value={
            "analysis_id": "commit_analysis_123",
            "commits_analyzed": 3,
            "issues_found": 1,
            "recommendations": ["Add more unit tests"]
        })
        
        # Start monitoring
        await code_agent._start_repository_monitoring()
        
        # Let monitoring run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await code_agent._stop_repository_monitoring()
        
        # Verify monitoring was active
        assert code_agent._repository_monitoring_task is not None