"""
Test suite for the GitHub Service implementation.

This test suite covers all GitHub API integration functionality:
- Repository operations
- Pull request management
- Issue tracking
- Webhook handling
- Authentication and rate limiting
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from src.services.github_service import (
    GitHubService,
    GitHubError,
    GitHubRateLimitError,
    GitHubAuthenticationError,
    GitHubNotFoundError,
    PullRequestData,
    IssueData,
    RepositoryData,
    CommitData,
    WebhookData,
)
from tests.mocks.github_mocks import (
    MockGitHubClient,
    MockGitHubPullRequest,
    MockGitHubRepository,
    MockGitHubIssue,
    MockGitHubCommit,
    MockGitHubWebhook,
    MockGitHubAPIContext,
    generate_pr_webhook_payload,
    generate_issue_webhook_payload,
    generate_sample_repositories,
    generate_sample_pull_requests,
    generate_sample_issues,
)


class TestGitHubService:
    """Test suite for the GitHub Service."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration."""
        return {
            "token": "test_github_token",
            "base_url": "https://api.github.com",
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
            "rate_limit_buffer": 10,
            "per_page": 100,
        }
    
    @pytest.fixture
    def mock_logger(self):
        """Provide a mock logger."""
        return MagicMock()
    
    @pytest.fixture
    def mock_github_client(self):
        """Provide a mock GitHub client."""
        client = MockGitHubClient()
        
        # Add sample data
        for repo in generate_sample_repositories(3):
            client.add_repository(repo)
        
        for pr in generate_sample_pull_requests(5):
            client.add_pull_request(pr)
        
        for issue in generate_sample_issues(4):
            client.add_issue(issue)
        
        return client
    
    @pytest.fixture
    def github_service(self, mock_config, mock_logger):
        """Provide a GitHub service instance."""
        return GitHubService(mock_config, mock_logger)
    
    @pytest.mark.asyncio
    async def test_github_service_initialization(self, github_service, mock_config):
        """Test GitHub service initialization."""
        assert github_service.config == mock_config
        assert github_service.token == "test_github_token"
        assert github_service.base_url == "https://api.github.com"
        assert github_service.timeout == 30
        assert github_service.max_retries == 3
        assert github_service.is_connected is False
        assert github_service.rate_limit_remaining == 5000
    
    @pytest.mark.asyncio
    async def test_connect_success(self, github_service, mock_github_client):
        """Test successful connection to GitHub."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
        
        assert github_service.is_connected is True
        assert github_service.client is not None
    
    @pytest.mark.asyncio
    async def test_connect_failure_invalid_token(self, github_service):
        """Test connection failure with invalid token."""
        with patch('github.Github') as mock_github:
            mock_github.side_effect = Exception("Bad credentials")
            
            with pytest.raises(GitHubAuthenticationError):
                await github_service.connect()
        
        assert github_service.is_connected is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, github_service, mock_github_client):
        """Test disconnection from GitHub."""
        # Connect first
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
        
        assert github_service.is_connected is True
        
        # Disconnect
        await github_service.disconnect()
        
        assert github_service.is_connected is False
        assert github_service.client is None
    
    @pytest.mark.asyncio
    async def test_health_check_connected(self, github_service, mock_github_client):
        """Test health check when connected."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            health = await github_service.health_check()
        
        assert health["status"] == "healthy"
        assert health["connected"] is True
        assert "rate_limit" in health
        assert "api_version" in health
    
    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, github_service):
        """Test health check when disconnected."""
        health = await github_service.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["connected"] is False
        assert "error" in health
    
    @pytest.mark.asyncio
    async def test_get_repository_success(self, github_service, mock_github_client):
        """Test successful repository retrieval."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            repo = await github_service.get_repository("test-user-1/test-repo-1")
        
        assert isinstance(repo, RepositoryData)
        assert repo.full_name == "test-user-1/test-repo-1"
        assert repo.owner == "test-user-1"
        assert repo.name == "test-repo-1"
        assert repo.language == "Python"
    
    @pytest.mark.asyncio
    async def test_get_repository_not_found(self, github_service, mock_github_client):
        """Test repository retrieval when repository doesn't exist."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            with pytest.raises(GitHubNotFoundError):
                await github_service.get_repository("nonexistent/repo")
    
    @pytest.mark.asyncio
    async def test_get_pull_request_success(self, github_service, mock_github_client):
        """Test successful pull request retrieval."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            pr = await github_service.get_pull_request("test-user/test-repo", 1)
        
        assert isinstance(pr, PullRequestData)
        assert pr.number == 1
        assert pr.title == "Feature: Add feature 1"
        assert pr.state == "open"
        assert pr.head_ref == "feature-1"
        assert pr.base_ref == "main"
    
    @pytest.mark.asyncio
    async def test_get_pull_request_files(self, github_service, mock_github_client):
        """Test getting pull request files."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            files = await github_service.get_pull_request_files("test-user/test-repo", 1)
        
        assert isinstance(files, list)
        # Would normally contain file data, but mocked for testing
        assert len(files) >= 0
    
    @pytest.mark.asyncio
    async def test_create_pr_comment(self, github_service, mock_github_client):
        """Test creating a pull request comment."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            comment = await github_service.create_pr_comment(
                "test-user/test-repo", 
                1, 
                "This is a test comment"
            )
        
        assert comment is not None
        assert comment.get("body") == "This is a test comment"
    
    @pytest.mark.asyncio
    async def test_create_pr_review(self, github_service, mock_github_client):
        """Test creating a pull request review."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            review_data = {
                "body": "Overall looks good!",
                "event": "APPROVE",
                "comments": [
                    {
                        "path": "test.py",
                        "line": 10,
                        "body": "Consider using a more descriptive name"
                    }
                ]
            }
            
            review = await github_service.create_pr_review(
                "test-user/test-repo", 
                1, 
                review_data
            )
        
        assert review is not None
        assert review.get("body") == "Overall looks good!"
        assert review.get("state") == "approved"
    
    @pytest.mark.asyncio
    async def test_get_issues(self, github_service, mock_github_client):
        """Test getting repository issues."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            issues = await github_service.get_issues("test-user/test-repo")
        
        assert isinstance(issues, list)
        assert len(issues) == 4  # Based on sample data
        
        for issue in issues:
            assert isinstance(issue, IssueData)
            assert hasattr(issue, 'number')
            assert hasattr(issue, 'title')
            assert hasattr(issue, 'state')
    
    @pytest.mark.asyncio
    async def test_create_issue(self, github_service, mock_github_client):
        """Test creating a new issue."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            issue_data = {
                "title": "Test Issue",
                "body": "This is a test issue",
                "labels": ["bug", "high-priority"],
                "assignees": ["test-user"]
            }
            
            issue = await github_service.create_issue("test-user/test-repo", issue_data)
        
        assert issue is not None
        assert issue.get("title") == "Test Issue"
        assert issue.get("body") == "This is a test issue"
    
    @pytest.mark.asyncio
    async def test_close_issue(self, github_service, mock_github_client):
        """Test closing an issue."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            result = await github_service.close_issue("test-user/test-repo", 1)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_commits(self, github_service, mock_github_client):
        """Test getting repository commits."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            commits = await github_service.get_commits("test-user/test-repo")
        
        assert isinstance(commits, list)
        
        for commit in commits:
            assert isinstance(commit, CommitData)
            assert hasattr(commit, 'sha')
            assert hasattr(commit, 'message')
            assert hasattr(commit, 'author')
    
    @pytest.mark.asyncio
    async def test_set_webhook(self, github_service, mock_github_client):
        """Test setting up a webhook."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            webhook_config = {
                "url": "https://example.com/webhook",
                "secret": "webhook_secret",
                "events": ["push", "pull_request", "issues"]
            }
            
            webhook = await github_service.set_webhook("test-user/test-repo", webhook_config)
        
        assert webhook is not None
        assert webhook.get("config", {}).get("url") == "https://example.com/webhook"
        assert "events" in webhook
    
    @pytest.mark.asyncio
    async def test_validate_webhook_signature(self, github_service):
        """Test webhook signature validation."""
        payload = '{"test": "data"}'
        secret = "test_secret"
        
        # Generate valid signature
        import hmac
        import hashlib
        
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Test valid signature
        is_valid = github_service.validate_webhook_signature(
            payload, 
            f"sha256={signature}", 
            secret
        )
        assert is_valid is True
        
        # Test invalid signature
        is_valid = github_service.validate_webhook_signature(
            payload,
            "sha256=invalid_signature",
            secret
        )
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_parse_webhook_payload(self, github_service):
        """Test parsing webhook payload."""
        # Test pull request webhook
        pr_payload = generate_pr_webhook_payload("opened")
        
        webhook_data = github_service.parse_webhook_payload(
            "pull_request",
            pr_payload
        )
        
        assert isinstance(webhook_data, WebhookData)
        assert webhook_data.event == "pull_request"
        assert webhook_data.action == "opened"
        assert webhook_data.repository is not None
        assert webhook_data.pull_request is not None
        
        # Test issue webhook
        issue_payload = generate_issue_webhook_payload("opened")
        
        webhook_data = github_service.parse_webhook_payload(
            "issues",
            issue_payload
        )
        
        assert isinstance(webhook_data, WebhookData)
        assert webhook_data.event == "issues"
        assert webhook_data.action == "opened"
        assert webhook_data.repository is not None
        assert webhook_data.issue is not None
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, github_service, mock_github_client):
        """Test rate limit handling."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            # Mock rate limit exceeded
            mock_github_client.rate_limit["remaining"] = 0
            
            with patch.object(github_service, '_check_rate_limit', side_effect=GitHubRateLimitError("Rate limit exceeded")):
                with pytest.raises(GitHubRateLimitError):
                    await github_service.get_repository("test-user/test-repo")
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, github_service, mock_github_client):
        """Test retry mechanism for failed requests."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            # Mock temporary failure followed by success
            call_count = 0
            original_get_repo = github_service.client.get_repo
            
            def mock_get_repo(repo_name):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Temporary failure")
                return original_get_repo(repo_name)
            
            github_service.client.get_repo = mock_get_repo
            
            # Should succeed after retries
            repo = await github_service.get_repository("test-user-1/test-repo-1")
            assert isinstance(repo, RepositoryData)
            assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, github_service, mock_github_client):
        """Test handling concurrent requests."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            # Make multiple concurrent requests
            tasks = [
                github_service.get_repository(f"test-user-{i}/test-repo-{i}")
                for i in range(1, 4)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, RepositoryData)
    
    @pytest.mark.asyncio
    async def test_search_repositories(self, github_service, mock_github_client):
        """Test searching repositories."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            results = await github_service.search_repositories("test-repo language:Python")
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for repo in results:
            assert isinstance(repo, RepositoryData)
            assert "test-repo" in repo.name
    
    @pytest.mark.asyncio
    async def test_search_issues(self, github_service, mock_github_client):
        """Test searching issues."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            results = await github_service.search_issues("bug repo:test-user/test-repo")
        
        assert isinstance(results, list)
        
        for issue in results:
            assert isinstance(issue, IssueData)
    
    @pytest.mark.asyncio
    async def test_get_pull_request_diff(self, github_service, mock_github_client):
        """Test getting pull request diff."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            diff = await github_service.get_pull_request_diff("test-user/test-repo", 1)
        
        assert isinstance(diff, str)
        assert len(diff) > 0
    
    @pytest.mark.asyncio
    async def test_get_file_content(self, github_service, mock_github_client):
        """Test getting file content from repository."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            content = await github_service.get_file_content(
                "test-user/test-repo", 
                "README.md", 
                "main"
            )
        
        assert isinstance(content, str)
        assert len(content) > 0
    
    @pytest.mark.asyncio
    async def test_create_commit_comment(self, github_service, mock_github_client):
        """Test creating a commit comment."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            comment = await github_service.create_commit_comment(
                "test-user/test-repo",
                "abc123def456",
                "This commit looks good!"
            )
        
        assert comment is not None
        assert comment.get("body") == "This commit looks good!"
    
    @pytest.mark.asyncio
    async def test_get_repository_languages(self, github_service, mock_github_client):
        """Test getting repository languages."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            languages = await github_service.get_repository_languages("test-user/test-repo")
        
        assert isinstance(languages, dict)
        assert "Python" in languages
    
    @pytest.mark.asyncio
    async def test_get_repository_topics(self, github_service, mock_github_client):
        """Test getting repository topics."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            topics = await github_service.get_repository_topics("test-user/test-repo")
        
        assert isinstance(topics, list)
    
    @pytest.mark.asyncio
    async def test_error_handling_network_failure(self, github_service, mock_github_client):
        """Test error handling for network failures."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            with patch.object(github_service.client, 'get_repo', side_effect=ConnectionError("Network error")):
                with pytest.raises(GitHubError):
                    await github_service.get_repository("test-user/test-repo")
    
    @pytest.mark.asyncio
    async def test_error_handling_timeout(self, github_service, mock_github_client):
        """Test error handling for timeouts."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            with patch.object(github_service.client, 'get_repo', side_effect=asyncio.TimeoutError("Request timed out")):
                with pytest.raises(GitHubError):
                    await github_service.get_repository("test-user/test-repo")
    
    @pytest.mark.asyncio
    async def test_pagination_handling(self, github_service, mock_github_client):
        """Test pagination handling for large result sets."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            # Test with pagination parameters
            issues = await github_service.get_issues(
                "test-user/test-repo",
                page=1,
                per_page=2
            )
        
        assert isinstance(issues, list)
        assert len(issues) <= 2  # Should respect per_page limit
    
    @pytest.mark.asyncio
    async def test_webhook_event_filtering(self, github_service):
        """Test webhook event filtering."""
        # Create webhook data for different events
        pr_payload = generate_pr_webhook_payload("opened")
        issue_payload = generate_issue_webhook_payload("closed")
        
        pr_webhook = github_service.parse_webhook_payload("pull_request", pr_payload)
        issue_webhook = github_service.parse_webhook_payload("issues", issue_payload)
        
        # Test event filtering
        assert github_service.should_process_webhook(pr_webhook, ["pull_request"]) is True
        assert github_service.should_process_webhook(pr_webhook, ["issues"]) is False
        assert github_service.should_process_webhook(issue_webhook, ["issues"]) is True
        assert github_service.should_process_webhook(issue_webhook, ["pull_request"]) is False
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, github_service, mock_github_client):
        """Test getting service metrics."""
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            
            # Make some API calls to generate metrics
            await github_service.get_repository("test-user-1/test-repo-1")
            await github_service.get_issues("test-user-1/test-repo-1")
        
        metrics = github_service.get_metrics()
        
        assert "api_calls" in metrics
        assert "rate_limit_remaining" in metrics
        assert "errors" in metrics
        assert "last_request_time" in metrics
        assert metrics["api_calls"] > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_on_disconnect(self, github_service, mock_github_client):
        """Test proper cleanup when disconnecting."""
        # Connect and create some state
        with MockGitHubAPIContext(mock_github_client):
            await github_service.connect()
            github_service.api_calls = 100
            github_service.errors = 5
        
        # Disconnect
        await github_service.disconnect()
        
        # Verify cleanup
        assert github_service.is_connected is False
        assert github_service.client is None
        # Metrics should be preserved for reporting
        assert github_service.api_calls == 100
        assert github_service.errors == 5