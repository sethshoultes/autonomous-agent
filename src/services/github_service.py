"""
GitHub API integration service for the Code Agent.

This service provides comprehensive GitHub API integration including repository
management, pull request operations, issue tracking, webhook handling, and
authentication management with proper rate limiting and error handling.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import github
from github import Github, GithubException
from tenacity import retry, stop_after_attempt, wait_exponential


# Exception classes
class GitHubError(Exception):
    """Base exception for GitHub-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class GitHubAuthenticationError(GitHubError):
    """Exception raised when GitHub authentication fails."""
    pass


class GitHubRateLimitError(GitHubError):
    """Exception raised when GitHub rate limit is exceeded."""
    pass


class GitHubNotFoundError(GitHubError):
    """Exception raised when GitHub resource is not found."""
    pass


class GitHubPermissionError(GitHubError):
    """Exception raised when insufficient permissions for GitHub operation."""
    pass


# Data models
@dataclass
class RepositoryData:
    """Repository information."""
    id: int
    name: str
    full_name: str
    owner: str
    description: Optional[str]
    private: bool
    language: Optional[str]
    stars: int
    forks: int
    issues: int
    default_branch: str
    created_at: datetime
    updated_at: datetime
    clone_url: str
    html_url: str
    topics: List[str] = field(default_factory=list)
    languages: Dict[str, int] = field(default_factory=dict)


@dataclass
class PullRequestData:
    """Pull request information."""
    id: int
    number: int
    title: str
    body: Optional[str]
    state: str
    user: str
    head_ref: str
    base_ref: str
    head_sha: str
    base_sha: str
    mergeable: Optional[bool]
    merged: bool
    comments: int
    commits: int
    additions: int
    deletions: int
    changed_files: int
    created_at: datetime
    updated_at: datetime
    merged_at: Optional[datetime]
    html_url: str
    diff_url: str
    patch_url: str
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)


@dataclass
class IssueData:
    """Issue information."""
    id: int
    number: int
    title: str
    body: Optional[str]
    state: str
    user: str
    comments: int
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]
    html_url: str
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    milestone: Optional[str] = None


@dataclass
class CommitData:
    """Commit information."""
    sha: str
    message: str
    author: str
    author_email: str
    committer: str
    committer_email: str
    date: datetime
    html_url: str
    parents: List[str] = field(default_factory=list)
    stats: Optional[Dict[str, int]] = None


@dataclass
class WebhookData:
    """Webhook event data."""
    event: str
    action: str
    repository: Optional[RepositoryData]
    pull_request: Optional[PullRequestData] = None
    issue: Optional[IssueData] = None
    sender: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FileData:
    """File information."""
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: Optional[str] = None
    content: Optional[str] = None
    blob_url: Optional[str] = None


class GitHubService:
    """
    GitHub API integration service.
    
    Provides comprehensive GitHub API functionality with proper error handling,
    rate limiting, authentication, and retry mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the GitHub service.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Configuration
        self.token = config.get("token")
        self.base_url = config.get("base_url", "https://api.github.com")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.rate_limit_buffer = config.get("rate_limit_buffer", 10)
        self.per_page = config.get("per_page", 100)
        
        # GitHub client
        self.client: Optional[Github] = None
        self.is_connected = False
        
        # Rate limiting
        self.rate_limit_remaining = 5000
        self.rate_limit_reset_time = 0
        self.last_rate_limit_check = 0
        
        # Metrics
        self.api_calls = 0
        self.errors = 0
        self.last_request_time: Optional[datetime] = None
        self.start_time = time.time()
    
    async def connect(self) -> None:
        """Connect to GitHub API."""
        try:
            if not self.token:
                raise GitHubAuthenticationError("GitHub token not provided")
            
            # Initialize GitHub client
            self.client = Github(
                self.token,
                base_url=self.base_url,
                timeout=self.timeout,
                per_page=self.per_page
            )
            
            # Test connection
            user = self.client.get_user()
            self.logger.info(f"Connected to GitHub as {user.login}")
            
            # Get initial rate limit
            await self._update_rate_limit()
            
            self.is_connected = True
            
        except GithubException as e:
            if e.status == 401:
                raise GitHubAuthenticationError(f"Invalid GitHub token: {e}")
            else:
                raise GitHubError(f"GitHub connection failed: {e}") from e
        except Exception as e:
            raise GitHubError(f"Failed to connect to GitHub: {e}") from e
    
    async def disconnect(self) -> None:
        """Disconnect from GitHub API."""
        self.client = None
        self.is_connected = False
        self.logger.info("Disconnected from GitHub")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on GitHub service.
        
        Returns:
            Health status dictionary
        """
        try:
            if not self.is_connected or not self.client:
                return {
                    "status": "unhealthy",
                    "connected": False,
                    "error": "Not connected to GitHub",
                    "last_check": datetime.now(timezone.utc).isoformat()
                }
            
            # Check rate limit
            rate_limit = self.client.get_rate_limit()
            
            uptime = time.time() - self.start_time
            
            return {
                "status": "healthy",
                "connected": True,
                "rate_limit": {
                    "remaining": rate_limit.core.remaining,
                    "limit": rate_limit.core.limit,
                    "reset": rate_limit.core.reset.isoformat()
                },
                "api_calls": self.api_calls,
                "errors": self.errors,
                "uptime": uptime,
                "api_version": "v3",
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"GitHub health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def get_repository(self, repository: str) -> RepositoryData:
        """
        Get repository information.
        
        Args:
            repository: Repository name in format "owner/repo"
            
        Returns:
            RepositoryData object
            
        Raises:
            GitHubNotFoundError: If repository is not found
            GitHubError: If API call fails
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            
            # Get additional data
            topics = repo.get_topics()
            languages = repo.get_languages()
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return RepositoryData(
                id=repo.id,
                name=repo.name,
                full_name=repo.full_name,
                owner=repo.owner.login,
                description=repo.description,
                private=repo.private,
                language=repo.language,
                stars=repo.stargazers_count,
                forks=repo.forks_count,
                issues=repo.open_issues_count,
                default_branch=repo.default_branch,
                created_at=repo.created_at,
                updated_at=repo.updated_at,
                clone_url=repo.clone_url,
                html_url=repo.html_url,
                topics=topics,
                languages=languages
            )
            
        except GithubException as e:
            self.errors += 1
            if e.status == 404:
                raise GitHubNotFoundError(f"Repository {repository} not found")
            elif e.status == 403:
                if "rate limit" in str(e).lower():
                    raise GitHubRateLimitError(f"Rate limit exceeded: {e}")
                else:
                    raise GitHubPermissionError(f"Insufficient permissions: {e}")
            else:
                raise GitHubError(f"Failed to get repository {repository}: {e}") from e
    
    async def get_pull_request(self, repository: str, pr_number: int) -> PullRequestData:
        """
        Get pull request information.
        
        Args:
            repository: Repository name in format "owner/repo"
            pr_number: Pull request number
            
        Returns:
            PullRequestData object
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            pr = repo.get_pull(pr_number)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return PullRequestData(
                id=pr.id,
                number=pr.number,
                title=pr.title,
                body=pr.body,
                state=pr.state,
                user=pr.user.login,
                head_ref=pr.head.ref,
                base_ref=pr.base.ref,
                head_sha=pr.head.sha,
                base_sha=pr.base.sha,
                mergeable=pr.mergeable,
                merged=pr.merged,
                comments=pr.comments,
                commits=pr.commits,
                additions=pr.additions,
                deletions=pr.deletions,
                changed_files=pr.changed_files,
                created_at=pr.created_at,
                updated_at=pr.updated_at,
                merged_at=pr.merged_at,
                html_url=pr.html_url,
                diff_url=pr.diff_url,
                patch_url=pr.patch_url,
                labels=[label.name for label in pr.labels],
                assignees=[assignee.login for assignee in pr.assignees],
                reviewers=[reviewer.login for reviewer in pr.get_review_requests()[0]]
            )
            
        except GithubException as e:
            self.errors += 1
            if e.status == 404:
                raise GitHubNotFoundError(f"Pull request {pr_number} not found in {repository}")
            else:
                raise GitHubError(f"Failed to get pull request {pr_number}: {e}") from e
    
    async def get_pull_request_files(self, repository: str, pr_number: int) -> List[FileData]:
        """
        Get files changed in a pull request.
        
        Args:
            repository: Repository name in format "owner/repo"
            pr_number: Pull request number
            
        Returns:
            List of FileData objects
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            pr = repo.get_pull(pr_number)
            files = pr.get_files()
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [
                FileData(
                    filename=file.filename,
                    status=file.status,
                    additions=file.additions,
                    deletions=file.deletions,
                    changes=file.changes,
                    patch=file.patch,
                    blob_url=file.blob_url
                )
                for file in files
            ]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get pull request files: {e}") from e
    
    async def get_pull_request_diff(self, repository: str, pr_number: int) -> str:
        """
        Get pull request diff.
        
        Args:
            repository: Repository name in format "owner/repo"
            pr_number: Pull request number
            
        Returns:
            Diff content as string
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            pr = repo.get_pull(pr_number)
            
            # Get diff using API
            import requests
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3.diff"
            }
            
            response = requests.get(pr.url, headers=headers)
            response.raise_for_status()
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return response.text
            
        except Exception as e:
            self.errors += 1
            raise GitHubError(f"Failed to get pull request diff: {e}") from e
    
    async def create_pr_comment(
        self, 
        repository: str, 
        pr_number: int, 
        comment: str
    ) -> Dict[str, Any]:
        """
        Create a comment on a pull request.
        
        Args:
            repository: Repository name in format "owner/repo"
            pr_number: Pull request number
            comment: Comment text
            
        Returns:
            Comment data
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            pr = repo.get_pull(pr_number)
            comment_obj = pr.create_issue_comment(comment)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return {
                "id": comment_obj.id,
                "body": comment_obj.body,
                "user": comment_obj.user.login,
                "created_at": comment_obj.created_at.isoformat(),
                "html_url": comment_obj.html_url
            }
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to create PR comment: {e}") from e
    
    async def create_pr_review(
        self, 
        repository: str, 
        pr_number: int, 
        review_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a review on a pull request.
        
        Args:
            repository: Repository name in format "owner/repo"
            pr_number: Pull request number
            review_data: Review data including body, event, and comments
            
        Returns:
            Review data
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            pr = repo.get_pull(pr_number)
            
            # Prepare review comments
            comments = []
            for comment_data in review_data.get("comments", []):
                comment = {
                    "path": comment_data["path"],
                    "body": comment_data["body"]
                }
                if "line" in comment_data:
                    comment["line"] = comment_data["line"]
                elif "position" in comment_data:
                    comment["position"] = comment_data["position"]
                
                comments.append(comment)
            
            # Create review
            review = pr.create_review(
                body=review_data.get("body", ""),
                event=review_data.get("event", "COMMENT"),
                comments=comments
            )
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return {
                "id": review.id,
                "body": review.body,
                "state": review.state,
                "user": review.user.login,
                "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
                "html_url": review.html_url
            }
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to create PR review: {e}") from e
    
    async def get_issues(
        self, 
        repository: str, 
        state: str = "open",
        page: int = 1,
        per_page: int = 30
    ) -> List[IssueData]:
        """
        Get repository issues.
        
        Args:
            repository: Repository name in format "owner/repo"
            state: Issue state ("open", "closed", "all")
            page: Page number for pagination
            per_page: Items per page
            
        Returns:
            List of IssueData objects
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            issues = repo.get_issues(state=state).get_page(page-1)[:per_page]
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [
                IssueData(
                    id=issue.id,
                    number=issue.number,
                    title=issue.title,
                    body=issue.body,
                    state=issue.state,
                    user=issue.user.login,
                    comments=issue.comments,
                    created_at=issue.created_at,
                    updated_at=issue.updated_at,
                    closed_at=issue.closed_at,
                    html_url=issue.html_url,
                    labels=[label.name for label in issue.labels],
                    assignees=[assignee.login for assignee in issue.assignees],
                    milestone=issue.milestone.title if issue.milestone else None
                )
                for issue in issues
                if not issue.pull_request  # Filter out pull requests
            ]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get issues: {e}") from e
    
    async def create_issue(
        self, 
        repository: str, 
        issue_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new issue.
        
        Args:
            repository: Repository name in format "owner/repo"
            issue_data: Issue data including title, body, labels, assignees
            
        Returns:
            Issue data
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            
            issue = repo.create_issue(
                title=issue_data["title"],
                body=issue_data.get("body", ""),
                labels=issue_data.get("labels", []),
                assignees=issue_data.get("assignees", [])
            )
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return {
                "id": issue.id,
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "state": issue.state,
                "user": issue.user.login,
                "created_at": issue.created_at.isoformat(),
                "html_url": issue.html_url
            }
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to create issue: {e}") from e
    
    async def close_issue(self, repository: str, issue_number: int) -> bool:
        """
        Close an issue.
        
        Args:
            repository: Repository name in format "owner/repo"
            issue_number: Issue number
            
        Returns:
            True if successful
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            issue = repo.get_issue(issue_number)
            issue.edit(state="closed")
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return True
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to close issue {issue_number}: {e}") from e
    
    async def get_commits(
        self, 
        repository: str, 
        branch: str = None,
        limit: int = 30
    ) -> List[CommitData]:
        """
        Get repository commits.
        
        Args:
            repository: Repository name in format "owner/repo"
            branch: Branch name (default: default branch)
            limit: Maximum number of commits to return
            
        Returns:
            List of CommitData objects
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            
            if branch:
                commits = repo.get_commits(sha=branch)[:limit]
            else:
                commits = repo.get_commits()[:limit]
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [
                CommitData(
                    sha=commit.sha,
                    message=commit.commit.message,
                    author=commit.commit.author.name,
                    author_email=commit.commit.author.email,
                    committer=commit.commit.committer.name,
                    committer_email=commit.commit.committer.email,
                    date=commit.commit.author.date,
                    html_url=commit.html_url,
                    parents=[parent.sha for parent in commit.parents],
                    stats={"additions": commit.stats.additions, "deletions": commit.stats.deletions} if commit.stats else None
                )
                for commit in commits
            ]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get commits: {e}") from e
    
    async def get_pull_requests(
        self, 
        repository: str, 
        state: str = "open",
        limit: int = 30
    ) -> List[PullRequestData]:
        """
        Get repository pull requests.
        
        Args:
            repository: Repository name in format "owner/repo"
            state: PR state ("open", "closed", "all")
            limit: Maximum number of PRs to return
            
        Returns:
            List of PullRequestData objects
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            pulls = repo.get_pulls(state=state)[:limit]
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [
                PullRequestData(
                    id=pr.id,
                    number=pr.number,
                    title=pr.title,
                    body=pr.body,
                    state=pr.state,
                    user=pr.user.login,
                    head_ref=pr.head.ref,
                    base_ref=pr.base.ref,
                    head_sha=pr.head.sha,
                    base_sha=pr.base.sha,
                    mergeable=pr.mergeable,
                    merged=pr.merged,
                    comments=pr.comments,
                    commits=pr.commits,
                    additions=pr.additions,
                    deletions=pr.deletions,
                    changed_files=pr.changed_files,
                    created_at=pr.created_at,
                    updated_at=pr.updated_at,
                    merged_at=pr.merged_at,
                    html_url=pr.html_url,
                    diff_url=pr.diff_url,
                    patch_url=pr.patch_url,
                    labels=[label.name for label in pr.labels],
                    assignees=[assignee.login for assignee in pr.assignees]
                )
                for pr in pulls
            ]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get pull requests: {e}") from e
    
    async def get_file_content(
        self, 
        repository: str, 
        file_path: str, 
        branch: str = None
    ) -> str:
        """
        Get file content from repository.
        
        Args:
            repository: Repository name in format "owner/repo"
            file_path: Path to the file
            branch: Branch name (default: default branch)
            
        Returns:
            File content as string
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            
            if branch:
                file_content = repo.get_contents(file_path, ref=branch)
            else:
                file_content = repo.get_contents(file_path)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return file_content.decoded_content.decode('utf-8')
            
        except GithubException as e:
            self.errors += 1
            if e.status == 404:
                raise GitHubNotFoundError(f"File {file_path} not found in {repository}")
            else:
                raise GitHubError(f"Failed to get file content: {e}") from e
    
    async def get_repository_files(self, repository: str, branch: str = None) -> List[str]:
        """
        Get list of files in repository.
        
        Args:
            repository: Repository name in format "owner/repo"
            branch: Branch name (default: default branch)
            
        Returns:
            List of file paths
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            
            if branch:
                tree = repo.get_git_tree(branch, recursive=True)
            else:
                tree = repo.get_git_tree(repo.default_branch, recursive=True)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [item.path for item in tree.tree if item.type == "blob"]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get repository files: {e}") from e
    
    async def set_webhook(
        self, 
        repository: str, 
        webhook_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set up a webhook for the repository.
        
        Args:
            repository: Repository name in format "owner/repo"
            webhook_config: Webhook configuration
            
        Returns:
            Webhook data
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            
            config = {
                "url": webhook_config["url"],
                "content_type": "json"
            }
            
            if "secret" in webhook_config:
                config["secret"] = webhook_config["secret"]
            
            webhook = repo.create_hook(
                name="web",
                config=config,
                events=webhook_config.get("events", ["push", "pull_request"]),
                active=True
            )
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return {
                "id": webhook.id,
                "name": webhook.name,
                "config": webhook.config,
                "events": webhook.events,
                "active": webhook.active
            }
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to set webhook: {e}") from e
    
    async def search_repositories(self, query: str) -> List[RepositoryData]:
        """
        Search repositories.
        
        Args:
            query: Search query
            
        Returns:
            List of RepositoryData objects
        """
        await self._check_rate_limit()
        
        try:
            repositories = self.client.search_repositories(query)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [
                RepositoryData(
                    id=repo.id,
                    name=repo.name,
                    full_name=repo.full_name,
                    owner=repo.owner.login,
                    description=repo.description,
                    private=repo.private,
                    language=repo.language,
                    stars=repo.stargazers_count,
                    forks=repo.forks_count,
                    issues=repo.open_issues_count,
                    default_branch=repo.default_branch,
                    created_at=repo.created_at,
                    updated_at=repo.updated_at,
                    clone_url=repo.clone_url,
                    html_url=repo.html_url
                )
                for repo in repositories[:30]  # Limit results
            ]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to search repositories: {e}") from e
    
    async def search_issues(self, query: str) -> List[IssueData]:
        """
        Search issues.
        
        Args:
            query: Search query
            
        Returns:
            List of IssueData objects
        """
        await self._check_rate_limit()
        
        try:
            issues = self.client.search_issues(query)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return [
                IssueData(
                    id=issue.id,
                    number=issue.number,
                    title=issue.title,
                    body=issue.body,
                    state=issue.state,
                    user=issue.user.login,
                    comments=issue.comments,
                    created_at=issue.created_at,
                    updated_at=issue.updated_at,
                    closed_at=issue.closed_at,
                    html_url=issue.html_url,
                    labels=[label.name for label in issue.labels],
                    assignees=[assignee.login for assignee in issue.assignees]
                )
                for issue in issues[:30]  # Limit results
                if not issue.pull_request  # Filter out pull requests
            ]
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to search issues: {e}") from e
    
    async def create_commit_comment(
        self, 
        repository: str, 
        commit_sha: str, 
        comment: str
    ) -> Dict[str, Any]:
        """
        Create a comment on a commit.
        
        Args:
            repository: Repository name in format "owner/repo"
            commit_sha: Commit SHA
            comment: Comment text
            
        Returns:
            Comment data
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            commit = repo.get_commit(commit_sha)
            comment_obj = commit.create_comment(comment)
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return {
                "id": comment_obj.id,
                "body": comment_obj.body,
                "user": comment_obj.user.login,
                "created_at": comment_obj.created_at.isoformat(),
                "html_url": comment_obj.html_url
            }
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to create commit comment: {e}") from e
    
    async def get_repository_languages(self, repository: str) -> Dict[str, int]:
        """
        Get repository languages.
        
        Args:
            repository: Repository name in format "owner/repo"
            
        Returns:
            Dictionary of languages and their byte counts
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            languages = repo.get_languages()
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return languages
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get repository languages: {e}") from e
    
    async def get_repository_topics(self, repository: str) -> List[str]:
        """
        Get repository topics.
        
        Args:
            repository: Repository name in format "owner/repo"
            
        Returns:
            List of topic names
        """
        await self._check_rate_limit()
        
        try:
            repo = self.client.get_repo(repository)
            topics = repo.get_topics()
            
            self.api_calls += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            return topics
            
        except GithubException as e:
            self.errors += 1
            raise GitHubError(f"Failed to get repository topics: {e}") from e
    
    def validate_webhook_signature(
        self, 
        payload: str, 
        signature: str, 
        secret: str
    ) -> bool:
        """
        Validate webhook signature.
        
        Args:
            payload: Webhook payload
            signature: Webhook signature
            secret: Webhook secret
            
        Returns:
            True if signature is valid
        """
        try:
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            received_signature = signature.replace('sha256=', '')
            
            return hmac.compare_digest(expected_signature, received_signature)
            
        except Exception as e:
            self.logger.error(f"Failed to validate webhook signature: {e}")
            return False
    
    def parse_webhook_payload(self, event_type: str, payload: Dict[str, Any]) -> WebhookData:
        """
        Parse webhook payload into structured data.
        
        Args:
            event_type: Type of webhook event
            payload: Webhook payload
            
        Returns:
            WebhookData object
        """
        try:
            action = payload.get("action", "")
            repository_data = None
            pull_request_data = None
            issue_data = None
            sender = None
            
            # Parse repository data
            if "repository" in payload:
                repo = payload["repository"]
                repository_data = RepositoryData(
                    id=repo["id"],
                    name=repo["name"],
                    full_name=repo["full_name"],
                    owner=repo["owner"]["login"],
                    description=repo.get("description"),
                    private=repo["private"],
                    language=repo.get("language"),
                    stars=repo["stargazers_count"],
                    forks=repo["forks_count"],
                    issues=repo["open_issues_count"],
                    default_branch=repo["default_branch"],
                    created_at=datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(repo["updated_at"].replace("Z", "+00:00")),
                    clone_url=repo["clone_url"],
                    html_url=repo["html_url"]
                )
            
            # Parse pull request data
            if "pull_request" in payload:
                pr = payload["pull_request"]
                pull_request_data = PullRequestData(
                    id=pr["id"],
                    number=pr["number"],
                    title=pr["title"],
                    body=pr.get("body"),
                    state=pr["state"],
                    user=pr["user"]["login"],
                    head_ref=pr["head"]["ref"],
                    base_ref=pr["base"]["ref"],
                    head_sha=pr["head"]["sha"],
                    base_sha=pr["base"]["sha"],
                    mergeable=pr.get("mergeable"),
                    merged=pr["merged"],
                    comments=pr["comments"],
                    commits=pr["commits"],
                    additions=pr["additions"],
                    deletions=pr["deletions"],
                    changed_files=pr["changed_files"],
                    created_at=datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00")),
                    merged_at=datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00")) if pr.get("merged_at") else None,
                    html_url=pr["html_url"],
                    diff_url=pr["diff_url"],
                    patch_url=pr["patch_url"]
                )
            
            # Parse issue data
            if "issue" in payload:
                issue = payload["issue"]
                issue_data = IssueData(
                    id=issue["id"],
                    number=issue["number"],
                    title=issue["title"],
                    body=issue.get("body"),
                    state=issue["state"],
                    user=issue["user"]["login"],
                    comments=issue["comments"],
                    created_at=datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(issue["updated_at"].replace("Z", "+00:00")),
                    closed_at=datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00")) if issue.get("closed_at") else None,
                    html_url=issue["html_url"],
                    labels=[label["name"] for label in issue.get("labels", [])],
                    assignees=[assignee["login"] for assignee in issue.get("assignees", [])]
                )
            
            # Parse sender data
            if "sender" in payload:
                sender = payload["sender"]["login"]
            
            return WebhookData(
                event=event_type,
                action=action,
                repository=repository_data,
                pull_request=pull_request_data,
                issue=issue_data,
                sender=sender
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse webhook payload: {e}")
            raise GitHubError(f"Failed to parse webhook payload: {e}") from e
    
    def should_process_webhook(self, webhook_data: WebhookData, allowed_events: List[str]) -> bool:
        """
        Check if webhook should be processed based on configuration.
        
        Args:
            webhook_data: Webhook data
            allowed_events: List of allowed event types
            
        Returns:
            True if webhook should be processed
        """
        return webhook_data.event in allowed_events
    
    async def _check_rate_limit(self) -> None:
        """Check and handle GitHub API rate limits."""
        try:
            current_time = time.time()
            
            # Check rate limit every 60 seconds
            if current_time - self.last_rate_limit_check > 60:
                await self._update_rate_limit()
                self.last_rate_limit_check = current_time
            
            # Check if we're approaching rate limit
            if self.rate_limit_remaining <= self.rate_limit_buffer:
                # Calculate wait time until reset
                wait_time = max(0, self.rate_limit_reset_time - current_time)
                
                if wait_time > 0:
                    self.logger.warning(f"Rate limit approached, waiting {wait_time:.1f} seconds")
                    raise GitHubRateLimitError(f"Rate limit exceeded, reset in {wait_time:.1f} seconds")
                else:
                    # Reset time has passed, update rate limit
                    await self._update_rate_limit()
            
        except GitHubRateLimitError:
            raise
        except Exception as e:
            self.logger.warning(f"Failed to check rate limit: {e}")
    
    async def _update_rate_limit(self) -> None:
        """Update rate limit information from GitHub API."""
        try:
            if self.client:
                rate_limit = self.client.get_rate_limit()
                self.rate_limit_remaining = rate_limit.core.remaining
                self.rate_limit_reset_time = rate_limit.core.reset.timestamp()
                
        except Exception as e:
            self.logger.error(f"Failed to update rate limit: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.
        
        Returns:
            Dictionary containing service metrics
        """
        uptime = time.time() - self.start_time
        
        return {
            "api_calls": self.api_calls,
            "errors": self.errors,
            "rate_limit_remaining": self.rate_limit_remaining,
            "success_rate": (self.api_calls - self.errors) / max(1, self.api_calls),
            "uptime": uptime,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "is_connected": self.is_connected
        }