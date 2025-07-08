"""
Mock configurations for GitHub API testing.

This module provides comprehensive mocks for GitHub API interactions,
following the TDD approach and ensuring complete isolation from external services.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockGitHubRepository:
    """Mock GitHub repository with realistic data."""
    
    def __init__(self,
                 repo_id: int = 123456789,
                 name: str = "test-repo",
                 owner: str = "test-user",
                 description: str = "A test repository",
                 private: bool = False,
                 language: str = "Python",
                 stars: int = 10,
                 forks: int = 5,
                 issues: int = 2):
        
        self.repo_id = repo_id
        self.name = name
        self.owner = owner
        self.description = description
        self.private = private
        self.language = language
        self.stars = stars
        self.forks = forks
        self.issues = issues
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def to_github_format(self) -> Dict[str, Any]:
        """Convert to GitHub API format."""
        return {
            "id": self.repo_id,
            "name": self.name,
            "full_name": f"{self.owner}/{self.name}",
            "owner": {
                "login": self.owner,
                "id": 12345,
                "avatar_url": f"https://avatars.githubusercontent.com/{self.owner}",
                "url": f"https://api.github.com/users/{self.owner}",
                "html_url": f"https://github.com/{self.owner}",
                "type": "User",
            },
            "private": self.private,
            "html_url": f"https://github.com/{self.owner}/{self.name}",
            "description": self.description,
            "fork": False,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pushed_at": self.updated_at.isoformat(),
            "git_url": f"git://github.com/{self.owner}/{self.name}.git",
            "ssh_url": f"git@github.com:{self.owner}/{self.name}.git",
            "clone_url": f"https://github.com/{self.owner}/{self.name}.git",
            "svn_url": f"https://github.com/{self.owner}/{self.name}",
            "size": 1024,
            "stargazers_count": self.stars,
            "watchers_count": self.stars,
            "language": self.language,
            "has_issues": True,
            "has_projects": True,
            "has_wiki": True,
            "has_pages": False,
            "has_downloads": True,
            "archived": False,
            "disabled": False,
            "open_issues_count": self.issues,
            "license": {
                "key": "mit",
                "name": "MIT License",
                "url": "https://api.github.com/licenses/mit",
            },
            "forks": self.forks,
            "open_issues": self.issues,
            "watchers": self.stars,
            "default_branch": "main",
        }


class MockGitHubPullRequest:
    """Mock GitHub pull request with realistic data."""
    
    def __init__(self,
                 pr_id: int = 987654321,
                 number: int = 1,
                 title: str = "Test Pull Request",
                 body: str = "This is a test pull request",
                 state: str = "open",
                 user: str = "test-user",
                 head_ref: str = "feature-branch",
                 base_ref: str = "main",
                 head_sha: str = "abc123def456",
                 base_sha: str = "def456ghi789",
                 repo_name: str = "test-repo",
                 repo_owner: str = "test-user"):
        
        self.pr_id = pr_id
        self.number = number
        self.title = title
        self.body = body
        self.state = state
        self.user = user
        self.head_ref = head_ref
        self.base_ref = base_ref
        self.head_sha = head_sha
        self.base_sha = base_sha
        self.repo_name = repo_name
        self.repo_owner = repo_owner
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.mergeable = True
        self.comments = 0
        self.commits = 1
        self.additions = 10
        self.deletions = 5
        self.changed_files = 2
    
    def to_github_format(self) -> Dict[str, Any]:
        """Convert to GitHub API format."""
        return {
            "id": self.pr_id,
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "user": {
                "login": self.user,
                "id": 12345,
                "avatar_url": f"https://avatars.githubusercontent.com/{self.user}",
                "url": f"https://api.github.com/users/{self.user}",
                "html_url": f"https://github.com/{self.user}",
                "type": "User",
            },
            "assignee": None,
            "assignees": [],
            "requested_reviewers": [],
            "requested_teams": [],
            "labels": [],
            "milestone": None,
            "locked": False,
            "active_lock_reason": None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": None,
            "merged_at": None,
            "merge_commit_sha": None,
            "draft": False,
            "head": {
                "label": f"{self.user}:{self.head_ref}",
                "ref": self.head_ref,
                "sha": self.head_sha,
                "user": {
                    "login": self.user,
                    "id": 12345,
                    "type": "User",
                },
                "repo": {
                    "id": 123456789,
                    "name": self.repo_name,
                    "full_name": f"{self.repo_owner}/{self.repo_name}",
                    "owner": {"login": self.repo_owner, "id": 12345},
                    "private": False,
                    "default_branch": "main",
                },
            },
            "base": {
                "label": f"{self.repo_owner}:{self.base_ref}",
                "ref": self.base_ref,
                "sha": self.base_sha,
                "user": {
                    "login": self.repo_owner,
                    "id": 12345,
                    "type": "User",
                },
                "repo": {
                    "id": 123456789,
                    "name": self.repo_name,
                    "full_name": f"{self.repo_owner}/{self.repo_name}",
                    "owner": {"login": self.repo_owner, "id": 12345},
                    "private": False,
                    "default_branch": "main",
                },
            },
            "mergeable": self.mergeable,
            "mergeable_state": "clean" if self.mergeable else "dirty",
            "merged": False,
            "comments": self.comments,
            "review_comments": 0,
            "maintainer_can_modify": True,
            "commits": self.commits,
            "additions": self.additions,
            "deletions": self.deletions,
            "changed_files": self.changed_files,
            "url": f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls/{self.number}",
            "html_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{self.number}",
            "diff_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{self.number}.diff",
            "patch_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/pull/{self.number}.patch",
        }


class MockGitHubIssue:
    """Mock GitHub issue with realistic data."""
    
    def __init__(self,
                 issue_id: int = 456789123,
                 number: int = 1,
                 title: str = "Test Issue",
                 body: str = "This is a test issue",
                 state: str = "open",
                 user: str = "test-user",
                 labels: Optional[List[str]] = None,
                 assignees: Optional[List[str]] = None):
        
        self.issue_id = issue_id
        self.number = number
        self.title = title
        self.body = body
        self.state = state
        self.user = user
        self.labels = labels or []
        self.assignees = assignees or []
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.comments = 0
    
    def to_github_format(self) -> Dict[str, Any]:
        """Convert to GitHub API format."""
        return {
            "id": self.issue_id,
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "user": {
                "login": self.user,
                "id": 12345,
                "avatar_url": f"https://avatars.githubusercontent.com/{self.user}",
                "url": f"https://api.github.com/users/{self.user}",
                "html_url": f"https://github.com/{self.user}",
                "type": "User",
            },
            "labels": [{"name": label, "color": "ffffff"} for label in self.labels],
            "assignees": [{"login": assignee, "id": 12345} for assignee in self.assignees],
            "milestone": None,
            "locked": False,
            "active_lock_reason": None,
            "comments": self.comments,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": None,
            "author_association": "OWNER",
            "url": f"https://api.github.com/repos/test-user/test-repo/issues/{self.number}",
            "html_url": f"https://github.com/test-user/test-repo/issues/{self.number}",
        }


class MockGitHubCommit:
    """Mock GitHub commit with realistic data."""
    
    def __init__(self,
                 sha: str = "abc123def456ghi789",
                 message: str = "Test commit message",
                 author: str = "test-user",
                 author_email: str = "test@example.com"):
        
        self.sha = sha
        self.message = message
        self.author = author
        self.author_email = author_email
        self.created_at = datetime.now(timezone.utc)
    
    def to_github_format(self) -> Dict[str, Any]:
        """Convert to GitHub API format."""
        return {
            "sha": self.sha,
            "commit": {
                "message": self.message,
                "author": {
                    "name": self.author,
                    "email": self.author_email,
                    "date": self.created_at.isoformat(),
                },
                "committer": {
                    "name": self.author,
                    "email": self.author_email,
                    "date": self.created_at.isoformat(),
                },
                "tree": {
                    "sha": "tree123abc456def789",
                    "url": "https://api.github.com/repos/test-user/test-repo/git/trees/tree123abc456def789",
                },
                "url": f"https://api.github.com/repos/test-user/test-repo/git/commits/{self.sha}",
            },
            "author": {
                "login": self.author,
                "id": 12345,
                "avatar_url": f"https://avatars.githubusercontent.com/{self.author}",
                "url": f"https://api.github.com/users/{self.author}",
                "html_url": f"https://github.com/{self.author}",
                "type": "User",
            },
            "committer": {
                "login": self.author,
                "id": 12345,
                "avatar_url": f"https://avatars.githubusercontent.com/{self.author}",
                "url": f"https://api.github.com/users/{self.author}",
                "html_url": f"https://github.com/{self.author}",
                "type": "User",
            },
            "parents": [
                {
                    "sha": "parent123abc456def789",
                    "url": f"https://api.github.com/repos/test-user/test-repo/commits/parent123abc456def789",
                    "html_url": f"https://github.com/test-user/test-repo/commit/parent123abc456def789",
                }
            ],
            "url": f"https://api.github.com/repos/test-user/test-repo/commits/{self.sha}",
            "html_url": f"https://github.com/test-user/test-repo/commit/{self.sha}",
        }


class MockGitHubClient:
    """Mock GitHub client with realistic behavior."""
    
    def __init__(self):
        self.repositories = []
        self.pull_requests = []
        self.issues = []
        self.commits = []
        self.webhooks = []
        self.rate_limit = {
            "limit": 5000,
            "remaining": 4999,
            "reset": int(datetime.now(timezone.utc).timestamp()) + 3600,
        }
    
    def add_repository(self, repo: MockGitHubRepository) -> None:
        """Add a repository to the mock client."""
        self.repositories.append(repo)
    
    def add_pull_request(self, pr: MockGitHubPullRequest) -> None:
        """Add a pull request to the mock client."""
        self.pull_requests.append(pr)
    
    def add_issue(self, issue: MockGitHubIssue) -> None:
        """Add an issue to the mock client."""
        self.issues.append(issue)
    
    def add_commit(self, commit: MockGitHubCommit) -> None:
        """Add a commit to the mock client."""
        self.commits.append(commit)
    
    def create_client_mock(self) -> MagicMock:
        """Create a mock client object."""
        client = MagicMock()
        
        # Mock repository operations
        client.get_repo.side_effect = self._get_repo_mock
        client.get_user.side_effect = self._get_user_mock
        client.search_repositories.side_effect = self._search_repositories_mock
        client.search_issues.side_effect = self._search_issues_mock
        client.get_rate_limit.return_value = MagicMock(core=MagicMock(remaining=self.rate_limit["remaining"]))
        
        return client
    
    def _get_repo_mock(self, full_name: str) -> MagicMock:
        """Mock get_repo method."""
        repo = next((r for r in self.repositories if f"{r.owner}/{r.name}" == full_name), None)
        if repo:
            repo_mock = MagicMock()
            repo_mock.full_name = full_name
            repo_mock.name = repo.name
            repo_mock.owner = MagicMock(login=repo.owner)
            repo_mock.description = repo.description
            repo_mock.private = repo.private
            repo_mock.language = repo.language
            repo_mock.stargazers_count = repo.stars
            repo_mock.forks_count = repo.forks
            repo_mock.open_issues_count = repo.issues
            repo_mock.default_branch = "main"
            
            # Mock pull requests
            repo_mock.get_pulls.return_value = [
                self._create_pr_mock(pr) for pr in self.pull_requests
            ]
            
            # Mock issues
            repo_mock.get_issues.return_value = [
                self._create_issue_mock(issue) for issue in self.issues
            ]
            
            # Mock commits
            repo_mock.get_commits.return_value = [
                self._create_commit_mock(commit) for commit in self.commits
            ]
            
            return repo_mock
        
        raise Exception(f"Repository {full_name} not found")
    
    def _get_user_mock(self, login: str) -> MagicMock:
        """Mock get_user method."""
        user_mock = MagicMock()
        user_mock.login = login
        user_mock.id = 12345
        user_mock.avatar_url = f"https://avatars.githubusercontent.com/{login}"
        user_mock.html_url = f"https://github.com/{login}"
        user_mock.type = "User"
        return user_mock
    
    def _search_repositories_mock(self, query: str) -> MagicMock:
        """Mock search_repositories method."""
        search_mock = MagicMock()
        search_mock.totalCount = len(self.repositories)
        search_mock.__iter__ = lambda: iter([self._create_repo_mock(repo) for repo in self.repositories])
        return search_mock
    
    def _search_issues_mock(self, query: str) -> MagicMock:
        """Mock search_issues method."""
        search_mock = MagicMock()
        search_mock.totalCount = len(self.issues)
        search_mock.__iter__ = lambda: iter([self._create_issue_mock(issue) for issue in self.issues])
        return search_mock
    
    def _create_repo_mock(self, repo: MockGitHubRepository) -> MagicMock:
        """Create a mock repository object."""
        repo_mock = MagicMock()
        repo_mock.id = repo.repo_id
        repo_mock.name = repo.name
        repo_mock.full_name = f"{repo.owner}/{repo.name}"
        repo_mock.owner = MagicMock(login=repo.owner)
        repo_mock.description = repo.description
        repo_mock.private = repo.private
        repo_mock.language = repo.language
        repo_mock.stargazers_count = repo.stars
        repo_mock.forks_count = repo.forks
        repo_mock.open_issues_count = repo.issues
        return repo_mock
    
    def _create_pr_mock(self, pr: MockGitHubPullRequest) -> MagicMock:
        """Create a mock pull request object."""
        pr_mock = MagicMock()
        pr_mock.id = pr.pr_id
        pr_mock.number = pr.number
        pr_mock.title = pr.title
        pr_mock.body = pr.body
        pr_mock.state = pr.state
        pr_mock.user = MagicMock(login=pr.user)
        pr_mock.head = MagicMock(ref=pr.head_ref, sha=pr.head_sha)
        pr_mock.base = MagicMock(ref=pr.base_ref, sha=pr.base_sha)
        pr_mock.mergeable = pr.mergeable
        pr_mock.comments = pr.comments
        pr_mock.commits = pr.commits
        pr_mock.additions = pr.additions
        pr_mock.deletions = pr.deletions
        pr_mock.changed_files = pr.changed_files
        return pr_mock
    
    def _create_issue_mock(self, issue: MockGitHubIssue) -> MagicMock:
        """Create a mock issue object."""
        issue_mock = MagicMock()
        issue_mock.id = issue.issue_id
        issue_mock.number = issue.number
        issue_mock.title = issue.title
        issue_mock.body = issue.body
        issue_mock.state = issue.state
        issue_mock.user = MagicMock(login=issue.user)
        issue_mock.labels = [MagicMock(name=label) for label in issue.labels]
        issue_mock.assignees = [MagicMock(login=assignee) for assignee in issue.assignees]
        issue_mock.comments = issue.comments
        return issue_mock
    
    def _create_commit_mock(self, commit: MockGitHubCommit) -> MagicMock:
        """Create a mock commit object."""
        commit_mock = MagicMock()
        commit_mock.sha = commit.sha
        commit_mock.commit = MagicMock(
            message=commit.message,
            author=MagicMock(name=commit.author, email=commit.author_email),
        )
        commit_mock.author = MagicMock(login=commit.author)
        return commit_mock


class MockGitHubWebhook:
    """Mock GitHub webhook for testing webhook handling."""
    
    def __init__(self, event_type: str, payload: Dict[str, Any]):
        self.event_type = event_type
        self.payload = payload
        self.headers = {
            "X-GitHub-Event": event_type,
            "X-GitHub-Delivery": "test-delivery-uuid",
            "X-Hub-Signature-256": "sha256=test-signature",
            "Content-Type": "application/json",
        }
    
    def to_request_format(self) -> Dict[str, Any]:
        """Convert to HTTP request format."""
        return {
            "headers": self.headers,
            "body": json.dumps(self.payload),
            "method": "POST",
        }


# Pytest fixtures for GitHub mocks
@pytest.fixture
def mock_github_repo() -> MockGitHubRepository:
    """Provide a mock GitHub repository."""
    return MockGitHubRepository()


@pytest.fixture
def mock_github_pull_request() -> MockGitHubPullRequest:
    """Provide a mock GitHub pull request."""
    return MockGitHubPullRequest()


@pytest.fixture
def mock_github_issue() -> MockGitHubIssue:
    """Provide a mock GitHub issue."""
    return MockGitHubIssue()


@pytest.fixture
def mock_github_commit() -> MockGitHubCommit:
    """Provide a mock GitHub commit."""
    return MockGitHubCommit()


@pytest.fixture
def mock_github_client() -> MockGitHubClient:
    """Provide a mock GitHub client."""
    return MockGitHubClient()


@pytest.fixture
def mock_github_webhook() -> MockGitHubWebhook:
    """Provide a mock GitHub webhook."""
    payload = {
        "action": "opened",
        "pull_request": MockGitHubPullRequest().to_github_format(),
        "repository": MockGitHubRepository().to_github_format(),
        "sender": {"login": "test-user", "id": 12345},
    }
    return MockGitHubWebhook("pull_request", payload)


# Context managers for patching GitHub API
class MockGitHubAPIContext:
    """Context manager for mocking GitHub API calls."""
    
    def __init__(self, client: MockGitHubClient):
        self.client = client
        self.patches = []
    
    def __enter__(self):
        # Mock GitHub client
        github_patch = patch("github.Github")
        mock_github = github_patch.start()
        mock_github.return_value = self.client.create_client_mock()
        self.patches.append(github_patch)
        
        # Mock requests for webhook handling
        requests_patch = patch("requests.post")
        mock_requests = requests_patch.start()
        mock_requests.return_value = MagicMock(status_code=200, json=lambda: {"success": True})
        self.patches.append(requests_patch)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


# Decorator for GitHub API mocking
def mock_github_api(client: Optional[MockGitHubClient] = None):
    """Decorator to mock GitHub API for a test function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            mock_client = client or MockGitHubClient()
            with MockGitHubAPIContext(mock_client):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Sample data generators
def generate_sample_repositories(count: int = 5) -> List[MockGitHubRepository]:
    """Generate sample repositories for testing."""
    repos = []
    for i in range(count):
        repo = MockGitHubRepository(
            repo_id=123456789 + i,
            name=f"test-repo-{i+1}",
            owner=f"test-user-{i+1}",
            description=f"Test repository {i+1}",
            private=i % 2 == 0,
            language="Python" if i % 2 == 0 else "JavaScript",
            stars=10 + i,
            forks=5 + i,
            issues=2 + i,
        )
        repos.append(repo)
    return repos


def generate_sample_pull_requests(count: int = 3) -> List[MockGitHubPullRequest]:
    """Generate sample pull requests for testing."""
    prs = []
    for i in range(count):
        pr = MockGitHubPullRequest(
            pr_id=987654321 + i,
            number=i + 1,
            title=f"Feature: Add feature {i+1}",
            body=f"This pull request adds feature {i+1} to the codebase.",
            state="open" if i % 2 == 0 else "closed",
            user=f"contributor-{i+1}",
            head_ref=f"feature-{i+1}",
            base_ref="main",
            head_sha=f"abc123def456{i:03d}",
            base_sha=f"def456ghi789{i:03d}",
        )
        prs.append(pr)
    return prs


def generate_sample_issues(count: int = 4) -> List[MockGitHubIssue]:
    """Generate sample issues for testing."""
    issues = []
    for i in range(count):
        issue = MockGitHubIssue(
            issue_id=456789123 + i,
            number=i + 1,
            title=f"Bug: Issue {i+1}",
            body=f"This is a description of issue {i+1}.",
            state="open" if i % 2 == 0 else "closed",
            user=f"reporter-{i+1}",
            labels=["bug", "high-priority"] if i % 2 == 0 else ["enhancement"],
            assignees=[f"assignee-{i+1}"] if i % 2 == 0 else [],
        )
        issues.append(issue)
    return issues


def generate_pr_webhook_payload(action: str = "opened") -> Dict[str, Any]:
    """Generate a pull request webhook payload."""
    pr = MockGitHubPullRequest()
    repo = MockGitHubRepository()
    
    return {
        "action": action,
        "number": pr.number,
        "pull_request": pr.to_github_format(),
        "repository": repo.to_github_format(),
        "sender": {"login": "test-user", "id": 12345},
    }


def generate_issue_webhook_payload(action: str = "opened") -> Dict[str, Any]:
    """Generate an issue webhook payload."""
    issue = MockGitHubIssue()
    repo = MockGitHubRepository()
    
    return {
        "action": action,
        "issue": issue.to_github_format(),
        "repository": repo.to_github_format(),
        "sender": {"login": "test-user", "id": 12345},
    }