"""
Data models for the Code Agent.

This module defines all data models and structures used by the Code Agent
for representing GitHub data, code analysis results, workflow configurations,
and other agent-specific data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


class ReviewStatus(Enum):
    """Pull request review status."""
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"


class WorkflowType(Enum):
    """Workflow automation types."""
    AUTO_MERGE = "auto_merge"
    AUTO_DEPLOY = "auto_deploy"
    CI_INTEGRATION = "ci_integration"
    NOTIFICATION = "notification"
    SECURITY_SCAN = "security_scan"
    DOCUMENTATION = "documentation"


class MonitoringEvent(Enum):
    """Repository monitoring events."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    ISSUES = "issues"
    REPOSITORY = "repository"
    RELEASE = "release"
    COMMIT_COMMENT = "commit_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"


@dataclass
class CodeReviewRequest:
    """Request for code review analysis."""
    repository: str
    pull_request: int
    pr_data: Dict[str, Any]
    files: List[Dict[str, Any]]
    review_type: str = "full"
    focus_areas: List[str] = field(default_factory=lambda: ["security", "performance", "style"])
    requester: Optional[str] = None
    priority: str = "normal"
    custom_rules: List[str] = field(default_factory=list)
    exclude_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository": self.repository,
            "pull_request": self.pull_request,
            "pr_data": self.pr_data,
            "files": self.files,
            "review_type": self.review_type,
            "focus_areas": self.focus_areas,
            "requester": self.requester,
            "priority": self.priority,
            "custom_rules": self.custom_rules,
            "exclude_files": self.exclude_files
        }


@dataclass
class CodeReviewResult:
    """Result of code review analysis."""
    review_id: str
    repository: str
    pull_request: int
    status: ReviewStatus
    overall_score: float
    summary: str
    comments: List[Dict[str, Any]]
    security_issues: List[Dict[str, Any]]
    performance_issues: List[Dict[str, Any]]
    style_issues: List[Dict[str, Any]]
    recommendations: List[str]
    files_analyzed: int
    analysis_duration: float
    reviewer: str = "ai_code_agent"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "repository": self.repository,
            "pull_request": self.pull_request,
            "status": self.status.value,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "comments": self.comments,
            "security_issues": self.security_issues,
            "performance_issues": self.performance_issues,
            "style_issues": self.style_issues,
            "recommendations": self.recommendations,
            "files_analyzed": self.files_analyzed,
            "analysis_duration": self.analysis_duration,
            "reviewer": self.reviewer,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class VulnerabilityReport:
    """Security vulnerability report."""
    scan_id: str
    repository: str
    branch: str
    scan_type: str
    vulnerabilities: List[Dict[str, Any]]
    summary: Dict[str, Any]
    risk_score: float
    recommendations: List[str]
    scan_duration: float
    scanned_files: List[str]
    dependencies_scanned: bool = False
    dependency_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_id": self.scan_id,
            "repository": self.repository,
            "branch": self.branch,
            "scan_type": self.scan_type,
            "vulnerabilities": self.vulnerabilities,
            "summary": self.summary,
            "risk_score": self.risk_score,
            "recommendations": self.recommendations,
            "scan_duration": self.scan_duration,
            "scanned_files": self.scanned_files,
            "dependencies_scanned": self.dependencies_scanned,
            "dependency_vulnerabilities": self.dependency_vulnerabilities,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class RepositoryMonitoringConfig:
    """Configuration for repository monitoring."""
    repository: str
    enabled: bool = True
    events: List[MonitoringEvent] = field(default_factory=lambda: [
        MonitoringEvent.PUSH, 
        MonitoringEvent.PULL_REQUEST, 
        MonitoringEvent.ISSUES
    ])
    polling_interval: int = 300  # seconds
    auto_review_prs: bool = True
    auto_scan_security: bool = True
    notify_on_vulnerabilities: bool = True
    workflow_automations: List[WorkflowType] = field(default_factory=list)
    branch_filters: List[str] = field(default_factory=lambda: ["main", "master", "develop"])
    file_filters: List[str] = field(default_factory=list)  # File patterns to monitor
    ignore_patterns: List[str] = field(default_factory=lambda: ["*.md", "*.txt", "*.json"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository": self.repository,
            "enabled": self.enabled,
            "events": [event.value for event in self.events],
            "polling_interval": self.polling_interval,
            "auto_review_prs": self.auto_review_prs,
            "auto_scan_security": self.auto_scan_security,
            "notify_on_vulnerabilities": self.notify_on_vulnerabilities,
            "workflow_automations": [workflow.value for workflow in self.workflow_automations],
            "branch_filters": self.branch_filters,
            "file_filters": self.file_filters,
            "ignore_patterns": self.ignore_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryMonitoringConfig":
        """Create from dictionary."""
        return cls(
            repository=data["repository"],
            enabled=data.get("enabled", True),
            events=[MonitoringEvent(event) for event in data.get("events", ["push", "pull_request", "issues"])],
            polling_interval=data.get("polling_interval", 300),
            auto_review_prs=data.get("auto_review_prs", True),
            auto_scan_security=data.get("auto_scan_security", True),
            notify_on_vulnerabilities=data.get("notify_on_vulnerabilities", True),
            workflow_automations=[WorkflowType(wf) for wf in data.get("workflow_automations", [])],
            branch_filters=data.get("branch_filters", ["main", "master", "develop"]),
            file_filters=data.get("file_filters", []),
            ignore_patterns=data.get("ignore_patterns", ["*.md", "*.txt", "*.json"])
        )


@dataclass
class WebhookEvent:
    """GitHub webhook event data."""
    event_type: str
    action: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    
    def mark_processing_started(self) -> None:
        """Mark event as processing started."""
        self.processing_started_at = datetime.now(timezone.utc)
    
    def mark_processing_completed(self) -> None:
        """Mark event as processing completed."""
        self.processing_completed_at = datetime.now(timezone.utc)
        self.processed = True
    
    def mark_processing_failed(self, error: str) -> None:
        """Mark event processing as failed."""
        self.processing_error = error
        self.processing_completed_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "action": self.action,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "processing_error": self.processing_error
        }


@dataclass
class WorkflowAutomationConfig:
    """Configuration for workflow automation."""
    workflow_type: WorkflowType
    repository: str
    enabled: bool = True
    triggers: List[str] = field(default_factory=list)  # Events that trigger this workflow
    conditions: Dict[str, Any] = field(default_factory=dict)  # Conditions for execution
    actions: List[Dict[str, Any]] = field(default_factory=list)  # Actions to perform
    notification_channels: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_type": self.workflow_type.value,
            "repository": self.repository,
            "enabled": self.enabled,
            "triggers": self.triggers,
            "conditions": self.conditions,
            "actions": self.actions,
            "notification_channels": self.notification_channels,
            "retry_count": self.retry_count,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowAutomationConfig":
        """Create from dictionary."""
        return cls(
            workflow_type=WorkflowType(data["workflow_type"]),
            repository=data["repository"],
            enabled=data.get("enabled", True),
            triggers=data.get("triggers", []),
            conditions=data.get("conditions", {}),
            actions=data.get("actions", []),
            notification_channels=data.get("notification_channels", []),
            retry_count=data.get("retry_count", 3),
            timeout=data.get("timeout", 300)
        )


@dataclass
class WorkflowExecution:
    """Workflow execution record."""
    execution_id: str
    workflow_type: WorkflowType
    repository: str
    trigger_event: str
    status: str  # "pending", "running", "completed", "failed"
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    
    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark execution as completed."""
        self.completed_at = datetime.now(timezone.utc)
        self.duration = (self.completed_at - self.started_at).total_seconds()
        self.status = "completed"
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark execution as failed."""
        self.completed_at = datetime.now(timezone.utc)
        self.duration = (self.completed_at - self.started_at).total_seconds()
        self.status = "failed"
        self.error = error
    
    def add_log(self, message: str) -> None:
        """Add log message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "workflow_type": self.workflow_type.value,
            "repository": self.repository,
            "trigger_event": self.trigger_event,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "result": self.result,
            "error": self.error,
            "logs": self.logs
        }


@dataclass
class DocumentationTask:
    """Documentation generation task."""
    task_id: str
    repository: str
    files: List[str]
    doc_type: str = "api"  # "api", "user", "developer"
    format_type: str = "markdown"  # "markdown", "html", "rst"
    output_path: Optional[str] = None
    template: Optional[str] = None
    include_examples: bool = True
    include_diagrams: bool = False
    language: Optional[str] = None
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "repository": self.repository,
            "files": self.files,
            "doc_type": self.doc_type,
            "format_type": self.format_type,
            "output_path": self.output_path,
            "template": self.template,
            "include_examples": self.include_examples,
            "include_diagrams": self.include_diagrams,
            "language": self.language,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }


@dataclass
class CodeMetrics:
    """Code quality metrics."""
    repository: str
    branch: str
    language: str
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    test_coverage: Optional[float] = None
    code_smells: int = 0
    technical_debt: float = 0.0
    duplicate_code_percentage: float = 0.0
    security_hotspots: int = 0
    performance_issues: int = 0
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository": self.repository,
            "branch": self.branch,
            "language": self.language,
            "lines_of_code": self.lines_of_code,
            "complexity_score": self.complexity_score,
            "maintainability_index": self.maintainability_index,
            "test_coverage": self.test_coverage,
            "code_smells": self.code_smells,
            "technical_debt": self.technical_debt,
            "duplicate_code_percentage": self.duplicate_code_percentage,
            "security_hotspots": self.security_hotspots,
            "performance_issues": self.performance_issues,
            "calculated_at": self.calculated_at.isoformat()
        }


@dataclass
class AgentTask:
    """Generic agent task."""
    task_id: str
    task_type: str
    repository: str
    parameters: Dict[str, Any]
    priority: str = "normal"  # "low", "normal", "high", "critical"
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def mark_started(self) -> None:
        """Mark task as started."""
        self.started_at = datetime.now(timezone.utc)
        self.status = "processing"
    
    def mark_completed(self, result: Dict[str, Any]) -> None:
        """Mark task as completed."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = "completed"
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """Mark task as failed."""
        self.completed_at = datetime.now(timezone.utc)
        self.status = "failed"
        self.error = error
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
        self.status = "pending"
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "repository": self.repository,
            "parameters": self.parameters,
            "priority": self.priority,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTask":
        """Create from dictionary."""
        task = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            repository=data["repository"],
            parameters=data["parameters"],
            priority=data.get("priority", "normal"),
            status=data.get("status", "pending"),
            assigned_to=data.get("assigned_to"),
            created_by=data.get("created_by"),
            created_at=datetime.fromisoformat(data["created_at"]),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )
        
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return task


@dataclass
class NotificationConfig:
    """Notification configuration."""
    channel_type: str  # "email", "slack", "webhook", "github"
    enabled: bool = True
    events: List[str] = field(default_factory=list)  # Events to notify on
    recipients: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)  # Channel-specific settings
    filters: Dict[str, Any] = field(default_factory=dict)  # Filtering criteria
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_type": self.channel_type,
            "enabled": self.enabled,
            "events": self.events,
            "recipients": self.recipients,
            "settings": self.settings,
            "filters": self.filters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationConfig":
        """Create from dictionary."""
        return cls(
            channel_type=data["channel_type"],
            enabled=data.get("enabled", True),
            events=data.get("events", []),
            recipients=data.get("recipients", []),
            settings=data.get("settings", {}),
            filters=data.get("filters", {})
        )


# Utility functions for creating tasks
def create_code_review_task(repository: str, pr_number: int, **kwargs) -> AgentTask:
    """Create a code review task."""
    return AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="code_review",
        repository=repository,
        parameters={
            "pull_request": pr_number,
            **kwargs
        }
    )


def create_security_scan_task(repository: str, branch: str = "main", **kwargs) -> AgentTask:
    """Create a security scan task."""
    return AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="vulnerability_scan",
        repository=repository,
        parameters={
            "branch": branch,
            **kwargs
        }
    )


def create_documentation_task(repository: str, files: List[str], **kwargs) -> AgentTask:
    """Create a documentation generation task."""
    return AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="generate_documentation",
        repository=repository,
        parameters={
            "files": files,
            **kwargs
        }
    )


def create_monitoring_task(repository: str, action: str = "add", **kwargs) -> AgentTask:
    """Create a repository monitoring task."""
    return AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="monitor_repository",
        repository=repository,
        parameters={
            "action": action,
            **kwargs
        }
    )