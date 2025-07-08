"""
Agent-related database models.

This module defines models for agent instances, tasks, metrics, and configuration
in the autonomous agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, validator
from sqlmodel import Column, ForeignKey, Index, Relationship, SQLModel, String, Text
from sqlalchemy import Boolean, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

from .base import BaseModel, SoftDeleteMixin, TimestampMixin, AuditMixin


class AgentType(str, Enum):
    """Agent type enumeration."""
    GMAIL = "gmail"
    RESEARCH = "research"
    CODE = "code"
    INTELLIGENCE = "intelligence"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    BUSY = "busy"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class Agent(BaseModel, SoftDeleteMixin, AuditMixin, table=True):
    """
    Agent instance model.
    
    Represents an individual agent in the autonomous agent system with
    configuration, status tracking, and performance metrics.
    """
    
    __tablename__ = "agents"
    
    # Agent identification
    agent_id: str = Field(
        unique=True,
        index=True,
        description="Unique agent identifier"
    )
    
    name: str = Field(
        description="Human-readable agent name"
    )
    
    agent_type: AgentType = Field(
        index=True,
        description="Type of agent"
    )
    
    # Agent status
    status: AgentStatus = Field(
        default=AgentStatus.INACTIVE,
        index=True,
        description="Current agent status"
    )
    
    # Timing
    started_at: Optional[datetime] = Field(
        default=None,
        description="When agent was started"
    )
    
    stopped_at: Optional[datetime] = Field(
        default=None,
        description="When agent was stopped"
    )
    
    last_heartbeat: Optional[datetime] = Field(
        default=None,
        description="Last heartbeat timestamp"
    )
    
    # Performance metrics
    uptime_seconds: int = Field(
        default=0,
        description="Total uptime in seconds"
    )
    
    tasks_completed: int = Field(
        default=0,
        description="Number of tasks completed"
    )
    
    tasks_failed: int = Field(
        default=0,
        description="Number of tasks failed"
    )
    
    messages_processed: int = Field(
        default=0,
        description="Number of messages processed"
    )
    
    errors_encountered: int = Field(
        default=0,
        description="Number of errors encountered"
    )
    
    # Current workload
    current_tasks: int = Field(
        default=0,
        description="Number of currently active tasks"
    )
    
    max_concurrent_tasks: int = Field(
        default=5,
        description="Maximum concurrent tasks"
    )
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(
        default=None,
        description="Current memory usage in MB"
    )
    
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        description="Current CPU usage percentage"
    )
    
    # Agent capabilities
    capabilities: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of agent capabilities"
    )
    
    supported_tasks: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of supported task types"
    )
    
    # Health information
    health_status: str = Field(
        default="unknown",
        description="Health status"
    )
    
    last_health_check: Optional[datetime] = Field(
        default=None,
        description="Last health check timestamp"
    )
    
    # Version information
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    
    # Relationships
    configuration: Optional["AgentConfiguration"] = Relationship(
        back_populates="agent",
        cascade_delete=True
    )
    
    tasks: List["AgentTask"] = Relationship(
        back_populates="agent",
        cascade_delete=True
    )
    
    metrics: List["AgentMetrics"] = Relationship(
        back_populates="agent",
        cascade_delete=True
    )
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        """Validate agent ID format."""
        if not v:
            raise ValueError('Agent ID is required')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Agent ID can only contain letters, numbers, underscores, and hyphens')
        return v
    
    def is_active(self) -> bool:
        """Check if agent is active."""
        return self.status == AgentStatus.ACTIVE
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return self.health_status == "healthy"
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.is_active() and
            self.is_healthy() and
            self.current_tasks < self.max_concurrent_tasks
        )
    
    def calculate_success_rate(self) -> float:
        """Calculate task success rate."""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 0.0
        return self.tasks_completed / total_tasks
    
    def calculate_utilization(self) -> float:
        """Calculate current utilization."""
        if self.max_concurrent_tasks == 0:
            return 0.0
        return self.current_tasks / self.max_concurrent_tasks
    
    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
    
    def start_agent(self) -> None:
        """Mark agent as started."""
        self.status = AgentStatus.ACTIVE
        self.started_at = datetime.utcnow()
        self.update_heartbeat()
    
    def stop_agent(self) -> None:
        """Mark agent as stopped."""
        self.status = AgentStatus.INACTIVE
        self.stopped_at = datetime.utcnow()
        if self.started_at:
            self.uptime_seconds += int((self.stopped_at - self.started_at).total_seconds())
    
    def increment_task_completed(self) -> None:
        """Increment completed task counter."""
        self.tasks_completed += 1
        self.current_tasks = max(0, self.current_tasks - 1)
    
    def increment_task_failed(self) -> None:
        """Increment failed task counter."""
        self.tasks_failed += 1
        self.current_tasks = max(0, self.current_tasks - 1)
    
    def add_current_task(self) -> None:
        """Add to current task count."""
        self.current_tasks += 1


class AgentTask(BaseModel, SoftDeleteMixin, table=True):
    """
    Agent task model.
    
    Represents a task assigned to an agent with tracking and result storage.
    """
    
    __tablename__ = "agent_tasks"
    
    # Foreign key to agent
    agent_id: UUID = Field(
        foreign_key="agents.id",
        index=True,
        description="Reference to the agent"
    )
    
    # Task identification
    task_id: str = Field(
        unique=True,
        index=True,
        description="Unique task identifier"
    )
    
    task_type: str = Field(
        index=True,
        description="Type of task"
    )
    
    # Task metadata
    title: str = Field(
        description="Task title"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Task description"
    )
    
    # Task parameters
    parameters: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Task parameters"
    )
    
    # Task priority and scheduling
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL,
        index=True,
        description="Task priority"
    )
    
    scheduled_for: Optional[datetime] = Field(
        default=None,
        description="When task is scheduled to run"
    )
    
    deadline: Optional[datetime] = Field(
        default=None,
        description="Task deadline"
    )
    
    # Task status
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        index=True,
        description="Current task status"
    )
    
    # Timing
    queued_at: Optional[datetime] = Field(
        default=None,
        description="When task was queued"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="When task execution started"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When task completed"
    )
    
    # Execution information
    execution_duration: Optional[float] = Field(
        default=None,
        description="Task execution duration in seconds"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries"
    )
    
    # Results
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Task result data"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if task failed"
    )
    
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Detailed error information"
    )
    
    # Progress tracking
    progress_percent: float = Field(
        default=0.0,
        description="Task completion percentage"
    )
    
    progress_message: Optional[str] = Field(
        default=None,
        description="Current progress message"
    )
    
    # Dependencies
    depends_on: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of task IDs this task depends on"
    )
    
    # Relationships
    agent: Agent = Relationship(back_populates="tasks")
    
    def is_pending(self) -> bool:
        """Check if task is pending."""
        return self.status == TaskStatus.PENDING
    
    def is_running(self) -> bool:
        """Check if task is running."""
        return self.status == TaskStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status == TaskStatus.FAILED
    
    def is_overdue(self) -> bool:
        """Check if task is overdue."""
        if not self.deadline:
            return False
        return datetime.utcnow() > self.deadline
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def start_execution(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.progress_percent = 0.0
    
    def complete_task(self, result: Dict[str, Any]) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress_percent = 100.0
        
        if self.started_at:
            self.execution_duration = (self.completed_at - self.started_at).total_seconds()
    
    def fail_task(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_details = error_details
        
        if self.started_at:
            self.execution_duration = (self.completed_at - self.started_at).total_seconds()
    
    def update_progress(self, percent: float, message: Optional[str] = None) -> None:
        """Update task progress."""
        self.progress_percent = min(max(percent, 0.0), 100.0)
        if message:
            self.progress_message = message
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING


class AgentMetrics(BaseModel, table=True):
    """
    Agent metrics model.
    
    Stores performance and operational metrics for agents.
    """
    
    __tablename__ = "agent_metrics"
    
    # Foreign key to agent
    agent_id: UUID = Field(
        foreign_key="agents.id",
        index=True,
        description="Reference to the agent"
    )
    
    # Metric information
    metric_name: str = Field(
        description="Name of the metric"
    )
    
    metric_value: float = Field(
        description="Numeric value of the metric"
    )
    
    metric_unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement"
    )
    
    # Metric metadata
    metric_type: str = Field(
        description="Type of metric (gauge, counter, histogram, etc.)"
    )
    
    category: str = Field(
        description="Metric category"
    )
    
    # Timing
    recorded_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When metric was recorded"
    )
    
    # Additional data
    tags: Dict[str, str] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Additional metric tags"
    )
    
    # Relationships
    agent: Agent = Relationship(back_populates="metrics")
    
    class Config:
        """Model configuration."""
        # Ensure unique metric per agent and timestamp
        indexes = [
            Index(
                "idx_agent_metrics_agent_name_time",
                "agent_id",
                "metric_name",
                "recorded_at"
            )
        ]


class AgentConfiguration(BaseModel, SoftDeleteMixin, table=True):
    """
    Agent configuration model.
    
    Stores configuration settings for agents.
    """
    
    __tablename__ = "agent_configurations"
    
    # Foreign key to agent
    agent_id: UUID = Field(
        foreign_key="agents.id",
        unique=True,
        description="Reference to the agent"
    )
    
    # Configuration data
    config_data: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Agent configuration data"
    )
    
    # Configuration metadata
    config_version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )
    
    is_active: bool = Field(
        default=True,
        description="Whether configuration is active"
    )
    
    # Validation
    is_valid: bool = Field(
        default=True,
        description="Whether configuration is valid"
    )
    
    validation_errors: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Configuration validation errors"
    )
    
    # Environment
    environment: str = Field(
        default="production",
        description="Environment this configuration is for"
    )
    
    # Relationships
    agent: Agent = Relationship(back_populates="configuration")
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        # Placeholder for configuration validation logic
        self.is_valid = True
        self.validation_errors = None
        return self.is_valid
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config_data.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config_data[key] = value
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config_data.update(updates)
        self.validate_config()