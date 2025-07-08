"""
Research-related database models.

This module defines models for research queries, results, sources, and knowledge base
management in the autonomous agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, HttpUrl, validator
from sqlmodel import Column, ForeignKey, Index, Relationship, SQLModel, String, Text
from sqlalchemy import Boolean, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

from .base import BaseModel, SoftDeleteMixin, SearchableMixin, VersionedMixin, AuditMixin


class ResearchStatus(str, Enum):
    """Research query status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchPriority(str, Enum):
    """Research priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ResearchType(str, Enum):
    """Research type enumeration."""
    GENERAL = "general"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    MARKET = "market"
    COMPETITIVE = "competitive"
    TREND = "trend"
    FACTUAL = "factual"
    ANALYSIS = "analysis"


class SourceType(str, Enum):
    """Source type enumeration."""
    WEBSITE = "website"
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    DOCUMENTATION = "documentation"
    SOCIAL = "social"
    FORUM = "forum"
    VIDEO = "video"
    PODCAST = "podcast"
    BOOK = "book"
    REPORT = "report"
    API = "api"
    DATABASE = "database"


class SourceReliability(str, Enum):
    """Source reliability enumeration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ResearchQuery(BaseModel, SoftDeleteMixin, VersionedMixin, AuditMixin, table=True):
    """
    Research query model.
    
    Represents a research request with parameters, progress tracking,
    and result management.
    """
    
    __tablename__ = "research_queries"
    
    # Query identification
    query_hash: str = Field(
        unique=True,
        index=True,
        description="Hash of the query for deduplication"
    )
    
    # Query content
    query_text: str = Field(
        max_length=2000,
        description="Original research query text"
    )
    
    refined_query: Optional[str] = Field(
        default=None,
        description="AI-refined query for better results"
    )
    
    # Query metadata
    research_type: ResearchType = Field(
        default=ResearchType.GENERAL,
        description="Type of research being conducted"
    )
    
    priority: ResearchPriority = Field(
        default=ResearchPriority.NORMAL,
        description="Research priority level"
    )
    
    status: ResearchStatus = Field(
        default=ResearchStatus.PENDING,
        index=True,
        description="Current status of the research"
    )
    
    # Query parameters
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return"
    )
    
    depth_level: int = Field(
        default=1,
        description="Depth of research (1=surface, 5=deep)"
    )
    
    time_range: Optional[str] = Field(
        default=None,
        description="Time range for research (e.g., '1y', '6m', '30d')"
    )
    
    language: str = Field(
        default="en",
        description="Language preference for results"
    )
    
    # Source preferences
    preferred_sources: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of preferred source domains"
    )
    
    excluded_sources: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of excluded source domains"
    )
    
    source_types: List[SourceType] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Preferred source types"
    )
    
    # Quality filters
    min_reliability: SourceReliability = Field(
        default=SourceReliability.MEDIUM,
        description="Minimum required source reliability"
    )
    
    academic_only: bool = Field(
        default=False,
        description="Whether to only return academic sources"
    )
    
    # Timing
    deadline: Optional[datetime] = Field(
        default=None,
        description="Research deadline"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="When research started"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When research completed"
    )
    
    # Processing information
    processed_by: Optional[UUID] = Field(
        default=None,
        description="Agent that processed the query"
    )
    
    processing_duration: Optional[float] = Field(
        default=None,
        description="Time taken to complete research (seconds)"
    )
    
    # Results summary
    total_results: int = Field(
        default=0,
        description="Total number of results found"
    )
    
    unique_sources: int = Field(
        default=0,
        description="Number of unique sources found"
    )
    
    # Quality metrics
    average_relevance: Optional[float] = Field(
        default=None,
        description="Average relevance score of results"
    )
    
    average_reliability: Optional[float] = Field(
        default=None,
        description="Average reliability score of sources"
    )
    
    # Generated content
    summary: Optional[str] = Field(
        default=None,
        description="AI-generated summary of findings"
    )
    
    key_findings: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of key findings"
    )
    
    recommendations: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of recommendations"
    )
    
    # Additional metadata
    research_context: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Additional research context"
    )
    
    # Relationships
    results: List["ResearchResult"] = Relationship(
        back_populates="query",
        cascade_delete=True
    )
    
    knowledge_entries: List["KnowledgeBase"] = Relationship(
        back_populates="source_query",
        cascade_delete=True
    )
    
    @validator('query_text')
    def validate_query_text(cls, v):
        """Validate query text."""
        if not v or len(v.strip()) < 3:
            raise ValueError('Query text must be at least 3 characters long')
        return v.strip()
    
    def is_overdue(self) -> bool:
        """Check if research is overdue."""
        if not self.deadline:
            return False
        return datetime.utcnow() > self.deadline
    
    def is_completed(self) -> bool:
        """Check if research is completed."""
        return self.status == ResearchStatus.COMPLETED
    
    def is_in_progress(self) -> bool:
        """Check if research is in progress."""
        return self.status == ResearchStatus.IN_PROGRESS
    
    def mark_as_started(self, agent_id: UUID) -> None:
        """Mark research as started."""
        self.status = ResearchStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
        self.processed_by = agent_id
    
    def mark_as_completed(self, duration: float) -> None:
        """Mark research as completed."""
        self.status = ResearchStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.processing_duration = duration
    
    def calculate_progress(self) -> float:
        """Calculate research progress percentage."""
        if self.status == ResearchStatus.COMPLETED:
            return 100.0
        elif self.status == ResearchStatus.IN_PROGRESS:
            # Calculate based on results found vs max_results
            if self.max_results > 0:
                return min(self.total_results / self.max_results * 100, 99.0)
            return 50.0
        return 0.0


class ResearchResult(BaseModel, SoftDeleteMixin, SearchableMixin, table=True):
    """
    Research result model.
    
    Represents an individual research result with content, metadata,
    and quality scoring.
    """
    
    __tablename__ = "research_results"
    
    # Foreign key to query
    query_id: UUID = Field(
        foreign_key="research_queries.id",
        description="Reference to the research query"
    )
    
    # Result identification
    result_hash: str = Field(
        unique=True,
        index=True,
        description="Hash of the result for deduplication"
    )
    
    # Content
    title: str = Field(
        max_length=500,
        description="Result title"
    )
    
    content: str = Field(
        description="Main content of the result"
    )
    
    summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Summary of the result"
    )
    
    # Source information
    source_url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of the source"
    )
    
    source_domain: Optional[str] = Field(
        default=None,
        description="Domain of the source"
    )
    
    source_type: SourceType = Field(
        description="Type of source"
    )
    
    source_title: Optional[str] = Field(
        default=None,
        description="Title of the source"
    )
    
    # Author information
    author: Optional[str] = Field(
        default=None,
        description="Author of the content"
    )
    
    author_credentials: Optional[str] = Field(
        default=None,
        description="Author's credentials"
    )
    
    # Publication information
    published_at: Optional[datetime] = Field(
        default=None,
        description="When the content was published"
    )
    
    last_updated: Optional[datetime] = Field(
        default=None,
        description="When the content was last updated"
    )
    
    # Quality metrics
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score (0-1)"
    )
    
    reliability_score: float = Field(
        default=0.0,
        description="Reliability score (0-1)"
    )
    
    credibility_score: float = Field(
        default=0.0,
        description="Credibility score (0-1)"
    )
    
    freshness_score: float = Field(
        default=0.0,
        description="Freshness score (0-1)"
    )
    
    overall_score: float = Field(
        default=0.0,
        description="Overall quality score (0-1)"
    )
    
    # Content analysis
    word_count: int = Field(
        default=0,
        description="Number of words in content"
    )
    
    reading_time: int = Field(
        default=0,
        description="Estimated reading time in minutes"
    )
    
    language: str = Field(
        default="en",
        description="Detected language of content"
    )
    
    # Extracted information
    entities: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Extracted entities from content"
    )
    
    keywords: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Extracted keywords"
    )
    
    topics: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Identified topics"
    )
    
    # Citations and references
    citations: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="References cited in the content"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Additional metadata"
    )
    
    # Relationships
    query: ResearchQuery = Relationship(back_populates="results")
    
    source: Optional["ResearchSource"] = Relationship(
        back_populates="results"
    )
    
    def calculate_overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = [
            self.relevance_score,
            self.reliability_score,
            self.credibility_score,
            self.freshness_score
        ]
        
        # Weighted average (relevance is most important)
        weights = [0.4, 0.3, 0.2, 0.1]
        
        self.overall_score = sum(score * weight for score, weight in zip(scores, weights))
        return self.overall_score
    
    def is_high_quality(self) -> bool:
        """Check if result is high quality."""
        return self.overall_score >= 0.7
    
    def is_recent(self, days: int = 30) -> bool:
        """Check if result is recent."""
        if not self.published_at:
            return False
        return (datetime.utcnow() - self.published_at).days <= days


class ResearchSource(BaseModel, SoftDeleteMixin, VersionedMixin, table=True):
    """
    Research source model.
    
    Represents a source of research information with quality metrics
    and usage tracking.
    """
    
    __tablename__ = "research_sources"
    
    # Source identification
    domain: str = Field(
        unique=True,
        index=True,
        description="Source domain"
    )
    
    name: str = Field(
        description="Human-readable source name"
    )
    
    # Source metadata
    source_type: SourceType = Field(
        description="Type of source"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Description of the source"
    )
    
    # Quality metrics
    reliability: SourceReliability = Field(
        default=SourceReliability.UNKNOWN,
        description="Reliability rating"
    )
    
    credibility_score: float = Field(
        default=0.0,
        description="Credibility score (0-1)"
    )
    
    authority_score: float = Field(
        default=0.0,
        description="Authority score (0-1)"
    )
    
    # Usage statistics
    total_results: int = Field(
        default=0,
        description="Total results from this source"
    )
    
    successful_results: int = Field(
        default=0,
        description="Number of successful results"
    )
    
    last_used: Optional[datetime] = Field(
        default=None,
        description="When source was last used"
    )
    
    # Access information
    requires_auth: bool = Field(
        default=False,
        description="Whether source requires authentication"
    )
    
    rate_limit: Optional[int] = Field(
        default=None,
        description="Rate limit (requests per minute)"
    )
    
    # Additional metadata
    tags: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Tags for categorization"
    )
    
    # Relationships
    results: List[ResearchResult] = Relationship(
        back_populates="source"
    )
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate for this source."""
        if self.total_results == 0:
            return 0.0
        return self.successful_results / self.total_results
    
    def is_reliable(self) -> bool:
        """Check if source is reliable."""
        return self.reliability in [SourceReliability.HIGH, SourceReliability.MEDIUM]
    
    def update_usage_stats(self, successful: bool = True) -> None:
        """Update usage statistics."""
        self.total_results += 1
        if successful:
            self.successful_results += 1
        self.last_used = datetime.utcnow()


class KnowledgeBase(BaseModel, SoftDeleteMixin, SearchableMixin, VersionedMixin, table=True):
    """
    Knowledge base model.
    
    Represents processed and stored knowledge from research results
    for future reference and learning.
    """
    
    __tablename__ = "knowledge_base"
    
    # Knowledge identification
    knowledge_hash: str = Field(
        unique=True,
        index=True,
        description="Hash of the knowledge for deduplication"
    )
    
    # Content
    title: str = Field(
        max_length=500,
        description="Knowledge title"
    )
    
    content: str = Field(
        description="Knowledge content"
    )
    
    summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Summary of the knowledge"
    )
    
    # Categorization
    category: str = Field(
        max_length=100,
        description="Knowledge category"
    )
    
    topics: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Topics covered"
    )
    
    tags: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Tags for categorization"
    )
    
    # Quality and confidence
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in the knowledge (0-1)"
    )
    
    validation_status: str = Field(
        default="pending",
        description="Validation status"
    )
    
    # Source information
    source_query_id: Optional[UUID] = Field(
        default=None,
        foreign_key="research_queries.id",
        description="Original research query"
    )
    
    source_urls: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Source URLs"
    )
    
    # Usage tracking
    access_count: int = Field(
        default=0,
        description="Number of times accessed"
    )
    
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="When knowledge was last accessed"
    )
    
    # Relationships
    source_query: Optional[ResearchQuery] = Relationship(
        back_populates="knowledge_entries"
    )
    
    def increment_access(self) -> None:
        """Increment access count."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def is_validated(self) -> bool:
        """Check if knowledge is validated."""
        return self.validation_status == "validated"
    
    def is_high_confidence(self) -> bool:
        """Check if knowledge has high confidence."""
        return self.confidence_score >= 0.8