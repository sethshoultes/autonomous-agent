"""
Email-related database models.

This module defines models for email processing, analytics, and thread management
in the autonomous agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import EmailStr, Field, validator
from sqlmodel import Column, ForeignKey, Index, Relationship, SQLModel, String, Text
from sqlalchemy import Boolean, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

from .base import BaseModel, SoftDeleteMixin, SearchableMixin, AuditMixin


class EmailStatus(str, Enum):
    """Email processing status enumeration."""
    UNREAD = "unread"
    READ = "read"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    DELETED = "deleted"
    SPAM = "spam"
    ERROR = "error"


class EmailPriority(str, Enum):
    """Email priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EmailDirection(str, Enum):
    """Email direction enumeration."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class EmailCategory(str, Enum):
    """Email category enumeration."""
    PERSONAL = "personal"
    WORK = "work"
    PROMOTION = "promotion"
    NOTIFICATION = "notification"
    NEWSLETTER = "newsletter"
    MEETING = "meeting"
    TASK = "task"
    SUPPORT = "support"
    SPAM = "spam"
    OTHER = "other"


class Email(BaseModel, SoftDeleteMixin, SearchableMixin, AuditMixin, table=True):
    """
    Email message model.
    
    Represents an email message with comprehensive metadata for processing,
    analysis, and management by the autonomous agent system.
    """
    
    __tablename__ = "emails"
    
    # Gmail API integration
    gmail_id: Optional[str] = Field(
        default=None,
        unique=True,
        index=True,
        description="Gmail message ID"
    )
    
    gmail_thread_id: Optional[str] = Field(
        default=None,
        index=True,
        description="Gmail thread ID"
    )
    
    # Email metadata
    message_id: str = Field(
        unique=True,
        index=True,
        description="Email Message-ID header"
    )
    
    in_reply_to: Optional[str] = Field(
        default=None,
        description="Message-ID of the message this is replying to"
    )
    
    references: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of Message-IDs this email references"
    )
    
    # Email content
    subject: str = Field(
        max_length=1000,
        description="Email subject"
    )
    
    body_text: Optional[str] = Field(
        default=None,
        description="Plain text body content"
    )
    
    body_html: Optional[str] = Field(
        default=None,
        description="HTML body content"
    )
    
    snippet: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Email snippet/preview"
    )
    
    # Sender and recipient information
    sender_email: EmailStr = Field(
        index=True,
        description="Sender's email address"
    )
    
    sender_name: Optional[str] = Field(
        default=None,
        description="Sender's display name"
    )
    
    recipients_to: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of TO recipients"
    )
    
    recipients_cc: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of CC recipients"
    )
    
    recipients_bcc: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of BCC recipients"
    )
    
    # Email timing
    sent_at: datetime = Field(
        index=True,
        description="Timestamp when email was sent"
    )
    
    received_at: datetime = Field(
        index=True,
        description="Timestamp when email was received"
    )
    
    # Email status and categorization
    status: EmailStatus = Field(
        default=EmailStatus.UNREAD,
        index=True,
        description="Email processing status"
    )
    
    priority: EmailPriority = Field(
        default=EmailPriority.NORMAL,
        index=True,
        description="Email priority level"
    )
    
    direction: EmailDirection = Field(
        description="Email direction (inbound/outbound)"
    )
    
    category: EmailCategory = Field(
        default=EmailCategory.OTHER,
        index=True,
        description="Email category"
    )
    
    # Gmail labels
    labels: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Gmail labels applied to the email"
    )
    
    # Processing information
    processed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when email was processed"
    )
    
    processed_by: Optional[UUID] = Field(
        default=None,
        description="Agent that processed the email"
    )
    
    processing_duration: Optional[float] = Field(
        default=None,
        description="Time taken to process the email (seconds)"
    )
    
    # AI analysis results
    sentiment_score: Optional[float] = Field(
        default=None,
        description="Sentiment analysis score (-1 to 1)"
    )
    
    urgency_score: Optional[float] = Field(
        default=None,
        description="Urgency score (0 to 1)"
    )
    
    importance_score: Optional[float] = Field(
        default=None,
        description="Importance score (0 to 1)"
    )
    
    # Extracted entities and keywords
    entities: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Extracted entities from email content"
    )
    
    keywords: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Extracted keywords"
    )
    
    # Actions and responses
    actions_taken: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of actions taken on this email"
    )
    
    response_generated: bool = Field(
        default=False,
        description="Whether a response was generated"
    )
    
    response_sent: bool = Field(
        default=False,
        description="Whether a response was sent"
    )
    
    # Additional metadata
    size_bytes: Optional[int] = Field(
        default=None,
        description="Email size in bytes"
    )
    
    attachment_count: int = Field(
        default=0,
        description="Number of attachments"
    )
    
    is_encrypted: bool = Field(
        default=False,
        description="Whether the email is encrypted"
    )
    
    is_signed: bool = Field(
        default=False,
        description="Whether the email is digitally signed"
    )
    
    # Relationships
    thread: Optional["EmailThread"] = Relationship(
        back_populates="emails"
    )
    
    attachments: List["EmailAttachment"] = Relationship(
        back_populates="email",
        cascade_delete=True
    )
    
    analytics: List["EmailAnalytics"] = Relationship(
        back_populates="email",
        cascade_delete=True
    )
    
    @validator('sender_email')
    def validate_sender_email(cls, v):
        """Validate sender email format."""
        return v.lower()
    
    @validator('recipients_to', 'recipients_cc', 'recipients_bcc')
    def validate_recipients(cls, v):
        """Validate recipient email formats."""
        return [email.lower() for email in v]
    
    def is_unread(self) -> bool:
        """Check if email is unread."""
        return self.status == EmailStatus.UNREAD
    
    def is_high_priority(self) -> bool:
        """Check if email is high priority."""
        return self.priority in [EmailPriority.HIGH, EmailPriority.URGENT]
    
    def is_urgent(self) -> bool:
        """Check if email is urgent."""
        return self.priority == EmailPriority.URGENT
    
    def mark_as_read(self) -> None:
        """Mark email as read."""
        self.status = EmailStatus.READ
    
    def mark_as_processed(self, processed_by: UUID, duration: float) -> None:
        """Mark email as processed."""
        self.status = EmailStatus.PROCESSED
        self.processed_at = datetime.utcnow()
        self.processed_by = processed_by
        self.processing_duration = duration
    
    def add_action(self, action: str) -> None:
        """Add an action to the email."""
        if action not in self.actions_taken:
            self.actions_taken.append(action)
    
    def get_all_recipients(self) -> List[str]:
        """Get all recipients (TO, CC, BCC)."""
        return self.recipients_to + self.recipients_cc + self.recipients_bcc


class EmailThread(BaseModel, SoftDeleteMixin, table=True):
    """
    Email thread model.
    
    Represents a conversation thread containing multiple related emails.
    """
    
    __tablename__ = "email_threads"
    
    # Thread identification
    gmail_thread_id: Optional[str] = Field(
        default=None,
        unique=True,
        index=True,
        description="Gmail thread ID"
    )
    
    # Thread metadata
    subject: str = Field(
        max_length=1000,
        description="Thread subject"
    )
    
    participants: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="List of participant email addresses"
    )
    
    # Thread timing
    first_message_at: datetime = Field(
        description="Timestamp of first message in thread"
    )
    
    last_message_at: datetime = Field(
        description="Timestamp of last message in thread"
    )
    
    # Thread status
    is_active: bool = Field(
        default=True,
        description="Whether thread is active"
    )
    
    message_count: int = Field(
        default=0,
        description="Number of messages in thread"
    )
    
    # Thread categorization
    category: EmailCategory = Field(
        default=EmailCategory.OTHER,
        description="Thread category"
    )
    
    priority: EmailPriority = Field(
        default=EmailPriority.NORMAL,
        description="Thread priority"
    )
    
    # Thread analysis
    sentiment_trend: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Sentiment trend over time"
    )
    
    urgency_trend: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(JSONB),
        description="Urgency trend over time"
    )
    
    # Thread summary
    summary: Optional[str] = Field(
        default=None,
        description="AI-generated thread summary"
    )
    
    key_points: List[str] = Field(
        default=[],
        sa_column=Column(ARRAY(String)),
        description="Key points from the thread"
    )
    
    # Relationships
    emails: List[Email] = Relationship(
        back_populates="thread",
        cascade_delete=True
    )
    
    def add_message(self, email: Email) -> None:
        """Add a message to the thread."""
        self.message_count += 1
        self.last_message_at = email.sent_at
        
        # Update participants
        if email.sender_email not in self.participants:
            self.participants.append(email.sender_email)
        
        for recipient in email.get_all_recipients():
            if recipient not in self.participants:
                self.participants.append(recipient)
    
    def get_latest_message(self) -> Optional[Email]:
        """Get the latest message in the thread."""
        if not self.emails:
            return None
        return max(self.emails, key=lambda e: e.sent_at)
    
    def get_participant_count(self) -> int:
        """Get number of participants in thread."""
        return len(self.participants)


class EmailAttachment(BaseModel, SoftDeleteMixin, table=True):
    """
    Email attachment model.
    
    Represents file attachments in emails with metadata and security information.
    """
    
    __tablename__ = "email_attachments"
    
    # Foreign key to email
    email_id: UUID = Field(
        foreign_key="emails.id",
        description="Reference to the email"
    )
    
    # Attachment metadata
    filename: str = Field(
        max_length=255,
        description="Original filename"
    )
    
    content_type: str = Field(
        max_length=100,
        description="MIME content type"
    )
    
    size_bytes: int = Field(
        description="Attachment size in bytes"
    )
    
    # Gmail API fields
    gmail_attachment_id: Optional[str] = Field(
        default=None,
        description="Gmail attachment ID"
    )
    
    # File storage
    file_path: Optional[str] = Field(
        default=None,
        description="Path to stored file"
    )
    
    file_hash: Optional[str] = Field(
        default=None,
        description="File hash for integrity checking"
    )
    
    # Security scanning
    is_scanned: bool = Field(
        default=False,
        description="Whether file has been scanned for malware"
    )
    
    scan_result: Optional[str] = Field(
        default=None,
        description="Malware scan result"
    )
    
    is_safe: bool = Field(
        default=False,
        description="Whether file is considered safe"
    )
    
    # Processing status
    is_extracted: bool = Field(
        default=False,
        description="Whether attachment content has been extracted"
    )
    
    extracted_text: Optional[str] = Field(
        default=None,
        description="Extracted text content"
    )
    
    # Relationships
    email: Email = Relationship(back_populates="attachments")
    
    def is_image(self) -> bool:
        """Check if attachment is an image."""
        return self.content_type.startswith("image/")
    
    def is_document(self) -> bool:
        """Check if attachment is a document."""
        document_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/csv",
        ]
        return self.content_type in document_types
    
    def is_executable(self) -> bool:
        """Check if attachment is executable."""
        executable_types = [
            "application/x-executable",
            "application/x-msdos-program",
            "application/x-msdownload",
        ]
        return self.content_type in executable_types
    
    def get_file_extension(self) -> str:
        """Get file extension from filename."""
        return self.filename.split(".")[-1].lower() if "." in self.filename else ""


class EmailAnalytics(BaseModel, table=True):
    """
    Email analytics model.
    
    Stores analytics data for email processing and agent performance.
    """
    
    __tablename__ = "email_analytics"
    
    # Foreign key to email
    email_id: UUID = Field(
        foreign_key="emails.id",
        description="Reference to the email"
    )
    
    # Analytics metadata
    metric_name: str = Field(
        max_length=100,
        description="Name of the metric"
    )
    
    metric_value: float = Field(
        description="Numeric value of the metric"
    )
    
    metric_unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement"
    )
    
    # Analysis context
    analysis_type: str = Field(
        max_length=50,
        description="Type of analysis performed"
    )
    
    analysis_version: str = Field(
        max_length=20,
        description="Version of analysis algorithm"
    )
    
    # Additional data
    details: Dict[str, Any] = Field(
        default={},
        sa_column=Column(JSONB),
        description="Additional analysis details"
    )
    
    # Relationships
    email: Email = Relationship(back_populates="analytics")
    
    class Config:
        """Model configuration."""
        # Ensure unique metric per email
        indexes = [
            Index("idx_email_analytics_email_metric", "email_id", "metric_name", unique=True)
        ]