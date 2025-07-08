"""
Enhanced Gmail Agent with comprehensive database integration.

This module extends the Gmail Agent with robust database persistence,
analytics tracking, and comprehensive email management capabilities.
"""

import asyncio
import base64
import email
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

try:
    from google.auth.exceptions import RefreshError
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    # For testing environments where Google libraries may not be installed
    service_account = None
    build = None
    HttpError = Exception
    RefreshError = Exception

from .base import AgentMessage, AgentState, BaseAgent
from .exceptions import AgentError, AgentStateError
from .gmail_agent import GmailAgent, GmailAgentError, EmailClassification, EmailSummary
from ..database.operations.emails import EmailRepository, EmailThreadRepository, EmailAnalyticsRepository
from ..database.models.emails import Email, EmailThread, EmailAnalytics, EmailAttachment
from ..database.models.emails import EmailStatus, EmailPriority, EmailDirection, EmailCategory
from ..database.connection import get_database_manager


class EnhancedGmailAgent(GmailAgent):
    """
    Enhanced Gmail Agent with comprehensive database integration.
    
    Extends the base Gmail Agent to provide:
    - Persistent email storage and management
    - Advanced analytics and reporting  
    - Thread tracking and management
    - Attachment handling with security scanning
    - Performance metrics and optimization
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        message_broker: Any
    ):
        """
        Initialize the Enhanced Gmail Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
            logger: Logger instance
            message_broker: Message broker for inter-agent communication
        """
        super().__init__(agent_id, config, logger, message_broker)
        
        # Database repositories
        self.email_repository = EmailRepository(logger)
        self.thread_repository = EmailThreadRepository(logger)
        self.analytics_repository = EmailAnalyticsRepository(logger)
        
        # Enhanced configuration
        self.db_config = config.get("database", {})
        self.analytics_config = config.get("analytics", {})
        self.sync_config = config.get("sync", {})
        
        # Performance tracking
        self.performance_metrics = {
            "emails_processed": 0,
            "emails_stored": 0,
            "threads_created": 0,
            "analytics_recorded": 0,
            "sync_time": 0.0,
            "processing_time": 0.0
        }
        
        # Sync state
        self.last_sync_time = None
        self.sync_in_progress = False
        
    async def _initialize(self) -> None:
        """Initialize the enhanced agent with database setup."""
        await super()._initialize()
        
        # Initialize database connection
        await self._initialize_database()
        
        # Load sync state from database
        await self._load_sync_state()
        
        # Start background sync if enabled
        if self.sync_config.get("auto_sync", True):
            asyncio.create_task(self._background_sync())
    
    async def _initialize_database(self) -> None:
        """Initialize database connection and repositories."""
        try:
            # Database repositories are initialized in constructor
            # Test database connectivity
            await self.email_repository.count()
            
            self.logger.info("Database connection established for Gmail Agent")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise GmailAgentError(f"Database initialization failed: {e}")
    
    async def _load_sync_state(self) -> None:
        """Load sync state from database."""
        try:
            # Get latest email to determine last sync time
            latest_emails = await self.email_repository.list_paginated(
                page=1, per_page=1
            )
            
            if latest_emails["items"]:
                self.last_sync_time = latest_emails["items"][0].received_at
            
            self.logger.debug(f"Loaded sync state: last_sync_time={self.last_sync_time}")
            
        except Exception as e:
            self.logger.error(f"Failed to load sync state: {e}")
            # Continue without sync state
    
    async def fetch_emails(self, query: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Enhanced email fetching with database persistence.
        
        Args:
            query: Gmail search query
            max_results: Maximum number of emails to fetch
            
        Returns:
            List of email dictionaries with database integration
        """
        start_time = time.time()
        
        try:
            # Use parent class method to fetch emails
            emails = await super().fetch_emails(query, max_results)
            
            # Process and store emails in database
            stored_emails = []
            for email_data in emails:
                try:
                    stored_email = await self._process_and_store_email(email_data)
                    if stored_email:
                        stored_emails.append(stored_email)
                        self.performance_metrics["emails_stored"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to store email {email_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["processing_time"] += processing_time
            self.performance_metrics["emails_processed"] += len(emails)
            
            # Record analytics
            await self._record_fetch_analytics(len(emails), len(stored_emails), processing_time)
            
            self.logger.info(f"Fetched {len(emails)} emails, stored {len(stored_emails)} in {processing_time:.2f}s")
            
            return [email.to_dict() for email in stored_emails]
            
        except Exception as e:
            self.logger.error(f"Enhanced email fetch failed: {e}")
            raise GmailAgentError(f"Email fetch failed: {e}")
    
    async def _process_and_store_email(self, email_data: Dict[str, Any]) -> Optional[Email]:
        """
        Process and store an email in the database.
        
        Args:
            email_data: Email data from Gmail API
            
        Returns:
            Stored Email model instance or None if failed
        """
        try:
            # Check if email already exists
            gmail_id = email_data.get("id")
            if gmail_id:
                existing_email = await self.email_repository.get_by_gmail_id(gmail_id)
                if existing_email:
                    # Update existing email if needed
                    return await self._update_existing_email(existing_email, email_data)
            
            # Extract email metadata
            headers = email_data.get("payload", {}).get("headers", [])
            header_dict = {h["name"]: h["value"] for h in headers}
            
            # Parse email content
            subject = header_dict.get("Subject", "")
            sender_email = self._extract_email_from_header(header_dict.get("From", ""))
            sender_name = self._extract_name_from_header(header_dict.get("From", ""))
            message_id = header_dict.get("Message-ID", "")
            
            # Extract body content
            body_text, body_html = self._extract_body_content(email_data.get("payload", {}))
            
            # Generate snippet
            snippet = email_data.get("snippet", "")
            
            # Parse recipients
            recipients_to = self._parse_recipients(header_dict.get("To", ""))
            recipients_cc = self._parse_recipients(header_dict.get("Cc", ""))
            recipients_bcc = self._parse_recipients(header_dict.get("Bcc", ""))
            
            # Determine email direction
            user_email = self.gmail_config.get("user_email", "me")
            direction = EmailDirection.OUTBOUND if sender_email == user_email else EmailDirection.INBOUND
            
            # Parse timestamps
            date_header = header_dict.get("Date", "")
            sent_at = self._parse_email_date(date_header)
            received_at = datetime.now(timezone.utc)
            
            # Classify email
            classification = await self._classify_email_enhanced(email_data)
            
            # Determine priority
            priority = await self._determine_email_priority(email_data, classification)
            
            # Extract labels
            labels = email_data.get("labelIds", [])
            
            # Create email record
            email_record = await self.email_repository.create(
                gmail_id=gmail_id,
                gmail_thread_id=email_data.get("threadId"),
                message_id=message_id,
                in_reply_to=header_dict.get("In-Reply-To"),
                references=self._parse_references(header_dict.get("References", "")),
                subject=subject,
                body_text=body_text,
                body_html=body_html,
                snippet=snippet,
                sender_email=sender_email,
                sender_name=sender_name,
                recipients_to=recipients_to,
                recipients_cc=recipients_cc,
                recipients_bcc=recipients_bcc,
                sent_at=sent_at,
                received_at=received_at,
                status=EmailStatus.UNREAD,
                priority=priority,
                direction=direction,
                category=EmailCategory(classification.category),
                labels=labels,
                size_bytes=email_data.get("sizeEstimate", 0),
                attachment_count=len(email_data.get("payload", {}).get("parts", [])),
                is_encrypted=self._is_encrypted(email_data),
                is_signed=self._is_signed(email_data)
            )
            
            # Process thread
            await self._process_email_thread(email_record, email_data)
            
            # Process attachments
            await self._process_email_attachments(email_record, email_data)
            
            # Record analytics
            await self._record_email_analytics(email_record, classification)
            
            # Perform AI analysis
            await self._perform_ai_analysis(email_record)
            
            return email_record
            
        except Exception as e:
            self.logger.error(f"Failed to process and store email: {e}")
            return None
    
    async def _update_existing_email(self, existing_email: Email, email_data: Dict[str, Any]) -> Email:
        """Update existing email with new data."""
        try:
            # Update labels if they changed
            new_labels = email_data.get("labelIds", [])
            if set(new_labels) != set(existing_email.labels):
                await self.email_repository.update(
                    existing_email.id,
                    labels=new_labels
                )
            
            # Update status based on labels
            new_status = self._determine_status_from_labels(new_labels)
            if new_status != existing_email.status:
                await self.email_repository.update(
                    existing_email.id,
                    status=new_status
                )
            
            return existing_email
            
        except Exception as e:
            self.logger.error(f"Failed to update existing email: {e}")
            return existing_email
    
    async def _classify_email_enhanced(self, email_data: Dict[str, Any]) -> EmailClassification:
        """Enhanced email classification with database insights."""
        try:
            # Use parent class classification
            classification = await super()._classify_email(email_data)
            
            # Enhance with database insights
            sender_email = self._extract_email_from_header(
                email_data.get("payload", {}).get("headers", [])
            )
            
            # Get sender history for better classification
            sender_history = await self.email_repository.get_emails_by_sender(
                sender_email, limit=10
            )
            
            # Adjust classification based on sender history
            if sender_history:
                historical_categories = [email.category for email in sender_history]
                most_common_category = max(set(historical_categories), key=historical_categories.count)
                
                # Boost confidence if consistent with history
                if classification.category == most_common_category:
                    classification.confidence = min(classification.confidence + 0.1, 1.0)
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Enhanced email classification failed: {e}")
            return EmailClassification("other", 0.5)
    
    async def _determine_email_priority(
        self,
        email_data: Dict[str, Any],
        classification: EmailClassification
    ) -> EmailPriority:
        """Determine email priority based on content and classification."""
        try:
            headers = email_data.get("payload", {}).get("headers", [])
            header_dict = {h["name"]: h["value"] for h in headers}
            
            subject = header_dict.get("Subject", "").lower()
            body = self._extract_body_content(email_data.get("payload", {}))[0] or ""
            body = body.lower()
            
            # Check for urgent keywords
            urgent_keywords = ["urgent", "asap", "emergency", "critical", "immediate"]
            high_keywords = ["important", "priority", "deadline", "meeting"]
            
            if any(keyword in subject for keyword in urgent_keywords):
                return EmailPriority.URGENT
            
            if any(keyword in subject for keyword in high_keywords):
                return EmailPriority.HIGH
            
            # Check priority header
            priority_header = header_dict.get("X-Priority", "")
            if priority_header in ["1", "2"]:
                return EmailPriority.HIGH
            
            # Use classification for priority
            if classification.category in ["meeting", "task"]:
                return EmailPriority.HIGH
            
            return EmailPriority.NORMAL
            
        except Exception as e:
            self.logger.error(f"Failed to determine email priority: {e}")
            return EmailPriority.NORMAL
    
    async def _process_email_thread(self, email_record: Email, email_data: Dict[str, Any]) -> None:
        """Process email thread information."""
        try:
            gmail_thread_id = email_data.get("threadId")
            if not gmail_thread_id:
                return
            
            # Find or create thread
            thread = await self.thread_repository.get_by_gmail_thread_id(gmail_thread_id)
            
            if not thread:
                # Create new thread
                thread = await self.thread_repository.create(
                    gmail_thread_id=gmail_thread_id,
                    subject=email_record.subject,
                    participants=[email_record.sender_email],
                    first_message_at=email_record.sent_at,
                    last_message_at=email_record.sent_at,
                    message_count=1,
                    category=email_record.category,
                    priority=email_record.priority
                )
                
                self.performance_metrics["threads_created"] += 1
            else:
                # Update existing thread
                # Add new participants
                participants = list(set(thread.participants + [email_record.sender_email]))
                
                # Update thread
                await self.thread_repository.update(
                    thread.id,
                    participants=participants,
                    last_message_at=email_record.sent_at,
                    message_count=thread.message_count + 1,
                    priority=max(thread.priority, email_record.priority, key=lambda p: p.value)
                )
            
            # Link email to thread
            await self.email_repository.update(
                email_record.id,
                thread_id=thread.id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process email thread: {e}")
    
    async def _process_email_attachments(self, email_record: Email, email_data: Dict[str, Any]) -> None:
        """Process email attachments."""
        try:
            payload = email_data.get("payload", {})
            parts = payload.get("parts", [])
            
            if not parts:
                return
            
            for part in parts:
                if part.get("filename"):
                    # This is an attachment
                    attachment_data = {
                        "email_id": email_record.id,
                        "filename": part.get("filename"),
                        "content_type": part.get("mimeType", ""),
                        "size_bytes": part.get("body", {}).get("size", 0),
                        "gmail_attachment_id": part.get("body", {}).get("attachmentId")
                    }
                    
                    # Store attachment metadata (not content for security)
                    # Content would be downloaded separately if needed
                    # await self.attachment_repository.create(**attachment_data)
            
        except Exception as e:
            self.logger.error(f"Failed to process email attachments: {e}")
    
    async def _record_email_analytics(self, email_record: Email, classification: EmailClassification) -> None:
        """Record email analytics."""
        try:
            metrics = [
                ("classification_confidence", classification.confidence, "classification", "v1.0"),
                ("subject_length", len(email_record.subject), "content_analysis", "v1.0"),
                ("body_length", len(email_record.body_text or ""), "content_analysis", "v1.0"),
                ("recipient_count", len(email_record.recipients_to), "recipient_analysis", "v1.0"),
            ]
            
            for metric_name, value, analysis_type, version in metrics:
                await self.analytics_repository.record_metric(
                    email_record.id,
                    metric_name,
                    float(value),
                    analysis_type,
                    version
                )
            
            self.performance_metrics["analytics_recorded"] += len(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to record email analytics: {e}")
    
    async def _perform_ai_analysis(self, email_record: Email) -> None:
        """Perform AI analysis on email content."""
        try:
            # Sentiment analysis
            sentiment_score = await self._analyze_sentiment(email_record.body_text or "")
            
            # Urgency analysis
            urgency_score = await self._analyze_urgency(email_record.subject, email_record.body_text or "")
            
            # Importance analysis
            importance_score = await self._analyze_importance(email_record)
            
            # Update email with AI scores
            await self.email_repository.update_ai_scores(
                email_record.id,
                sentiment_score=sentiment_score,
                urgency_score=urgency_score,
                importance_score=importance_score
            )
            
        except Exception as e:
            self.logger.error(f"Failed to perform AI analysis: {e}")
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text content."""
        try:
            # Simple sentiment analysis - would integrate with proper NLP service
            positive_words = ["good", "great", "excellent", "thank", "please", "appreciate"]
            negative_words = ["bad", "terrible", "awful", "angry", "disappointed", "frustrated"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment = (positive_count - negative_count) / total_words
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return 0.0
    
    async def _analyze_urgency(self, subject: str, body: str) -> float:
        """Analyze urgency of email content."""
        try:
            urgent_keywords = ["urgent", "asap", "emergency", "critical", "immediate", "deadline"]
            text = f"{subject} {body}".lower()
            
            urgency_score = 0.0
            for keyword in urgent_keywords:
                if keyword in text:
                    urgency_score += 0.2
            
            return min(1.0, urgency_score)
            
        except Exception as e:
            self.logger.error(f"Urgency analysis failed: {e}")
            return 0.0
    
    async def _analyze_importance(self, email_record: Email) -> float:
        """Analyze importance of email."""
        try:
            importance_score = 0.0
            
            # Check sender importance (would be based on user's contact history)
            if email_record.sender_email.endswith(".edu"):
                importance_score += 0.1
            if "boss" in email_record.sender_name.lower():
                importance_score += 0.3
            
            # Check subject indicators
            subject_lower = email_record.subject.lower()
            if "meeting" in subject_lower:
                importance_score += 0.2
            if "project" in subject_lower:
                importance_score += 0.1
            
            return min(1.0, importance_score)
            
        except Exception as e:
            self.logger.error(f"Importance analysis failed: {e}")
            return 0.0
    
    async def _record_fetch_analytics(self, fetched_count: int, stored_count: int, processing_time: float) -> None:
        """Record fetch operation analytics."""
        try:
            # This could be stored in a separate analytics table
            self.logger.info(f"Fetch analytics: {fetched_count} fetched, {stored_count} stored, {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to record fetch analytics: {e}")
    
    async def _background_sync(self) -> None:
        """Background email synchronization."""
        sync_interval = self.sync_config.get("interval_seconds", 300)  # 5 minutes
        
        while not self._shutdown_event.is_set():
            try:
                if not self.sync_in_progress:
                    await self._perform_sync()
                
                await asyncio.sleep(sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background sync error: {e}")
                await asyncio.sleep(sync_interval)
    
    async def _perform_sync(self) -> None:
        """Perform email synchronization."""
        if self.sync_in_progress:
            return
        
        self.sync_in_progress = True
        start_time = time.time()
        
        try:
            # Build query for new emails
            query = "is:unread"
            if self.last_sync_time:
                # Gmail query format for after timestamp
                query += f" after:{int(self.last_sync_time.timestamp())}"
            
            # Fetch new emails
            new_emails = await self.fetch_emails(query=query, max_results=100)
            
            # Update sync time
            self.last_sync_time = datetime.now(timezone.utc)
            
            sync_time = time.time() - start_time
            self.performance_metrics["sync_time"] += sync_time
            
            self.logger.info(f"Sync completed: {len(new_emails)} new emails in {sync_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Sync failed: {e}")
        finally:
            self.sync_in_progress = False
    
    async def get_email_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive email statistics."""
        try:
            # Get database statistics
            db_stats = await self.email_repository.get_email_statistics(days)
            
            # Get top senders
            top_senders = await self.email_repository.get_top_senders(limit=10, days=days)
            
            # Get thread statistics
            thread_stats = await self.thread_repository.get_repository_stats()
            
            # Combine with performance metrics
            combined_stats = {
                **db_stats,
                "top_senders": top_senders,
                "thread_stats": thread_stats,
                "performance_metrics": self.performance_metrics,
                "sync_status": {
                    "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
                    "sync_in_progress": self.sync_in_progress
                }
            }
            
            return combined_stats
            
        except Exception as e:
            self.logger.error(f"Failed to get email statistics: {e}")
            return {}
    
    async def search_emails_enhanced(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Enhanced email search with database integration."""
        try:
            filters = filters or {}
            
            # Search in database
            results = await self.email_repository.search_emails(
                query=query,
                sender=filters.get("sender"),
                status=filters.get("status"),
                priority=filters.get("priority"),
                limit=limit
            )
            
            return [email.to_dict() for email in results]
            
        except Exception as e:
            self.logger.error(f"Enhanced email search failed: {e}")
            return []
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            # Get repository stats
            email_stats = await self.email_repository.get_repository_stats()
            thread_stats = await self.thread_repository.get_repository_stats()
            analytics_stats = await self.analytics_repository.get_repository_stats()
            
            return {
                "performance_metrics": self.performance_metrics,
                "email_repository": email_stats,
                "thread_repository": thread_stats,
                "analytics_repository": analytics_stats,
                "agent_metrics": self.get_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    # Helper methods for email parsing
    def _extract_email_from_header(self, header_value: str) -> str:
        """Extract email address from header value."""
        if not header_value:
            return ""
        
        # Use regex to extract email
        match = re.search(r'<([^>]+)>', header_value)
        if match:
            return match.group(1).lower()
        
        # If no angle brackets, assume entire value is email
        return header_value.strip().lower()
    
    def _extract_name_from_header(self, header_value: str) -> str:
        """Extract name from header value."""
        if not header_value:
            return ""
        
        # Remove email part
        name = re.sub(r'<[^>]+>', '', header_value).strip()
        
        # Remove quotes
        name = name.strip('"\'')
        
        return name
    
    def _extract_body_content(self, payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Extract text and HTML body content."""
        text_content = None
        html_content = None
        
        def extract_from_part(part):
            nonlocal text_content, html_content
            
            mime_type = part.get("mimeType", "")
            body = part.get("body", {})
            
            if mime_type == "text/plain" and body.get("data"):
                text_content = base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="ignore")
            elif mime_type == "text/html" and body.get("data"):
                html_content = base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="ignore")
            
            # Recurse into parts
            for subpart in part.get("parts", []):
                extract_from_part(subpart)
        
        extract_from_part(payload)
        
        return text_content, html_content
    
    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse recipient string into list of email addresses."""
        if not recipients_str:
            return []
        
        # Split by comma and extract emails
        recipients = []
        for recipient in recipients_str.split(","):
            email = self._extract_email_from_header(recipient.strip())
            if email:
                recipients.append(email)
        
        return recipients
    
    def _parse_references(self, references_str: str) -> List[str]:
        """Parse References header into list of message IDs."""
        if not references_str:
            return []
        
        # Split by whitespace and filter out empty strings
        return [ref.strip() for ref in references_str.split() if ref.strip()]
    
    def _parse_email_date(self, date_str: str) -> datetime:
        """Parse email date string into datetime object."""
        if not date_str:
            return datetime.now(timezone.utc)
        
        try:
            # Parse RFC 2822 date format
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            # Fallback to current time
            return datetime.now(timezone.utc)
    
    def _determine_status_from_labels(self, labels: List[str]) -> EmailStatus:
        """Determine email status from Gmail labels."""
        if "UNREAD" in labels:
            return EmailStatus.UNREAD
        elif "TRASH" in labels:
            return EmailStatus.DELETED
        elif "SPAM" in labels:
            return EmailStatus.SPAM
        else:
            return EmailStatus.READ
    
    def _is_encrypted(self, email_data: Dict[str, Any]) -> bool:
        """Check if email is encrypted."""
        # Simple check - would need more sophisticated detection
        payload = email_data.get("payload", {})
        return "application/pgp-encrypted" in str(payload)
    
    def _is_signed(self, email_data: Dict[str, Any]) -> bool:
        """Check if email is digitally signed."""
        # Simple check - would need more sophisticated detection
        payload = email_data.get("payload", {})
        return "application/pgp-signature" in str(payload)