"""
Gmail Agent implementation for the autonomous agent system.

This module provides comprehensive Gmail integration including email fetching,
classification, automated responses, archiving, and organization following
TDD principles and the established architecture patterns.
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


class GmailAgentError(AgentError):
    """Gmail Agent specific error."""
    pass


class GmailAuthenticationError(GmailAgentError):
    """Gmail authentication error."""
    pass


class GmailQuotaError(GmailAgentError):
    """Gmail quota exceeded error."""
    pass


class GmailRateLimitError(GmailAgentError):
    """Gmail rate limit error."""
    pass


class EmailClassification:
    """Email classification result."""
    
    def __init__(self, category: str, confidence: float, keywords: List[str] = None):
        self.category = category
        self.confidence = confidence
        self.keywords = keywords or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "confidence": self.confidence,
            "keywords": self.keywords
        }


class EmailSummary:
    """Email summary data structure."""
    
    def __init__(self):
        self.total_emails = 0
        self.unread_emails = 0
        self.important_emails = 0
        self.spam_emails = 0
        self.categories = defaultdict(int)
        self.senders = defaultdict(int)
        self.time_range = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_emails": self.total_emails,
            "unread_emails": self.unread_emails,
            "important_emails": self.important_emails,
            "spam_emails": self.spam_emails,
            "categories": dict(self.categories),
            "top_senders": dict(sorted(self.senders.items(), key=lambda x: x[1], reverse=True)[:10]),
            "time_range": self.time_range
        }


class GmailRateLimiter:
    """Gmail API rate limiter."""
    
    def __init__(self, requests_per_minute: int = 250):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        async with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            cutoff_time = current_time - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            # Check if we need to wait
            if len(self.request_times) >= self.requests_per_minute:
                # Wait until the oldest request is more than 1 minute old
                wait_time = 60 - (current_time - self.request_times[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Re-clean the list after waiting
                    current_time = time.time()
                    cutoff_time = current_time - 60
                    self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            # Record this request
            self.request_times.append(current_time)


class GmailAgent(BaseAgent):
    """
    Gmail Agent for email processing and management.
    
    Provides comprehensive Gmail integration including:
    - Email fetching and parsing
    - Intelligent classification
    - Automated responses
    - Email archiving and organization
    - Attachment processing
    - Thread management
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        message_broker: Any,
        service_manager: Optional[Any] = None
    ):
        """
        Initialize Gmail Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
            logger: Logger instance
            message_broker: Message broker for communication
            service_manager: AI service manager for enhanced capabilities
        """
        super().__init__(agent_id, config, logger, message_broker)
        
        # Gmail-specific configuration
        self.gmail_config = config.get("gmail", {})
        self.classification_config = config.get("classification", {})
        self.auto_response_config = config.get("auto_response", {})
        self.archiving_config = config.get("archiving", {})
        
        # Gmail API objects
        self.gmail_service = None
        self.gmail_credentials = None
        
        # Rate limiting
        self.rate_limiter = GmailRateLimiter(
            self.gmail_config.get("rate_limit_per_minute", 250)
        )
        
        # Classification cache
        self.classification_cache = {}
        
        # Auto-response tracking
        self.auto_responses_sent_today = 0
        self.auto_response_reset_time = None
        self.out_of_office_mode = False
        
        # AI service manager for enhanced capabilities
        self.service_manager = service_manager
        self.ai_enabled = service_manager is not None and getattr(service_manager, 'is_initialized', False)
        
        # AI-enhanced features configuration
        self.ai_classification_enabled = config.get("ai_classification_enabled", True)
        self.ai_response_generation_enabled = config.get("ai_response_generation_enabled", False)
        self.ai_summarization_enabled = config.get("ai_summarization_enabled", True)
        self.ai_sentiment_analysis_enabled = config.get("ai_sentiment_analysis_enabled", True)
        
        # Email categories for AI classification
        self.email_categories = config.get("email_categories", [
            "urgent", "work", "personal", "newsletter", "promotion", "spam", "social"
        ])
        
        # Gmail-specific metrics
        self.metrics.update({
            "emails_processed": 0,
            "classifications_made": 0,
            "auto_responses_sent": 0,
            "emails_archived": 0,
            "attachments_processed": 0,
            "api_calls_made": 0,
            "rate_limit_hits": 0,
            "ai_classifications": 0,
            "ai_responses_generated": 0,
            "ai_summaries_created": 0,
        })
    
    async def _initialize(self) -> None:
        """Initialize Gmail Agent specific resources."""
        self.logger.info(f"Initializing Gmail Agent {self.agent_id}")
        
        try:
            # Authenticate with Gmail API
            await self._authenticate()
            
            # Initialize auto-response tracking
            self._reset_auto_response_tracking()
            
            # Set up periodic tasks
            asyncio.create_task(self._periodic_email_check())
            asyncio.create_task(self._periodic_auto_response_reset())
            
            self.logger.info(f"Gmail Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gmail Agent {self.agent_id}: {e}")
            raise GmailAgentError(f"Failed to initialize Gmail Agent: {e}") from e
    
    async def _cleanup(self) -> None:
        """Cleanup Gmail Agent specific resources."""
        self.logger.info(f"Cleaning up Gmail Agent {self.agent_id}")
        
        # Close Gmail service connection
        self.gmail_service = None
        self.gmail_credentials = None
        
        # Clear caches
        self.classification_cache.clear()
        
        self.logger.info(f"Gmail Agent {self.agent_id} cleaned up successfully")
    
    async def _authenticate(self) -> None:
        """Authenticate with Gmail API."""
        try:
            if not service_account or not build:
                raise GmailAuthenticationError("Google API libraries not available")
            
            credentials_path = self.gmail_config.get("credentials_path")
            scopes = self.gmail_config.get("scopes", [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.modify"
            ])
            
            if not credentials_path:
                raise GmailAuthenticationError("Gmail credentials path not configured")
            
            # Load service account credentials
            self.gmail_credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
            
            # Build Gmail service
            self.gmail_service = build('gmail', 'v1', credentials=self.gmail_credentials)
            
            # Test the connection
            await self._check_gmail_connection()
            
            self.logger.info("Gmail API authentication successful")
            
        except Exception as e:
            self.logger.error(f"Gmail authentication failed: {e}")
            raise GmailAuthenticationError(f"Gmail authentication failed: {e}") from e
    
    async def _refresh_credentials(self) -> None:
        """Refresh Gmail API credentials if needed."""
        try:
            if self.gmail_credentials and self.gmail_credentials.expired:
                self.gmail_credentials.refresh()
                self.logger.info("Gmail credentials refreshed")
        except RefreshError as e:
            self.logger.error(f"Failed to refresh Gmail credentials: {e}")
            raise GmailAuthenticationError(f"Failed to refresh credentials: {e}") from e
    
    async def _check_gmail_connection(self) -> bool:
        """Check Gmail API connection health."""
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Try to get user profile
            user_id = self.gmail_config.get("user_email", "me")
            profile = self.gmail_service.users().getProfile(userId=user_id).execute()
            
            self.metrics["api_calls_made"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Gmail connection check failed: {e}")
            return False
    
    async def _check_rate_limit(self) -> None:
        """Check and handle rate limiting."""
        await self.rate_limiter.wait_if_needed()
    
    async def _health_check(self) -> bool:
        """Perform Gmail Agent health check."""
        try:
            # Check Gmail API connection
            if not await self._check_gmail_connection():
                return False
            
            # Check credentials validity
            if self.gmail_credentials and self.gmail_credentials.expired:
                await self._refresh_credentials()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Gmail Agent health check failed: {e}")
            return False
    
    async def _fetch_emails(
        self,
        max_results: int = 100,
        query: Optional[str] = None,
        label_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch emails from Gmail.
        
        Args:
            max_results: Maximum number of emails to fetch
            query: Gmail search query
            label_ids: List of label IDs to filter by
            
        Returns:
            List of email dictionaries
        """
        try:
            await self._check_rate_limit()
            
            user_id = self.gmail_config.get("user_email", "me")
            
            # Build request parameters
            request_params = {
                "userId": user_id,
                "maxResults": max_results
            }
            
            if query:
                request_params["q"] = query
            
            if label_ids:
                request_params["labelIds"] = label_ids
            
            # Get message list
            messages_result = self.gmail_service.users().messages().list(**request_params).execute()
            messages = messages_result.get("messages", [])
            
            self.metrics["api_calls_made"] += 1
            
            # Fetch full message details
            emails = []
            for message in messages:
                await self._check_rate_limit()
                
                full_message = self.gmail_service.users().messages().get(
                    userId=user_id,
                    id=message["id"],
                    format="full"
                ).execute()
                
                self.metrics["api_calls_made"] += 1
                
                # Parse the message
                parsed_email = self._parse_email(full_message)
                emails.append(parsed_email)
            
            self.metrics["emails_processed"] += len(emails)
            self.logger.info(f"Fetched {len(emails)} emails")
            
            return emails
            
        except HttpError as e:
            if e.resp.status == 429:
                self.metrics["rate_limit_hits"] += 1
                raise GmailRateLimitError(f"Gmail API rate limit exceeded: {e}")
            elif e.resp.status == 403:
                raise GmailQuotaError(f"Gmail API quota exceeded: {e}")
            else:
                raise GmailAgentError(f"Gmail API error: {e}")
        except Exception as e:
            self.logger.error(f"Failed to fetch emails: {e}")
            raise GmailAgentError(f"Failed to fetch emails: {e}") from e
    
    def _parse_email(self, gmail_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Gmail message to extract useful information.
        
        Args:
            gmail_message: Raw Gmail message from API
            
        Returns:
            Parsed email dictionary
        """
        try:
            payload = gmail_message.get("payload", {})
            headers = payload.get("headers", [])
            
            # Extract headers
            header_dict = {}
            for header in headers:
                header_dict[header["name"]] = header["value"]
            
            # Extract body
            body = self._extract_email_body(payload)
            
            # Extract attachments
            attachments = self._extract_attachments(payload)
            
            # Parse date
            date_str = header_dict.get("Date", "")
            parsed_date = self._parse_email_date(date_str)
            
            parsed_email = {
                "id": gmail_message["id"],
                "thread_id": gmail_message["threadId"],
                "labels": gmail_message.get("labelIds", []),
                "snippet": gmail_message.get("snippet", ""),
                "from": header_dict.get("From", ""),
                "to": header_dict.get("To", ""),
                "cc": header_dict.get("Cc", ""),
                "bcc": header_dict.get("Bcc", ""),
                "subject": header_dict.get("Subject", ""),
                "date": parsed_date,
                "body": body,
                "attachments": attachments,
                "size_estimate": gmail_message.get("sizeEstimate", 0),
                "history_id": gmail_message.get("historyId", ""),
                "internal_date": gmail_message.get("internalDate", ""),
                "headers": header_dict,
            }
            
            return parsed_email
            
        except Exception as e:
            self.logger.error(f"Failed to parse email: {e}")
            return {
                "id": gmail_message.get("id", "unknown"),
                "error": f"Failed to parse: {e}"
            }
    
    def _extract_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract email body from payload."""
        try:
            # Try to get body from main payload
            if "body" in payload and payload["body"].get("data"):
                body_data = payload["body"]["data"]
                return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
            
            # Try to get body from parts
            if "parts" in payload:
                for part in payload["parts"]:
                    if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                        body_data = part["body"]["data"]
                        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                    
                    # Check nested parts
                    if "parts" in part:
                        nested_body = self._extract_email_body(part)
                        if nested_body:
                            return nested_body
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Failed to extract email body: {e}")
            return ""
    
    def _extract_attachments(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract attachment information from email payload."""
        attachments = []
        
        try:
            if "parts" in payload:
                for part in payload["parts"]:
                    if self._is_attachment(part):
                        attachment_info = {
                            "part_id": part.get("partId", ""),
                            "filename": part.get("filename", ""),
                            "mime_type": part.get("mimeType", ""),
                            "size": part.get("body", {}).get("size", 0),
                            "attachment_id": part.get("body", {}).get("attachmentId", ""),
                        }
                        attachments.append(attachment_info)
            
            self.metrics["attachments_processed"] += len(attachments)
            return attachments
            
        except Exception as e:
            self.logger.warning(f"Failed to extract attachments: {e}")
            return []
    
    def _is_attachment(self, part: Dict[str, Any]) -> bool:
        """Check if a message part is an attachment."""
        return (
            part.get("filename", "") != "" or
            part.get("body", {}).get("attachmentId") is not None
        )
    
    def _parse_email_date(self, date_str: str) -> Optional[datetime]:
        """Parse email date string to datetime object."""
        try:
            # Try different date formats
            import email.utils
            timestamp = email.utils.parsedate_to_datetime(date_str)
            return timestamp
        except Exception:
            try:
                # Fallback to basic parsing
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                return None
    
    async def _classify_email(self, email_data: Dict[str, Any]) -> EmailClassification:
        """
        Classify email into categories.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            EmailClassification object
        """
        try:
            if not self.classification_config.get("enabled", True):
                return EmailClassification("unknown", 0.0)
            
            # Check cache first
            email_id = email_data.get("id", "")
            if email_id in self.classification_cache:
                cached = self.classification_cache[email_id]
                return EmailClassification(cached["category"], cached["confidence"], cached["keywords"])
            
            # Extract features for classification
            subject = email_data.get("subject", "").lower()
            body = email_data.get("body", "").lower()
            sender = email_data.get("from", "").lower()
            
            # Combine text for analysis
            text_content = f"{subject} {body}"
            
            # Classification keywords
            keywords_config = self.classification_config.get("keywords", {})
            
            # Calculate category scores
            category_scores = {}
            
            for category, keywords in keywords_config.items():
                score = 0.0
                matched_keywords = []
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    # Count occurrences in subject (higher weight)
                    subject_matches = subject.count(keyword_lower)
                    body_matches = body.count(keyword_lower)
                    
                    if subject_matches > 0:
                        score += subject_matches * 2.0  # Higher weight for subject
                        matched_keywords.append(keyword)
                    
                    if body_matches > 0:
                        score += body_matches * 1.0
                        if keyword not in matched_keywords:
                            matched_keywords.append(keyword)
                
                # Normalize score by text length
                text_length = len(text_content.split())
                if text_length > 0:
                    score = score / text_length
                
                category_scores[category] = (score, matched_keywords)
            
            # Special handling for spam detection
            spam_indicators = [
                "lottery", "winner", "prize", "click here", "act now",
                "limited time", "urgent", "$", "free", "guarantee"
            ]
            spam_score = sum(1 for indicator in spam_indicators if indicator in text_content)
            if spam_score >= 3:
                category_scores["spam"] = (spam_score * 0.2, spam_indicators[:spam_score])
            
            # Find best category
            if not category_scores:
                best_category = "archive"
                confidence = 0.5
                keywords = []
            else:
                best_category = max(category_scores.keys(), key=lambda k: category_scores[k][0])
                confidence = min(category_scores[best_category][0] * 10, 1.0)  # Scale to 0-1
                keywords = category_scores[best_category][1]
            
            # Apply thresholds
            spam_threshold = self.classification_config.get("spam_threshold", 0.8)
            importance_threshold = self.classification_config.get("importance_threshold", 0.7)
            
            if best_category == "spam" and confidence < spam_threshold:
                best_category = "archive"
                confidence = 0.5
            
            if best_category == "important" and confidence < importance_threshold:
                best_category = "work" if "work" in category_scores else "personal"
            
            classification = EmailClassification(best_category, confidence, keywords)
            
            # Cache the result
            self.classification_cache[email_id] = classification.to_dict()
            self.metrics["classifications_made"] += 1
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Failed to classify email: {e}")
            return EmailClassification("unknown", 0.0)
    
    async def _classify_emails_batch(self, emails: List[Dict[str, Any]]) -> List[EmailClassification]:
        """Classify multiple emails in batch."""
        classifications = []
        
        for email_data in emails:
            classification = await self._classify_email(email_data)
            classifications.append(classification)
        
        return classifications
    
    async def _generate_auto_response(self, email_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate automatic response for an email.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            Response data dictionary or None if no response should be sent
        """
        try:
            if not self.auto_response_config.get("enabled", True):
                return None
            
            # Check rate limiting
            max_responses = self.auto_response_config.get("max_responses_per_day", 50)
            if self.auto_responses_sent_today >= max_responses:
                return None
            
            # Check if we should respond to this sender
            sender = email_data.get("from", "")
            if self._should_skip_auto_response(sender):
                return None
            
            # Classify the email to determine response type
            classification = await self._classify_email(email_data)
            
            # Generate response based on classification and triggers
            response = None
            
            if self.out_of_office_mode:
                response = self._generate_out_of_office_response(email_data)
            else:
                response = self._generate_contextual_response(email_data, classification)
            
            if response:
                self.auto_responses_sent_today += 1
                self.metrics["auto_responses_sent"] += 1
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate auto response: {e}")
            return None
    
    def _should_skip_auto_response(self, sender: str) -> bool:
        """Check if auto response should be skipped for this sender."""
        skip_patterns = [
            "noreply", "no-reply", "donotreply", "do-not-reply",
            "automated", "system", "daemon", "mailer"
        ]
        
        sender_lower = sender.lower()
        return any(pattern in sender_lower for pattern in skip_patterns)
    
    def _generate_out_of_office_response(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate out-of-office auto response."""
        template = self.auto_response_config.get("templates", {}).get(
            "out_of_office",
            "I'm currently out of office and will respond when I return."
        )
        
        return {
            "to": email_data.get("from", ""),
            "subject": f"Re: {email_data.get('subject', '')}",
            "body": template,
            "reply_to_message_id": email_data.get("id", ""),
        }
    
    def _generate_contextual_response(
        self,
        email_data: Dict[str, Any],
        classification: EmailClassification
    ) -> Optional[Dict[str, Any]]:
        """Generate contextual auto response based on email content."""
        subject = email_data.get("subject", "").lower()
        body = email_data.get("body", "").lower()
        
        # Check for meeting requests
        meeting_keywords = ["meeting", "call", "appointment", "schedule", "calendar"]
        if any(keyword in subject or keyword in body for keyword in meeting_keywords):
            template = self.auto_response_config.get("templates", {}).get(
                "meeting_request",
                "Thank you for the meeting request. I'll get back to you soon."
            )
            
            return {
                "to": email_data.get("from", ""),
                "subject": f"Re: {email_data.get('subject', '')}",
                "body": template,
                "reply_to_message_id": email_data.get("id", ""),
            }
        
        # Check for general inquiries
        inquiry_keywords = ["question", "inquiry", "help", "support", "information"]
        if any(keyword in subject or keyword in body for keyword in inquiry_keywords):
            template = self.auto_response_config.get("templates", {}).get(
                "general_inquiry",
                "Thank you for your email. I'll respond as soon as possible."
            )
            
            return {
                "to": email_data.get("from", ""),
                "subject": f"Re: {email_data.get('subject', '')}",
                "body": template,
                "reply_to_message_id": email_data.get("id", ""),
            }
        
        return None
    
    async def _send_auto_response(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send auto response via Gmail API.
        
        Args:
            response_data: Response data dictionary
            
        Returns:
            Sent message data or None if failed
        """
        try:
            await self._check_rate_limit()
            
            # Create email message
            message = MIMEText(response_data["body"])
            message["to"] = response_data["to"]
            message["subject"] = response_data["subject"]
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send via Gmail API
            user_id = self.gmail_config.get("user_email", "me")
            send_request = {
                "raw": raw_message
            }
            
            # Add threading information if available
            if "reply_to_message_id" in response_data:
                send_request["threadId"] = response_data.get("thread_id", "")
            
            result = self.gmail_service.users().messages().send(
                userId=user_id,
                body=send_request
            ).execute()
            
            self.metrics["api_calls_made"] += 1
            self.logger.info(f"Auto response sent to {response_data['to']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to send auto response: {e}")
            return None
    
    async def _determine_labels(self, email_data: Dict[str, Any]) -> List[str]:
        """
        Determine appropriate labels for an email.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            List of label names to apply
        """
        labels = []
        
        try:
            if not self.archiving_config.get("auto_label", True):
                return labels
            
            # Apply label rules
            label_rules = self.archiving_config.get("label_rules", [])
            subject = email_data.get("subject", "").lower()
            body = email_data.get("body", "").lower()
            sender = email_data.get("from", "").lower()
            
            text_content = f"{subject} {body} {sender}"
            
            for rule in label_rules:
                pattern = rule.get("pattern", "").lower()
                label = rule.get("label", "")
                
                if pattern in text_content and label:
                    labels.append(label)
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Failed to determine labels: {e}")
            return []
    
    async def _assign_smart_folder(self, email_data: Dict[str, Any]) -> Optional[str]:
        """
        Assign email to a smart folder based on content.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            Smart folder name or None
        """
        try:
            smart_folders = self.archiving_config.get("smart_folders", {})
            
            subject = email_data.get("subject", "").lower()
            body = email_data.get("body", "").lower()
            text_content = f"{subject} {body}"
            
            for folder_name, keywords in smart_folders.items():
                if any(keyword.lower() in text_content for keyword in keywords):
                    return folder_name
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to assign smart folder: {e}")
            return None
    
    async def _apply_labels(
        self,
        email_id: str,
        labels_to_add: List[str],
        labels_to_remove: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Apply labels to an email.
        
        Args:
            email_id: Email ID
            labels_to_add: Labels to add
            labels_to_remove: Labels to remove
            
        Returns:
            Modified email data or None if failed
        """
        try:
            await self._check_rate_limit()
            
            user_id = self.gmail_config.get("user_email", "me")
            
            modify_request = {}
            if labels_to_add:
                modify_request["addLabelIds"] = labels_to_add
            if labels_to_remove:
                modify_request["removeLabelIds"] = labels_to_remove
            
            if not modify_request:
                return None
            
            result = self.gmail_service.users().messages().modify(
                userId=user_id,
                id=email_id,
                body=modify_request
            ).execute()
            
            self.metrics["api_calls_made"] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to apply labels to email {email_id}: {e}")
            return None
    
    async def _archive_old_emails(self) -> int:
        """
        Archive old emails based on configuration.
        
        Returns:
            Number of emails archived
        """
        try:
            if not self.archiving_config.get("enabled", True):
                return 0
            
            archive_after_days = self.archiving_config.get("archive_after_days", 30)
            cutoff_date = datetime.now() - timedelta(days=archive_after_days)
            
            # Build query for old emails
            query = f"older_than:{archive_after_days}d"
            
            # Fetch old emails
            old_emails = await self._fetch_emails(
                max_results=100,
                query=query
            )
            
            archived_count = 0
            
            for email_data in old_emails:
                # Skip already archived emails
                if "INBOX" not in email_data.get("labels", []):
                    continue
                
                # Archive the email
                result = await self._apply_labels(
                    email_data["id"],
                    labels_to_add=[],
                    labels_to_remove=["INBOX"]
                )
                
                if result:
                    archived_count += 1
                    self.metrics["emails_archived"] += 1
            
            self.logger.info(f"Archived {archived_count} old emails")
            return archived_count
            
        except Exception as e:
            self.logger.error(f"Failed to archive old emails: {e}")
            return 0
    
    async def _process_emails_batch(self, batch_size: int = None) -> None:
        """Process emails in batches."""
        try:
            batch_size = batch_size or self.gmail_config.get("batch_size", 100)
            
            # Fetch unread emails
            emails = await self._fetch_emails(
                max_results=batch_size,
                query="is:unread"
            )
            
            for email_data in emails:
                # Classify email
                classification = await self._classify_email(email_data)
                
                # Determine labels
                labels = await self._determine_labels(email_data)
                
                # Apply auto-labeling
                if labels:
                    await self._apply_labels(email_data["id"], labels)
                
                # Generate auto response if needed
                auto_response = await self._generate_auto_response(email_data)
                if auto_response:
                    await self._send_auto_response(auto_response)
                
                # Assign to smart folder
                smart_folder = await self._assign_smart_folder(email_data)
                if smart_folder:
                    # In a real implementation, this would create/move to folder
                    pass
            
        except Exception as e:
            self.logger.error(f"Failed to process emails batch: {e}")
    
    def _reset_auto_response_tracking(self) -> None:
        """Reset auto-response tracking for new day."""
        now = datetime.now()
        if self.auto_response_reset_time is None or now.date() > self.auto_response_reset_time.date():
            self.auto_responses_sent_today = 0
            self.auto_response_reset_time = now
    
    async def _periodic_email_check(self) -> None:
        """Periodic email checking task."""
        while not self._shutdown_event.is_set():
            try:
                if self.state == AgentState.ACTIVE:
                    await self._process_emails_batch()
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic email check: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _periodic_auto_response_reset(self) -> None:
        """Periodic auto-response counter reset task."""
        while not self._shutdown_event.is_set():
            try:
                self._reset_auto_response_tracking()
                
                # Wait 1 hour before next check
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-response reset: {e}")
                await asyncio.sleep(3600)
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming inter-agent messages.
        
        Args:
            message: Incoming message
            
        Returns:
            Response message or None
        """
        try:
            message_type = message.message_type
            payload = message.payload
            
            if message_type == "fetch_emails":
                # Fetch emails request
                query = payload.get("query")
                max_results = payload.get("max_results", 100)
                label_ids = payload.get("label_ids")
                
                emails = await self._fetch_emails(max_results, query, label_ids)
                
                return self._create_response(
                    message,
                    "fetch_emails_response",
                    {"emails": emails, "count": len(emails)}
                )
            
            elif message_type == "classify_email":
                # Email classification request
                email_data = payload.get("email_data", {})
                classification = await self._classify_email(email_data)
                
                return self._create_response(
                    message,
                    "classify_email_response",
                    {"classification": classification.to_dict()}
                )
            
            elif message_type == "send_response":
                # Send auto response request
                response_data = payload
                result = await self._send_auto_response(response_data)
                
                return self._create_response(
                    message,
                    "send_response_response",
                    {"sent": result is not None, "result": result}
                )
            
            elif message_type == "get_summary":
                # Email summary request
                time_range = payload.get("time_range", "last_24_hours")
                summary = await self._generate_email_summary(time_range)
                
                return self._create_response(
                    message,
                    "get_summary_response",
                    {"summary": summary.to_dict()}
                )
            
            else:
                # Unknown message type
                return self._create_error_response(
                    message,
                    f"Unknown message type: {message_type}"
                )
        
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")
            return self._create_error_response(message, str(e))
    
    def _create_response(
        self,
        original_message: AgentMessage,
        response_type: str,
        payload: Dict[str, Any]
    ) -> AgentMessage:
        """Create response message."""
        return AgentMessage(
            id=f"resp_{original_message.id}",
            sender=self.agent_id,
            recipient=original_message.sender,
            message_type=response_type,
            payload=payload
        )
    
    def _create_error_response(
        self,
        original_message: AgentMessage,
        error_message: str
    ) -> AgentMessage:
        """Create error response message."""
        return AgentMessage(
            id=f"err_{original_message.id}",
            sender=self.agent_id,
            recipient=original_message.sender,
            message_type="error",
            payload={
                "error": True,
                "error_message": error_message,
                "original_message_id": original_message.id
            }
        )
    
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent-specific tasks.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        task_type = task.get("task_type", "")
        parameters = task.get("parameters", {})
        
        if task_type == "email_summary":
            # Generate email summary
            time_range = parameters.get("time_range", "last_24_hours")
            summary = await self._generate_email_summary(time_range)
            return {"summary": summary.to_dict()}
        
        elif task_type == "bulk_archive":
            # Bulk archive emails
            query = parameters.get("query", "older_than:30d")
            batch_size = parameters.get("batch_size", 100)
            dry_run = parameters.get("dry_run", False)
            
            if dry_run:
                emails = await self._fetch_emails(max_results=batch_size, query=query)
                return {"would_archive": len(emails), "dry_run": True}
            else:
                archived_count = await self._archive_old_emails()
                return {"archived_count": archived_count, "dry_run": False}
        
        elif task_type == "label_cleanup":
            # Label cleanup task
            remove_unused = parameters.get("remove_unused_labels", True)
            merge_similar = parameters.get("merge_similar_labels", True)
            dry_run = parameters.get("dry_run", False)
            
            # This would implement label cleanup logic
            return {"labels_removed": 0, "labels_merged": 0, "dry_run": dry_run}
        
        else:
            raise AgentError(f"Unknown task type: {task_type}")
    
    async def _generate_email_summary(self, time_range: str) -> EmailSummary:
        """
        Generate email summary for specified time range.
        
        Args:
            time_range: Time range for summary
            
        Returns:
            EmailSummary object
        """
        summary = EmailSummary()
        summary.time_range = time_range
        
        try:
            # Build query based on time range
            if time_range == "last_24_hours":
                query = "newer_than:1d"
            elif time_range == "last_week":
                query = "newer_than:7d"
            elif time_range == "last_month":
                query = "newer_than:30d"
            else:
                query = "newer_than:1d"
            
            # Fetch emails for the time range
            emails = await self._fetch_emails(max_results=1000, query=query)
            
            summary.total_emails = len(emails)
            
            for email_data in emails:
                # Count unread emails
                if "UNREAD" in email_data.get("labels", []):
                    summary.unread_emails += 1
                
                # Count important emails
                if "IMPORTANT" in email_data.get("labels", []):
                    summary.important_emails += 1
                
                # Count spam emails
                if "SPAM" in email_data.get("labels", []):
                    summary.spam_emails += 1
                
                # Classify and count categories
                classification = await self._classify_email(email_data)
                summary.categories[classification.category] += 1
                
                # Count senders
                sender = email_data.get("from", "")
                if sender:
                    summary.senders[sender] += 1
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate email summary: {e}")
            return summary
    
    # AI-Enhanced Email Processing Methods
    
    async def classify_email_with_ai(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify email using AI service.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            AI classification results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_classification_enabled:
                return {"error": "AI classification not available", "success": False}
            
            # Prepare email content for AI
            content = self._prepare_email_content_for_ai(email_data)
            
            # Perform AI classification
            classification = await self.service_manager.classify_content(
                content=content,
                categories=self.email_categories,
                classification_type="single",
                confidence_threshold=0.6
            )
            
            if classification.get("success"):
                self.metrics["ai_classifications"] += 1
                
                # Extract primary category
                classifications = classification.get("classifications", {})
                if classifications:
                    primary_category = max(classifications.keys(), key=lambda k: classifications[k])
                    confidence = classifications[primary_category]
                    
                    return {
                        "primary_category": primary_category,
                        "confidence": confidence,
                        "all_classifications": classifications,
                        "method": "ai",
                        "success": True
                    }
            
            return classification
            
        except Exception as e:
            self.logger.error(f"AI email classification failed: {e}")
            return {"error": str(e), "success": False}
    
    async def analyze_email_with_ai(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive email analysis using AI.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            AI analysis results
        """
        try:
            if not self.ai_enabled or not self.service_manager:
                return {"error": "AI analysis not available", "success": False}
            
            # Use AI service manager for comprehensive email analysis
            analysis = await self.service_manager.analyze_email(email_data)
            
            if analysis.get("success"):
                # Update metrics
                if self.ai_sentiment_analysis_enabled:
                    self.metrics["ai_summaries_created"] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"AI email analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    async def summarize_email_with_ai(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize email content using AI.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            AI summarization results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_summarization_enabled:
                return {"error": "AI summarization not available", "success": False}
            
            # Prepare content for summarization
            content = self._prepare_email_content_for_ai(email_data)
            
            # Generate AI summary
            summary_result = await self.service_manager.summarize_text(
                text=content,
                max_length=100,
                summary_type="bullet_points"
            )
            
            if summary_result.get("success"):
                self.metrics["ai_summaries_created"] += 1
            
            return {
                "summary": summary_result.get("content", ""),
                "processing_time": summary_result.get("processing_time", 0),
                "success": summary_result.get("success", False),
                "error": summary_result.get("error")
            }
            
        except Exception as e:
            self.logger.error(f"AI email summarization failed: {e}")
            return {"error": str(e), "success": False}
    
    async def generate_email_response_with_ai(
        self,
        email_data: Dict[str, Any],
        response_type: str = "reply",
        tone: str = "professional"
    ) -> List[str]:
        """Generate email response suggestions using AI.
        
        Args:
            email_data: Email data dictionary
            response_type: Type of response (reply, forward, etc.)
            tone: Tone of response (professional, casual, etc.)
            
        Returns:
            List of AI-generated response suggestions
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_response_generation_enabled:
                return ["AI response generation not available"]
            
            # Prepare context for response generation
            context = self._prepare_email_context_for_response(email_data)
            
            # Generate AI suggestions
            suggestions = await self.service_manager.generate_response_suggestions(
                context=context,
                response_type="email",
                tone=tone,
                max_length=200
            )
            
            if suggestions:
                self.metrics["ai_responses_generated"] += 1
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"AI response generation failed: {e}")
            return [f"Error generating AI responses: {e}"]
    
    async def process_email_with_ai_enhancements(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process email with comprehensive AI enhancements.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            Enhanced processing results
        """
        try:
            enhanced_result = {
                "email_id": email_data.get("id"),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "ai_enabled": self.ai_enabled,
                "enhancements": {}
            }
            
            # Perform AI classification if enabled
            if self.ai_classification_enabled and self.ai_enabled:
                classification = await self.classify_email_with_ai(email_data)
                enhanced_result["enhancements"]["ai_classification"] = classification
            
            # Perform AI analysis if enabled
            if self.ai_sentiment_analysis_enabled and self.ai_enabled:
                analysis = await self.analyze_email_with_ai(email_data)
                enhanced_result["enhancements"]["ai_analysis"] = analysis
            
            # Perform AI summarization if enabled
            if self.ai_summarization_enabled and self.ai_enabled:
                summary = await self.summarize_email_with_ai(email_data)
                enhanced_result["enhancements"]["ai_summary"] = summary
            
            # Generate response suggestions if enabled and appropriate
            if self.ai_response_generation_enabled and self.ai_enabled:
                # Only generate responses for certain categories
                classification = enhanced_result["enhancements"].get("ai_classification", {})
                category = classification.get("primary_category")
                
                if category in ["urgent", "work"]:
                    responses = await self.generate_email_response_with_ai(email_data)
                    enhanced_result["enhancements"]["ai_responses"] = responses
            
            enhanced_result["success"] = True
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"AI-enhanced email processing failed: {e}")
            return {
                "email_id": email_data.get("id"),
                "error": str(e),
                "success": False
            }
    
    def _prepare_email_content_for_ai(self, email_data: Dict[str, Any]) -> str:
        """Prepare email content for AI processing.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            Formatted content string for AI processing
        """
        parts = []
        
        if "subject" in email_data:
            parts.append(f"Subject: {email_data['subject']}")
        
        if "from" in email_data:
            parts.append(f"From: {email_data['from']}")
        
        if "body" in email_data:
            # Limit body length for AI processing
            body = email_data['body']
            if len(body) > 2000:  # Limit to ~2000 characters
                body = body[:2000] + "..."
            parts.append(f"Body: {body}")
        
        return "\n".join(parts)
    
    def _prepare_email_context_for_response(self, email_data: Dict[str, Any]) -> str:
        """Prepare email context for AI response generation.
        
        Args:
            email_data: Email data dictionary
            
        Returns:
            Formatted context string for response generation
        """
        context_parts = [
            "Email requiring response:",
            self._prepare_email_content_for_ai(email_data),
            "",
            "Generate professional response suggestions that:"
        ]
        
        # Add context-specific instructions
        if email_data.get("subject", "").lower().startswith("re:"):
            context_parts.append("- Continue the existing conversation thread")
        else:
            context_parts.append("- Address the main points raised in the email")
        
        context_parts.extend([
            "- Maintain a professional tone",
            "- Be concise and actionable",
            "- Include appropriate greetings and closings"
        ])
        
        return "\n".join(context_parts)
    
    async def _update_config(self, new_config: Dict[str, Any]) -> None:
        """Update agent configuration."""
        self.config = new_config
        self.gmail_config = new_config.get("gmail", {})
        self.classification_config = new_config.get("classification", {})
        self.auto_response_config = new_config.get("auto_response", {})
        self.archiving_config = new_config.get("archiving", {})
        
        # Update rate limiter
        self.rate_limiter = GmailRateLimiter(
            self.gmail_config.get("rate_limit_per_minute", 250)
        )
        
        self.logger.info("Gmail Agent configuration updated")
    
    async def _log_activity(self, activity: str, data: Dict[str, Any]) -> None:
        """Log agent activity."""
        self.logger.info(f"Gmail Agent Activity: {activity}", extra=data)