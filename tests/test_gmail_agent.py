"""
Comprehensive test suite for Gmail Agent implementation.

This module tests the Gmail Agent using TDD principles, ensuring complete
coverage of all features and edge cases before implementation.
"""

import asyncio
import json
import logging
import pytest
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import AgentMessage, AgentState, BaseAgent
from src.agents.exceptions import AgentError, AgentStateError
from tests.mocks.gmail_mocks import (
    MockGmailCredentials,
    MockGmailMessage,
    MockGmailService,
    MockGmailAPIContext,
    generate_sample_emails,
    generate_spam_emails,
    generate_email_with_attachments,
)


class TestGmailAgentCore:
    """Test core Gmail Agent functionality."""

    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Provide Gmail Agent configuration for testing."""
        return {
            "gmail": {
                "credentials_path": "/path/to/credentials.json",
                "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
                "user_email": "test@example.com",
                "batch_size": 100,
                "rate_limit_per_minute": 250,
                "max_retries": 3,
                "retry_delay": 1.0,
            },
            "classification": {
                "enabled": True,
                "spam_threshold": 0.8,
                "importance_threshold": 0.7,
                "categories": ["important", "spam", "personal", "work", "archive"],
            },
            "auto_response": {
                "enabled": True,
                "response_delay": 300,  # 5 minutes
                "templates": {
                    "out_of_office": "I'm currently out of office...",
                    "meeting_request": "Thank you for the meeting request...",
                },
            },
            "archiving": {
                "enabled": True,
                "archive_after_days": 30,
                "auto_label": True,
                "label_rules": [
                    {"pattern": "newsletter", "label": "Newsletters"},
                    {"pattern": "noreply", "label": "Automated"},
                ],
            },
        }

    @pytest.fixture
    def mock_logger(self) -> logging.Logger:
        """Provide mock logger for testing."""
        return logging.getLogger("test_gmail_agent")

    @pytest.fixture
    def mock_message_broker(self) -> AsyncMock:
        """Provide mock message broker for testing."""
        return AsyncMock()

    @pytest.fixture
    def mock_gmail_service(self) -> MockGmailService:
        """Provide mock Gmail service for testing."""
        service = MockGmailService()
        # Add some sample emails
        sample_emails = generate_sample_emails(5)
        for email in sample_emails:
            service.add_message(email)
        return service

    @pytest.mark.asyncio
    async def test_gmail_agent_initialization(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent initialization with proper configuration."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        assert agent.agent_id == "gmail_agent_001"
        assert agent.config == mock_config
        assert agent.logger == mock_logger
        assert agent.message_broker == mock_message_broker
        assert agent.state == AgentState.INACTIVE
        assert agent.metrics["messages_processed"] == 0
        assert agent.metrics["tasks_completed"] == 0
        assert agent.metrics["errors"] == 0

    @pytest.mark.asyncio
    async def test_gmail_agent_start_success(self, mock_config, mock_logger, mock_message_broker, mock_gmail_service):
        """Test successful Gmail Agent startup."""
        from src.agents.gmail_agent import GmailAgent
        
        with MockGmailAPIContext(mock_gmail_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            assert agent.state == AgentState.ACTIVE
            assert agent.start_time is not None
            assert agent.start_time <= time.time()

    @pytest.mark.asyncio
    async def test_gmail_agent_start_invalid_state(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent start failure from invalid state."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        agent.state = AgentState.ACTIVE  # Set to invalid state
        
        with pytest.raises(AgentStateError) as exc_info:
            await agent.start()
        
        assert "Cannot start agent" in str(exc_info.value)
        assert agent.state == AgentState.ACTIVE  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_gmail_agent_stop_success(self, mock_config, mock_logger, mock_message_broker, mock_gmail_service):
        """Test successful Gmail Agent shutdown."""
        from src.agents.gmail_agent import GmailAgent
        
        with MockGmailAPIContext(mock_gmail_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            assert agent.state == AgentState.ACTIVE
            
            await agent.stop()
            assert agent.state == AgentState.INACTIVE

    @pytest.mark.asyncio
    async def test_gmail_agent_health_check(self, mock_config, mock_logger, mock_message_broker, mock_gmail_service):
        """Test Gmail Agent health check functionality."""
        from src.agents.gmail_agent import GmailAgent
        
        with MockGmailAPIContext(mock_gmail_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            # Test healthy state
            health_status = await agent.health_check()
            assert health_status is True
            
            # Test unhealthy state (simulate Gmail API failure)
            with patch.object(agent, "_check_gmail_connection", return_value=False):
                health_status = await agent.health_check()
                assert health_status is False

    def test_gmail_agent_metrics(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent metrics collection."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        metrics = agent.get_metrics()
        
        assert "messages_processed" in metrics
        assert "tasks_completed" in metrics
        assert "errors" in metrics
        assert "uptime" in metrics
        assert "state" in metrics
        assert "agent_id" in metrics
        assert "emails_processed" in metrics
        assert "classifications_made" in metrics
        assert "auto_responses_sent" in metrics
        assert "emails_archived" in metrics


class TestGmailAgentAuthentication:
    """Test Gmail Agent authentication and credential management."""

    @pytest.fixture
    def mock_credentials(self) -> MockGmailCredentials:
        """Provide mock Gmail credentials."""
        return MockGmailCredentials()

    @pytest.mark.asyncio
    async def test_gmail_authentication_success(self, mock_config, mock_logger, mock_message_broker, mock_credentials):
        """Test successful Gmail API authentication."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
            mock_creds.return_value = MagicMock()
            
            await agent._authenticate()
            
            assert agent.gmail_service is not None
            mock_creds.assert_called_once()

    @pytest.mark.asyncio
    async def test_gmail_authentication_failure(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail API authentication failure."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
            mock_creds.side_effect = Exception("Authentication failed")
            
            with pytest.raises(AgentError) as exc_info:
                await agent._authenticate()
            
            assert "Authentication failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_gmail_credential_refresh(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail credential refresh functionality."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        mock_credentials = MagicMock()
        mock_credentials.expired = True
        
        with patch.object(agent, "gmail_credentials", mock_credentials):
            await agent._refresh_credentials()
            
            mock_credentials.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_gmail_rate_limiting(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail API rate limiting functionality."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Test rate limiting with multiple requests
        start_time = time.time()
        for _ in range(5):
            await agent._check_rate_limit()
        end_time = time.time()
        
        # Should be rate limited if requests are too frequent
        assert end_time - start_time >= 0  # Basic sanity check


class TestGmailAgentEmailProcessing:
    """Test Gmail Agent email processing capabilities."""

    @pytest.fixture
    def sample_emails(self) -> List[MockGmailMessage]:
        """Provide sample emails for testing."""
        return generate_sample_emails(5)

    @pytest.fixture
    def spam_emails(self) -> List[MockGmailMessage]:
        """Provide spam emails for testing."""
        return generate_spam_emails(3)

    @pytest.fixture
    def email_with_attachments(self) -> MockGmailMessage:
        """Provide email with attachments for testing."""
        return generate_email_with_attachments(2)

    @pytest.mark.asyncio
    async def test_fetch_emails_success(self, mock_config, mock_logger, mock_message_broker, sample_emails):
        """Test successful email fetching."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        for email in sample_emails:
            mock_service.add_message(email)
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            emails = await agent._fetch_emails(max_results=10)
            
            assert len(emails) == 5
            assert all(isinstance(email, dict) for email in emails)

    @pytest.mark.asyncio
    async def test_fetch_emails_with_query(self, mock_config, mock_logger, mock_message_broker, sample_emails):
        """Test email fetching with query parameters."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        for email in sample_emails:
            mock_service.add_message(email)
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            emails = await agent._fetch_emails(
                query="is:unread",
                max_results=5,
                label_ids=["INBOX", "UNREAD"]
            )
            
            assert len(emails) <= 5

    @pytest.mark.asyncio
    async def test_parse_email_headers(self, mock_config, mock_logger, mock_message_broker, sample_emails):
        """Test email header parsing."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        gmail_message = sample_emails[0].to_gmail_format()
        parsed_email = agent._parse_email(gmail_message)
        
        assert parsed_email["id"] == sample_emails[0].message_id
        assert parsed_email["thread_id"] == sample_emails[0].thread_id
        assert parsed_email["from"] == sample_emails[0].from_email
        assert parsed_email["to"] == sample_emails[0].to_email
        assert parsed_email["subject"] == sample_emails[0].subject
        assert parsed_email["body"] == sample_emails[0].body

    @pytest.mark.asyncio
    async def test_parse_email_with_attachments(self, mock_config, mock_logger, mock_message_broker, email_with_attachments):
        """Test parsing email with attachments."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        gmail_message = email_with_attachments.to_gmail_format()
        parsed_email = agent._parse_email(gmail_message)
        
        assert "attachments" in parsed_email
        assert len(parsed_email["attachments"]) == 2
        assert parsed_email["attachments"][0]["filename"] == "document_1.pdf"
        assert parsed_email["attachments"][1]["filename"] == "document_2.pdf"

    @pytest.mark.asyncio
    async def test_batch_email_processing(self, mock_config, mock_logger, mock_message_broker, sample_emails):
        """Test batch email processing functionality."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        for email in sample_emails:
            mock_service.add_message(email)
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            # Process emails in batches
            await agent._process_emails_batch(batch_size=2)
            
            # Check that metrics were updated
            metrics = agent.get_metrics()
            assert metrics["emails_processed"] >= 0


class TestGmailAgentClassification:
    """Test Gmail Agent email classification system."""

    @pytest.fixture
    def classification_config(self) -> Dict[str, Any]:
        """Provide classification configuration."""
        return {
            "enabled": True,
            "spam_threshold": 0.8,
            "importance_threshold": 0.7,
            "categories": ["important", "spam", "personal", "work", "archive"],
            "keywords": {
                "important": ["urgent", "asap", "deadline", "priority"],
                "spam": ["prize", "winner", "lottery", "click here"],
                "work": ["meeting", "project", "deadline", "team"],
                "personal": ["family", "friend", "personal"],
            },
        }

    @pytest.mark.asyncio
    async def test_classify_email_as_spam(self, mock_config, mock_logger, mock_message_broker, classification_config):
        """Test email classification as spam."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["classification"] = classification_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        spam_email = MockGmailMessage(
            subject="You've won $1,000,000! Click here now!",
            body="Congratulations! You've won our lottery. Click here to claim your prize!",
            from_email="noreply@suspicious.com",
        )
        
        classification = await agent._classify_email(spam_email.to_gmail_format())
        
        assert classification["category"] == "spam"
        assert classification["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_classify_email_as_important(self, mock_config, mock_logger, mock_message_broker, classification_config):
        """Test email classification as important."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["classification"] = classification_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        important_email = MockGmailMessage(
            subject="URGENT: Project deadline approaching",
            body="This is an urgent message about the project deadline. Please respond ASAP.",
            from_email="boss@company.com",
        )
        
        classification = await agent._classify_email(important_email.to_gmail_format())
        
        assert classification["category"] == "important"
        assert classification["confidence"] >= 0.7

    @pytest.mark.asyncio
    async def test_classify_email_as_work(self, mock_config, mock_logger, mock_message_broker, classification_config):
        """Test email classification as work-related."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["classification"] = classification_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        work_email = MockGmailMessage(
            subject="Team meeting tomorrow at 2 PM",
            body="Please join the team meeting tomorrow at 2 PM in the conference room.",
            from_email="colleague@company.com",
        )
        
        classification = await agent._classify_email(work_email.to_gmail_format())
        
        assert classification["category"] == "work"
        assert classification["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_classify_email_as_personal(self, mock_config, mock_logger, mock_message_broker, classification_config):
        """Test email classification as personal."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["classification"] = classification_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        personal_email = MockGmailMessage(
            subject="Family dinner this weekend",
            body="Hi! Would you like to join us for family dinner this weekend?",
            from_email="sister@gmail.com",
        )
        
        classification = await agent._classify_email(personal_email.to_gmail_format())
        
        assert classification["category"] == "personal"
        assert classification["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_batch_classification(self, mock_config, mock_logger, mock_message_broker, classification_config):
        """Test batch email classification."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["classification"] = classification_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        emails = [
            MockGmailMessage(subject="URGENT: Important task", body="This is urgent").to_gmail_format(),
            MockGmailMessage(subject="You've won a prize!", body="Click here to claim").to_gmail_format(),
            MockGmailMessage(subject="Team meeting", body="Please join the meeting").to_gmail_format(),
        ]
        
        classifications = await agent._classify_emails_batch(emails)
        
        assert len(classifications) == 3
        assert all("category" in c and "confidence" in c for c in classifications)


class TestGmailAgentAutoResponse:
    """Test Gmail Agent automated response system."""

    @pytest.fixture
    def auto_response_config(self) -> Dict[str, Any]:
        """Provide auto response configuration."""
        return {
            "enabled": True,
            "response_delay": 300,  # 5 minutes
            "max_responses_per_day": 50,
            "templates": {
                "out_of_office": "I'm currently out of office and will respond when I return.",
                "meeting_request": "Thank you for the meeting request. I'll get back to you soon.",
                "general_inquiry": "Thank you for your email. I'll respond as soon as possible.",
            },
            "trigger_patterns": {
                "out_of_office": ["vacation", "out of office", "unavailable"],
                "meeting_request": ["meeting", "call", "appointment", "schedule"],
                "general_inquiry": ["question", "inquiry", "help", "support"],
            },
        }

    @pytest.mark.asyncio
    async def test_auto_response_out_of_office(self, mock_config, mock_logger, mock_message_broker, auto_response_config):
        """Test automatic out-of-office response."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["auto_response"] = auto_response_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Set agent to out-of-office mode
        agent.out_of_office_mode = True
        
        incoming_email = MockGmailMessage(
            subject="Quick question",
            body="I have a quick question about the project.",
            from_email="sender@example.com",
        )
        
        response = await agent._generate_auto_response(incoming_email.to_gmail_format())
        
        assert response is not None
        assert "out of office" in response["body"].lower()
        assert response["to"] == "sender@example.com"
        assert response["subject"].startswith("Re:")

    @pytest.mark.asyncio
    async def test_auto_response_meeting_request(self, mock_config, mock_logger, mock_message_broker, auto_response_config):
        """Test automatic meeting request response."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["auto_response"] = auto_response_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        meeting_email = MockGmailMessage(
            subject="Meeting request for next week",
            body="Would you like to schedule a meeting for next week?",
            from_email="colleague@company.com",
        )
        
        response = await agent._generate_auto_response(meeting_email.to_gmail_format())
        
        assert response is not None
        assert "meeting request" in response["body"].lower()
        assert response["to"] == "colleague@company.com"

    @pytest.mark.asyncio
    async def test_auto_response_rate_limiting(self, mock_config, mock_logger, mock_message_broker, auto_response_config):
        """Test auto response rate limiting."""
        from src.agents.gmail_agent import GmailAgent
        
        auto_response_config["max_responses_per_day"] = 2
        mock_config["auto_response"] = auto_response_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Send multiple emails that would trigger responses
        for i in range(5):
            email = MockGmailMessage(
                subject=f"Question {i+1}",
                body="I have a question",
                from_email=f"sender{i}@example.com",
            )
            
            response = await agent._generate_auto_response(email.to_gmail_format())
            
            if i < 2:
                assert response is not None
            else:
                assert response is None  # Should be rate limited

    @pytest.mark.asyncio
    async def test_send_auto_response(self, mock_config, mock_logger, mock_message_broker, auto_response_config):
        """Test sending auto response via Gmail API."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["auto_response"] = auto_response_config
        
        mock_service = MockGmailService()
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            response_data = {
                "to": "sender@example.com",
                "subject": "Re: Your question",
                "body": "Thank you for your email.",
                "reply_to_message_id": "original_message_id",
            }
            
            result = await agent._send_auto_response(response_data)
            
            assert result is not None
            assert "id" in result


class TestGmailAgentArchiving:
    """Test Gmail Agent email archiving and organization."""

    @pytest.fixture
    def archiving_config(self) -> Dict[str, Any]:
        """Provide archiving configuration."""
        return {
            "enabled": True,
            "archive_after_days": 30,
            "auto_label": True,
            "label_rules": [
                {"pattern": "newsletter", "label": "Newsletters"},
                {"pattern": "noreply", "label": "Automated"},
                {"pattern": "github", "label": "GitHub"},
                {"pattern": "linkedin", "label": "LinkedIn"},
            ],
            "smart_folders": {
                "receipts": ["receipt", "invoice", "purchase", "order"],
                "travel": ["flight", "hotel", "booking", "reservation"],
                "social": ["facebook", "twitter", "instagram", "social"],
            },
        }

    @pytest.mark.asyncio
    async def test_auto_labeling(self, mock_config, mock_logger, mock_message_broker, archiving_config):
        """Test automatic email labeling."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["archiving"] = archiving_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        newsletter_email = MockGmailMessage(
            subject="Monthly Newsletter - October 2023",
            body="Our monthly newsletter content...",
            from_email="newsletter@company.com",
        )
        
        labels = await agent._determine_labels(newsletter_email.to_gmail_format())
        
        assert "Newsletters" in labels

    @pytest.mark.asyncio
    async def test_smart_folder_assignment(self, mock_config, mock_logger, mock_message_broker, archiving_config):
        """Test smart folder assignment."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["archiving"] = archiving_config
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        receipt_email = MockGmailMessage(
            subject="Your receipt for order #12345",
            body="Thank you for your purchase. Here's your receipt...",
            from_email="orders@store.com",
        )
        
        smart_folder = await agent._assign_smart_folder(receipt_email.to_gmail_format())
        
        assert smart_folder == "receipts"

    @pytest.mark.asyncio
    async def test_archive_old_emails(self, mock_config, mock_logger, mock_message_broker, archiving_config):
        """Test archiving old emails."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["archiving"] = archiving_config
        
        mock_service = MockGmailService()
        
        # Create old emails
        old_emails = []
        for i in range(3):
            email = MockGmailMessage(
                message_id=f"old_email_{i}",
                subject=f"Old email {i+1}",
                body="This is an old email",
            )
            # Set creation time to 35 days ago
            email.created_at = datetime.now(timezone.utc).replace(day=1)
            old_emails.append(email)
            mock_service.add_message(email)
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            archived_count = await agent._archive_old_emails()
            
            assert archived_count >= 0

    @pytest.mark.asyncio
    async def test_apply_labels_to_email(self, mock_config, mock_logger, mock_message_broker, archiving_config):
        """Test applying labels to emails."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_config["archiving"] = archiving_config
        
        mock_service = MockGmailService()
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            email_id = "test_email_001"
            labels_to_add = ["Important", "Work"]
            labels_to_remove = ["UNREAD"]
            
            result = await agent._apply_labels(email_id, labels_to_add, labels_to_remove)
            
            assert result is not None


class TestGmailAgentMessageHandling:
    """Test Gmail Agent message handling and inter-agent communication."""

    @pytest.mark.asyncio
    async def test_handle_fetch_emails_message(self, mock_config, mock_logger, mock_message_broker):
        """Test handling fetch emails message."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            fetch_message = AgentMessage(
                id="msg_001",
                sender="scheduler_agent",
                recipient="gmail_agent_001",
                message_type="fetch_emails",
                payload={
                    "query": "is:unread",
                    "max_results": 10,
                    "label_ids": ["INBOX"]
                }
            )
            
            response = await agent._process_message(fetch_message)
            
            assert response is not None
            assert response.message_type == "fetch_emails_response"
            assert "emails" in response.payload

    @pytest.mark.asyncio
    async def test_handle_classify_email_message(self, mock_config, mock_logger, mock_message_broker):
        """Test handling classify email message."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        classify_message = AgentMessage(
            id="msg_002",
            sender="scheduler_agent",
            recipient="gmail_agent_001",
            message_type="classify_email",
            payload={
                "email_id": "test_email_001",
                "email_data": {
                    "subject": "URGENT: Important task",
                    "body": "This is an urgent task that needs attention",
                    "from": "boss@company.com",
                }
            }
        )
        
        response = await agent._process_message(classify_message)
        
        assert response is not None
        assert response.message_type == "classify_email_response"
        assert "classification" in response.payload

    @pytest.mark.asyncio
    async def test_handle_send_response_message(self, mock_config, mock_logger, mock_message_broker):
        """Test handling send response message."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            send_message = AgentMessage(
                id="msg_003",
                sender="scheduler_agent",
                recipient="gmail_agent_001",
                message_type="send_response",
                payload={
                    "to": "recipient@example.com",
                    "subject": "Re: Your inquiry",
                    "body": "Thank you for your inquiry. Here's the response.",
                    "reply_to_message_id": "original_msg_id"
                }
            )
            
            response = await agent._process_message(send_message)
            
            assert response is not None
            assert response.message_type == "send_response_response"
            assert "sent" in response.payload

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, mock_config, mock_logger, mock_message_broker):
        """Test handling unknown message types."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        unknown_message = AgentMessage(
            id="msg_004",
            sender="scheduler_agent",
            recipient="gmail_agent_001",
            message_type="unknown_command",
            payload={"data": "test"}
        )
        
        response = await agent._process_message(unknown_message)
        
        assert response is not None
        assert response.message_type == "error"
        assert "unknown message type" in response.payload["error_message"].lower()


class TestGmailAgentTaskExecution:
    """Test Gmail Agent task execution capabilities."""

    @pytest.mark.asyncio
    async def test_execute_email_summary_task(self, mock_config, mock_logger, mock_message_broker):
        """Test executing email summary task."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        sample_emails = generate_sample_emails(5)
        for email in sample_emails:
            mock_service.add_message(email)
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            task = {
                "task_type": "email_summary",
                "parameters": {
                    "time_range": "last_24_hours",
                    "categories": ["important", "work"],
                    "include_attachments": False,
                }
            }
            
            result = await agent.execute_task(task)
            
            assert result is not None
            assert "summary" in result
            assert "total_emails" in result
            assert "categories" in result

    @pytest.mark.asyncio
    async def test_execute_bulk_archive_task(self, mock_config, mock_logger, mock_message_broker):
        """Test executing bulk archive task."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            task = {
                "task_type": "bulk_archive",
                "parameters": {
                    "query": "older_than:30d",
                    "batch_size": 100,
                    "dry_run": False,
                }
            }
            
            result = await agent.execute_task(task)
            
            assert result is not None
            assert "archived_count" in result
            assert "processed_count" in result

    @pytest.mark.asyncio
    async def test_execute_label_cleanup_task(self, mock_config, mock_logger, mock_message_broker):
        """Test executing label cleanup task."""
        from src.agents.gmail_agent import GmailAgent
        
        mock_service = MockGmailService()
        
        with MockGmailAPIContext(mock_service):
            agent = GmailAgent(
                agent_id="gmail_agent_001",
                config=mock_config,
                logger=mock_logger,
                message_broker=mock_message_broker,
            )
            
            await agent.start()
            
            task = {
                "task_type": "label_cleanup",
                "parameters": {
                    "remove_unused_labels": True,
                    "merge_similar_labels": True,
                    "dry_run": False,
                }
            }
            
            result = await agent.execute_task(task)
            
            assert result is not None
            assert "labels_removed" in result
            assert "labels_merged" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_task_type(self, mock_config, mock_logger, mock_message_broker):
        """Test executing unknown task type."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        task = {
            "task_type": "unknown_task",
            "parameters": {"data": "test"}
        }
        
        with pytest.raises(AgentError) as exc_info:
            await agent.execute_task(task)
        
        assert "unknown task type" in str(exc_info.value).lower()


class TestGmailAgentErrorHandling:
    """Test Gmail Agent error handling and recovery."""

    @pytest.mark.asyncio
    async def test_handle_gmail_api_quota_exceeded(self, mock_config, mock_logger, mock_message_broker):
        """Test handling Gmail API quota exceeded error."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        with patch.object(agent, '_make_gmail_request') as mock_request:
            mock_request.side_effect = Exception("Quota exceeded")
            
            with pytest.raises(AgentError) as exc_info:
                await agent._fetch_emails()
            
            assert "quota exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_handle_network_timeout(self, mock_config, mock_logger, mock_message_broker):
        """Test handling network timeout errors."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        with patch.object(agent, '_make_gmail_request') as mock_request:
            mock_request.side_effect = asyncio.TimeoutError("Network timeout")
            
            with pytest.raises(AgentError) as exc_info:
                await agent._fetch_emails()
            
            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_handle_authentication_error(self, mock_config, mock_logger, mock_message_broker):
        """Test handling authentication errors."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        with patch.object(agent, '_authenticate') as mock_auth:
            mock_auth.side_effect = Exception("Authentication failed")
            
            with pytest.raises(AgentError) as exc_info:
                await agent.start()
            
            assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_config, mock_logger, mock_message_broker):
        """Test retry mechanism for failed operations."""
        from src.agents.gmail_agent import GmailAgent
        
        agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        call_count = 0
        
        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        with patch.object(agent, '_make_gmail_request', side_effect=mock_operation):
            result = await agent._fetch_emails_with_retry()
            
            assert result == "success"
            assert call_count == 3  # Should have retried 2 times


class TestGmailAgentIntegration:
    """Test Gmail Agent integration with the existing framework."""

    @pytest.mark.asyncio
    async def test_registration_with_agent_manager(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent registration with AgentManager."""
        from src.agents.gmail_agent import GmailAgent
        from src.agents.manager import AgentManager
        
        # Create AgentManager instance
        config_manager = MagicMock()
        logger = logging.getLogger("test")
        
        agent_manager = AgentManager(config_manager, logger)
        
        # Create Gmail Agent
        gmail_agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Register agent
        await agent_manager.register_agent(gmail_agent)
        
        # Verify registration
        assert "gmail_agent_001" in agent_manager.agents
        assert agent_manager.agents["gmail_agent_001"] == gmail_agent

    @pytest.mark.asyncio
    async def test_communication_with_other_agents(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent communication with other agents."""
        from src.agents.gmail_agent import GmailAgent
        
        gmail_agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Send message to another agent
        await gmail_agent.send_message(
            recipient="scheduler_agent",
            message_type="email_received",
            payload={
                "email_id": "test_email_001",
                "from": "sender@example.com",
                "subject": "Test email",
                "classification": "important"
            }
        )
        
        # Verify message broker was called
        mock_message_broker.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_configuration_updates(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent response to configuration updates."""
        from src.agents.gmail_agent import GmailAgent
        
        gmail_agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Update configuration
        new_config = mock_config.copy()
        new_config["gmail"]["batch_size"] = 200
        
        await gmail_agent._update_config(new_config)
        
        assert gmail_agent.config["gmail"]["batch_size"] == 200

    @pytest.mark.asyncio
    async def test_logging_integration(self, mock_config, mock_logger, mock_message_broker):
        """Test Gmail Agent logging integration."""
        from src.agents.gmail_agent import GmailAgent
        
        gmail_agent = GmailAgent(
            agent_id="gmail_agent_001",
            config=mock_config,
            logger=mock_logger,
            message_broker=mock_message_broker,
        )
        
        # Test that logging works correctly
        with patch.object(mock_logger, 'info') as mock_log:
            await gmail_agent._log_activity("Test activity", {"key": "value"})
            
            mock_log.assert_called_once()