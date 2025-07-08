"""
Mock configurations for Gmail API testing.

This module provides comprehensive mocks for Gmail API interactions,
following the TDD approach and ensuring complete isolation from external services.
"""

import base64
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockGmailCredentials:
    """Mock Gmail API credentials."""
    
    def __init__(self):
        self.credentials = {
            "type": "service_account",
            "project_id": "test-project-12345",
            "private_key_id": "test-key-id-67890",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_PRIVATE_KEY_CONTENT\n-----END PRIVATE KEY-----\n",
            "client_email": "test-service-account@test-project-12345.iam.gserviceaccount.com",
            "client_id": "123456789012345678901",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test-service-account%40test-project-12345.iam.gserviceaccount.com"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Return credentials as dictionary."""
        return self.credentials
    
    def to_json(self) -> str:
        """Return credentials as JSON string."""
        return json.dumps(self.credentials, indent=2)


class MockGmailMessage:
    """Mock Gmail message with realistic data."""
    
    def __init__(self, 
                 message_id: str = "test_message_001",
                 thread_id: str = "test_thread_001",
                 from_email: str = "sender@example.com",
                 to_email: str = "recipient@example.com",
                 subject: str = "Test Email Subject",
                 body: str = "This is a test email body",
                 labels: Optional[List[str]] = None,
                 attachments: Optional[List[Dict[str, Any]]] = None):
        
        self.message_id = message_id
        self.thread_id = thread_id
        self.from_email = from_email
        self.to_email = to_email
        self.subject = subject
        self.body = body
        self.labels = labels or ["INBOX", "UNREAD"]
        self.attachments = attachments or []
        self.created_at = datetime.now(timezone.utc)
    
    def to_gmail_format(self) -> Dict[str, Any]:
        """Convert to Gmail API format."""
        encoded_body = base64.urlsafe_b64encode(self.body.encode()).decode()
        
        return {
            "id": self.message_id,
            "threadId": self.thread_id,
            "labelIds": self.labels,
            "snippet": self.body[:100] + "..." if len(self.body) > 100 else self.body,
            "payload": {
                "headers": [
                    {"name": "From", "value": self.from_email},
                    {"name": "To", "value": self.to_email},
                    {"name": "Subject", "value": self.subject},
                    {"name": "Date", "value": self.created_at.strftime("%a, %d %b %Y %H:%M:%S %z")},
                    {"name": "Message-ID", "value": f"<{self.message_id}@example.com>"},
                ],
                "body": {
                    "size": len(self.body),
                    "data": encoded_body,
                },
                "parts": self._create_attachment_parts(),
            },
            "internalDate": str(int(self.created_at.timestamp() * 1000)),
            "historyId": "12345",
            "sizeEstimate": len(self.body) + 500,  # Approximate size
        }
    
    def _create_attachment_parts(self) -> List[Dict[str, Any]]:
        """Create attachment parts for the message."""
        parts = []
        for attachment in self.attachments:
            parts.append({
                "partId": attachment.get("part_id", "001"),
                "mimeType": attachment.get("mime_type", "application/octet-stream"),
                "filename": attachment.get("filename", "attachment.txt"),
                "headers": [
                    {"name": "Content-Type", "value": attachment.get("mime_type", "application/octet-stream")},
                    {"name": "Content-Disposition", "value": f"attachment; filename=\"{attachment.get('filename', 'attachment.txt')}\""},
                ],
                "body": {
                    "attachmentId": attachment.get("attachment_id", "test_attachment_001"),
                    "size": attachment.get("size", 1024),
                },
            })
        return parts


class MockGmailService:
    """Mock Gmail service with realistic behavior."""
    
    def __init__(self):
        self.messages = []
        self.labels = [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "UNREAD", "name": "UNREAD", "type": "system"},
            {"id": "SPAM", "name": "SPAM", "type": "system"},
            {"id": "TRASH", "name": "TRASH", "type": "system"},
        ]
        self.drafts = []
        self.history = []
    
    def add_message(self, message: MockGmailMessage) -> None:
        """Add a message to the mock service."""
        self.messages.append(message)
    
    def create_service_mock(self) -> MagicMock:
        """Create a mock service object."""
        service = MagicMock()
        
        # Mock users().messages().list()
        list_mock = MagicMock()
        list_mock.execute.return_value = {
            "messages": [{"id": msg.message_id, "threadId": msg.thread_id} for msg in self.messages],
            "nextPageToken": None,
            "resultSizeEstimate": len(self.messages),
        }
        service.users.return_value.messages.return_value.list.return_value = list_mock
        
        # Mock users().messages().get()
        def get_message_mock(userId, id, format="full"):
            message = next((msg for msg in self.messages if msg.message_id == id), None)
            if message:
                return MagicMock(execute=lambda: message.to_gmail_format())
            return MagicMock(execute=lambda: {"error": "Message not found"})
        
        service.users.return_value.messages.return_value.get.side_effect = get_message_mock
        
        # Mock users().messages().modify()
        modify_mock = MagicMock()
        modify_mock.execute.return_value = {"id": "modified", "labelIds": ["INBOX"]}
        service.users.return_value.messages.return_value.modify.return_value = modify_mock
        
        # Mock users().messages().delete()
        delete_mock = MagicMock()
        delete_mock.execute.return_value = {}
        service.users.return_value.messages.return_value.delete.return_value = delete_mock
        
        # Mock users().messages().send()
        send_mock = MagicMock()
        send_mock.execute.return_value = {"id": "sent_message_id", "threadId": "sent_thread_id"}
        service.users.return_value.messages.return_value.send.return_value = send_mock
        
        # Mock users().labels().list()
        labels_list_mock = MagicMock()
        labels_list_mock.execute.return_value = {"labels": self.labels}
        service.users.return_value.labels.return_value.list.return_value = labels_list_mock
        
        # Mock users().history().list()
        history_mock = MagicMock()
        history_mock.execute.return_value = {"history": self.history, "nextPageToken": None}
        service.users.return_value.history.return_value.list.return_value = history_mock
        
        return service


class MockGmailBatchProcessor:
    """Mock Gmail batch processor for bulk operations."""
    
    def __init__(self):
        self.batch_requests = []
        self.batch_responses = []
    
    def add_request(self, request: Dict[str, Any]) -> None:
        """Add a request to the batch."""
        self.batch_requests.append(request)
    
    def execute_batch(self) -> List[Dict[str, Any]]:
        """Execute the batch and return responses."""
        responses = []
        for request in self.batch_requests:
            if request["method"] == "GET":
                responses.append({"status": "success", "data": {"id": "mock_id"}})
            elif request["method"] == "POST":
                responses.append({"status": "success", "data": {"id": "created_id"}})
            elif request["method"] == "PUT":
                responses.append({"status": "success", "data": {"id": "updated_id"}})
            elif request["method"] == "DELETE":
                responses.append({"status": "success", "data": {}})
            else:
                responses.append({"status": "error", "error": "Unknown method"})
        
        self.batch_responses = responses
        return responses
    
    def clear(self) -> None:
        """Clear all batch requests and responses."""
        self.batch_requests.clear()
        self.batch_responses.clear()


# Pytest fixtures for Gmail mocks
@pytest.fixture
def mock_gmail_credentials() -> MockGmailCredentials:
    """Provide mock Gmail credentials."""
    return MockGmailCredentials()


@pytest.fixture
def mock_gmail_message() -> MockGmailMessage:
    """Provide a mock Gmail message."""
    return MockGmailMessage()


@pytest.fixture
def mock_gmail_service() -> MockGmailService:
    """Provide a mock Gmail service."""
    return MockGmailService()


@pytest.fixture
def mock_gmail_batch_processor() -> MockGmailBatchProcessor:
    """Provide a mock Gmail batch processor."""
    return MockGmailBatchProcessor()


# Context managers for patching Gmail API
class MockGmailAPIContext:
    """Context manager for mocking Gmail API calls."""
    
    def __init__(self, service: MockGmailService):
        self.service = service
        self.patches = []
    
    def __enter__(self):
        # Mock the build function from googleapiclient.discovery
        build_patch = patch("googleapiclient.discovery.build")
        mock_build = build_patch.start()
        mock_build.return_value = self.service.create_service_mock()
        self.patches.append(build_patch)
        
        # Mock service account credentials
        creds_patch = patch("google.oauth2.service_account.Credentials.from_service_account_info")
        mock_creds = creds_patch.start()
        mock_creds.return_value = MagicMock()
        self.patches.append(creds_patch)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


# Decorator for Gmail API mocking
def mock_gmail_api(service: Optional[MockGmailService] = None):
    """Decorator to mock Gmail API for a test function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            mock_service = service or MockGmailService()
            with MockGmailAPIContext(mock_service):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Sample data generators
def generate_sample_emails(count: int = 5) -> List[MockGmailMessage]:
    """Generate sample emails for testing."""
    emails = []
    for i in range(count):
        email = MockGmailMessage(
            message_id=f"test_message_{i:03d}",
            thread_id=f"test_thread_{i:03d}",
            from_email=f"sender{i}@example.com",
            to_email=f"recipient{i}@example.com",
            subject=f"Test Email {i+1}",
            body=f"This is test email number {i+1} with some sample content.",
            labels=["INBOX", "UNREAD"] if i % 2 == 0 else ["INBOX"],
        )
        emails.append(email)
    return emails


def generate_spam_emails(count: int = 3) -> List[MockGmailMessage]:
    """Generate spam emails for testing."""
    emails = []
    for i in range(count):
        email = MockGmailMessage(
            message_id=f"spam_message_{i:03d}",
            thread_id=f"spam_thread_{i:03d}",
            from_email=f"spam{i}@suspicious.com",
            to_email="recipient@example.com",
            subject=f"URGENT: You've won $1,000,000! Act now {i+1}",
            body=f"Congratulations! You've won our lottery #{i+1}. Click here to claim your prize!",
            labels=["INBOX", "SPAM"],
        )
        emails.append(email)
    return emails


def generate_email_with_attachments(attachment_count: int = 2) -> MockGmailMessage:
    """Generate an email with attachments for testing."""
    attachments = []
    for i in range(attachment_count):
        attachments.append({
            "part_id": f"00{i+1}",
            "attachment_id": f"attachment_{i:03d}",
            "filename": f"document_{i+1}.pdf",
            "mime_type": "application/pdf",
            "size": 1024 * (i + 1),
        })
    
    return MockGmailMessage(
        message_id="email_with_attachments",
        thread_id="thread_with_attachments",
        from_email="sender@example.com",
        to_email="recipient@example.com",
        subject="Email with Attachments",
        body="Please find the attached documents.",
        labels=["INBOX", "UNREAD"],
        attachments=attachments,
    )