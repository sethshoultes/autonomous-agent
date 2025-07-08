"""
Comprehensive tests for database operations.

This module provides comprehensive test coverage for database operations
including CRUD operations, repositories, and data integrity.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch

from src.database.connection import DatabaseManager
from src.database.operations.base import BaseRepository, QueryBuilder, DatabaseOperationError
from src.database.operations.emails import EmailRepository, EmailThreadRepository, EmailAnalyticsRepository
from src.database.models.emails import Email, EmailThread, EmailAnalytics
from src.database.models.emails import EmailStatus, EmailPriority, EmailDirection, EmailCategory


class TestDatabaseManager:
    """Test cases for DatabaseManager."""
    
    def test_database_manager_init(self):
        """Test DatabaseManager initialization."""
        logger = Mock()
        manager = DatabaseManager(
            database_url="postgresql://user:pass@localhost/test",
            logger=logger
        )
        
        assert manager.database_url == "postgresql://user:pass@localhost/test"
        assert manager.logger == logger
        assert manager.pool_size == 10
        assert manager.max_overflow == 20
        assert not manager.is_healthy
    
    def test_parse_database_url(self):
        """Test database URL parsing."""
        logger = Mock()
        manager = DatabaseManager(
            database_url="postgresql://testuser:testpass@testhost:5433/testdb",
            logger=logger
        )
        
        assert manager.db_params["host"] == "testhost"
        assert manager.db_params["port"] == 5433
        assert manager.db_params["database"] == "testdb"
        assert manager.db_params["user"] == "testuser"
        assert manager.db_params["password"] == "testpass"
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure handling."""
        logger = Mock()
        manager = DatabaseManager(
            database_url="postgresql://invalid:invalid@localhost/test",
            logger=logger
        )
        
        # Mock failed connection
        with patch.object(manager, 'get_connection') as mock_get_connection:
            mock_get_connection.side_effect = Exception("Connection failed")
            
            result = await manager.health_check()
            assert not result
            assert not manager.is_healthy


class TestQueryBuilder:
    """Test cases for QueryBuilder."""
    
    def test_query_builder_init(self):
        """Test QueryBuilder initialization."""
        builder = QueryBuilder(Email)
        
        assert builder.model_class == Email
        assert builder._filters == []
        assert builder._orders == []
        assert builder._limit is None
        assert builder._offset is None
    
    def test_query_builder_filter_by(self):
        """Test filter_by method."""
        builder = QueryBuilder(Email)
        result = builder.filter_by(status=EmailStatus.UNREAD, priority=EmailPriority.HIGH)
        
        assert result is builder  # Fluent interface
        assert len(builder._filters) == 2
    
    def test_query_builder_order_by(self):
        """Test order_by method."""
        builder = QueryBuilder(Email)
        result = builder.order_by(Email.created_at)
        
        assert result is builder
        assert len(builder._orders) == 1
    
    def test_query_builder_pagination(self):
        """Test pagination method."""
        builder = QueryBuilder(Email)
        result = builder.paginate(page=2, per_page=25)
        
        assert result is builder
        assert builder._limit == 25
        assert builder._offset == 25
    
    def test_query_builder_search(self):
        """Test search method."""
        builder = QueryBuilder(Email)
        result = builder.search("test query", Email.subject, Email.body_text)
        
        assert result is builder
        assert len(builder._filters) == 1
    
    def test_query_builder_chaining(self):
        """Test method chaining."""
        builder = QueryBuilder(Email)
        result = (builder
                 .filter_by(status=EmailStatus.UNREAD)
                 .order_by(Email.created_at)
                 .limit(10)
                 .search("important"))
        
        assert result is builder
        assert len(builder._filters) >= 2  # filter_by + search
        assert len(builder._orders) == 1
        assert builder._limit == 10


class TestEmailRepository:
    """Test cases for EmailRepository."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.repository = EmailRepository(self.logger)
        self.sample_email_data = {
            "gmail_id": "test123",
            "message_id": "test@example.com",
            "subject": "Test Email",
            "body_text": "This is a test email",
            "sender_email": "sender@example.com",
            "sender_name": "Test Sender",
            "recipients_to": ["recipient@example.com"],
            "sent_at": datetime.now(timezone.utc),
            "received_at": datetime.now(timezone.utc),
            "direction": EmailDirection.INBOUND,
            "status": EmailStatus.UNREAD,
            "priority": EmailPriority.NORMAL,
            "category": EmailCategory.PERSONAL
        }
    
    @pytest.mark.asyncio
    async def test_get_by_gmail_id_success(self):
        """Test successful retrieval by Gmail ID."""
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = Mock(spec=Email)
            
            result = await self.repository.get_by_gmail_id("test123")
            
            assert result is not None
            mock_session.return_value.__aenter__.return_value.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_gmail_id_not_found(self):
        """Test retrieval by Gmail ID when not found."""
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = None
            
            result = await self.repository.get_by_gmail_id("nonexistent")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_by_gmail_id_error(self):
        """Test error handling in get_by_gmail_id."""
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute.side_effect = Exception("Database error")
            
            with pytest.raises(DatabaseOperationError):
                await self.repository.get_by_gmail_id("test123")
    
    @pytest.mark.asyncio
    async def test_get_unread_emails(self):
        """Test retrieving unread emails."""
        mock_emails = [Mock(spec=Email) for _ in range(3)]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalars.return_value.all.return_value = mock_emails
            
            result = await self.repository.get_unread_emails(limit=5)
            
            assert len(result) == 3
            assert all(isinstance(email, Mock) for email in result)
    
    @pytest.mark.asyncio
    async def test_get_high_priority_emails(self):
        """Test retrieving high priority emails."""
        mock_emails = [Mock(spec=Email) for _ in range(2)]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalars.return_value.all.return_value = mock_emails
            
            result = await self.repository.get_high_priority_emails(limit=10)
            
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_search_emails_with_filters(self):
        """Test email search with filters."""
        mock_emails = [Mock(spec=Email) for _ in range(5)]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalars.return_value.all.return_value = mock_emails
            
            result = await self.repository.search_emails(
                query="test",
                sender="sender@example.com",
                status=EmailStatus.UNREAD,
                priority=EmailPriority.HIGH,
                limit=20
            )
            
            assert len(result) == 5
    
    @pytest.mark.asyncio
    async def test_mark_as_read(self):
        """Test marking email as read."""
        email_id = uuid4()
        
        with patch.object(self.repository, 'update') as mock_update:
            mock_update.return_value = Mock(spec=Email)
            
            result = await self.repository.mark_as_read(email_id)
            
            assert result is True
            mock_update.assert_called_once_with(email_id, status=EmailStatus.READ)
    
    @pytest.mark.asyncio
    async def test_mark_as_processed(self):
        """Test marking email as processed."""
        email_id = uuid4()
        agent_id = uuid4()
        
        with patch.object(self.repository, 'update') as mock_update:
            mock_update.return_value = Mock(spec=Email)
            
            result = await self.repository.mark_as_processed(email_id, agent_id)
            
            assert result is True
            mock_update.assert_called_once()
            args, kwargs = mock_update.call_args
            assert args[0] == email_id
            assert kwargs['status'] == EmailStatus.PROCESSED
            assert kwargs['processed_by'] == agent_id
            assert 'processed_at' in kwargs
    
    @pytest.mark.asyncio
    async def test_update_ai_scores(self):
        """Test updating AI scores."""
        email_id = uuid4()
        
        with patch.object(self.repository, 'update') as mock_update:
            mock_update.return_value = Mock(spec=Email)
            
            result = await self.repository.update_ai_scores(
                email_id,
                sentiment_score=0.5,
                urgency_score=0.8,
                importance_score=0.3
            )
            
            assert result is True
            mock_update.assert_called_once_with(
                email_id,
                sentiment_score=0.5,
                urgency_score=0.8,
                importance_score=0.3
            )
    
    @pytest.mark.asyncio
    async def test_get_email_statistics(self):
        """Test getting email statistics."""
        mock_results = [
            Mock(scalar=Mock(return_value=100)),  # total emails
            Mock(scalar=Mock(return_value=25)),   # unread emails
            Mock(scalar=Mock(return_value=5)),    # high priority emails
            Mock(scalar=Mock(return_value=80)),   # processed emails
            Mock(scalar=Mock(return_value=45.5))  # avg processing time
        ]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.side_effect = mock_results
            
            result = await self.repository.get_email_statistics(days=7)
            
            assert result['total_emails'] == 100
            assert result['unread_emails'] == 25
            assert result['high_priority_emails'] == 5
            assert result['processed_emails'] == 80
            assert result['processing_rate'] == 0.8
            assert result['average_processing_time'] == 45.5
            assert result['period_days'] == 7
    
    @pytest.mark.asyncio
    async def test_get_top_senders(self):
        """Test getting top senders."""
        mock_senders = [
            Mock(sender_email="sender1@example.com", sender_name="Sender 1", email_count=15),
            Mock(sender_email="sender2@example.com", sender_name="Sender 2", email_count=10),
            Mock(sender_email="sender3@example.com", sender_name="Sender 3", email_count=5)
        ]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value = mock_senders
            
            result = await self.repository.get_top_senders(limit=5, days=30)
            
            assert len(result) == 3
            assert result[0]['sender_email'] == "sender1@example.com"
            assert result[0]['email_count'] == 15
            assert result[1]['sender_name'] == "Sender 2"
            assert result[2]['email_count'] == 5


class TestEmailThreadRepository:
    """Test cases for EmailThreadRepository."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.repository = EmailThreadRepository(self.logger)
    
    @pytest.mark.asyncio
    async def test_get_by_gmail_thread_id(self):
        """Test getting thread by Gmail thread ID."""
        mock_thread = Mock(spec=EmailThread)
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalar_one_or_none.return_value = mock_thread
            
            result = await self.repository.get_by_gmail_thread_id("thread123")
            
            assert result == mock_thread
    
    @pytest.mark.asyncio
    async def test_get_active_threads(self):
        """Test getting active threads."""
        mock_threads = [Mock(spec=EmailThread) for _ in range(3)]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalars.return_value.all.return_value = mock_threads
            
            result = await self.repository.get_active_threads(limit=5)
            
            assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_get_threads_by_participant(self):
        """Test getting threads by participant."""
        mock_threads = [Mock(spec=EmailThread) for _ in range(2)]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalars.return_value.all.return_value = mock_threads
            
            result = await self.repository.get_threads_by_participant("participant@example.com")
            
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_update_thread_summary(self):
        """Test updating thread summary."""
        thread_id = uuid4()
        
        with patch.object(self.repository, 'update') as mock_update:
            mock_update.return_value = Mock(spec=EmailThread)
            
            result = await self.repository.update_thread_summary(
                thread_id,
                "Test summary",
                ["key point 1", "key point 2"]
            )
            
            assert result is True
            mock_update.assert_called_once_with(
                thread_id,
                summary="Test summary",
                key_points=["key point 1", "key point 2"]
            )


class TestEmailAnalyticsRepository:
    """Test cases for EmailAnalyticsRepository."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.repository = EmailAnalyticsRepository(self.logger)
    
    @pytest.mark.asyncio
    async def test_record_metric(self):
        """Test recording analytics metric."""
        email_id = uuid4()
        
        with patch.object(self.repository, 'create') as mock_create:
            mock_analytics = Mock(spec=EmailAnalytics)
            mock_create.return_value = mock_analytics
            
            result = await self.repository.record_metric(
                email_id=email_id,
                metric_name="test_metric",
                metric_value=42.5,
                analysis_type="test_analysis",
                analysis_version="v1.0",
                details={"test": "data"}
            )
            
            assert result == mock_analytics
            mock_create.assert_called_once_with(
                email_id=email_id,
                metric_name="test_metric",
                metric_value=42.5,
                analysis_type="test_analysis",
                analysis_version="v1.0",
                details={"test": "data"}
            )
    
    @pytest.mark.asyncio
    async def test_get_email_metrics(self):
        """Test getting metrics for an email."""
        email_id = uuid4()
        mock_metrics = [Mock(spec=EmailAnalytics) for _ in range(3)]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value.scalars.return_value.all.return_value = mock_metrics
            
            result = await self.repository.get_email_metrics(email_id)
            
            assert len(result) == 3
            assert all(isinstance(metric, Mock) for metric in result)
    
    @pytest.mark.asyncio
    async def test_get_metric_trend(self):
        """Test getting metric trend data."""
        mock_trend_data = [
            Mock(date=datetime.now().date(), avg_value=10.5, count=5),
            Mock(date=datetime.now().date(), avg_value=12.0, count=8),
            Mock(date=datetime.now().date(), avg_value=9.5, count=3)
        ]
        
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.return_value = mock_trend_data
            
            result = await self.repository.get_metric_trend(
                metric_name="test_metric",
                days=7,
                analysis_type="test_analysis"
            )
            
            assert len(result) == 3
            assert result[0]['average_value'] == 10.5
            assert result[0]['count'] == 5
            assert result[1]['average_value'] == 12.0
            assert result[2]['count'] == 3


class TestRepositoryStats:
    """Test cases for repository statistics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.email_repo = EmailRepository(self.logger)
        self.thread_repo = EmailThreadRepository(self.logger)
        self.analytics_repo = EmailAnalyticsRepository(self.logger)
    
    @pytest.mark.asyncio
    async def test_email_repository_stats(self):
        """Test email repository statistics."""
        mock_results = [
            Mock(scalar=Mock(return_value=1000)),  # total emails
            [Mock(status=EmailStatus.UNREAD, count=100), Mock(status=EmailStatus.READ, count=900)],  # status dist
            [Mock(priority=EmailPriority.HIGH, count=50), Mock(priority=EmailPriority.NORMAL, count=950)]  # priority dist
        ]
        
        with patch.object(self.email_repo, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.side_effect = mock_results
            
            result = await self.email_repo.get_repository_stats()
            
            assert result['total_emails'] == 1000
            assert result['repository_type'] == "EmailRepository"
            assert EmailStatus.UNREAD in result['status_distribution']
            assert EmailPriority.HIGH in result['priority_distribution']
    
    @pytest.mark.asyncio
    async def test_thread_repository_stats(self):
        """Test thread repository statistics."""
        mock_results = [
            Mock(scalar=Mock(return_value=50)),   # total threads
            Mock(scalar=Mock(return_value=30)),   # active threads
            Mock(scalar=Mock(return_value=5.5))   # avg messages
        ]
        
        with patch.object(self.thread_repo, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.side_effect = mock_results
            
            result = await self.thread_repo.get_repository_stats()
            
            assert result['total_threads'] == 50
            assert result['active_threads'] == 30
            assert result['average_messages_per_thread'] == 5.5
            assert result['repository_type'] == "EmailThreadRepository"
    
    @pytest.mark.asyncio
    async def test_analytics_repository_stats(self):
        """Test analytics repository statistics."""
        mock_results = [
            Mock(scalar=Mock(return_value=5000)),  # total metrics
            Mock(scalar=Mock(return_value=25)),    # unique metrics
            [Mock(metric_name="sentiment", count=1000), Mock(metric_name="urgency", count=800)]  # common metrics
        ]
        
        with patch.object(self.analytics_repo, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            mock_session.return_value.__aenter__.return_value.execute.side_effect = mock_results
            
            result = await self.analytics_repo.get_repository_stats()
            
            assert result['total_metrics'] == 5000
            assert result['unique_metrics'] == 25
            assert len(result['common_metrics']) == 2
            assert result['repository_type'] == "EmailAnalyticsRepository"


class TestErrorHandling:
    """Test cases for error handling in database operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock()
        self.repository = EmailRepository(self.logger)
    
    @pytest.mark.asyncio
    async def test_database_operation_error_handling(self):
        """Test proper error handling in database operations."""
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute.side_effect = Exception("Database connection failed")
            
            with pytest.raises(DatabaseOperationError) as exc_info:
                await self.repository.get_by_gmail_id("test123")
            
            assert "Failed to get email by Gmail ID" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_repository_logging_on_error(self):
        """Test that errors are properly logged."""
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute.side_effect = Exception("Test error")
            
            with pytest.raises(DatabaseOperationError):
                await self.repository.get_unread_emails()
            
            self.logger.error.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when database is unavailable."""
        with patch.object(self.repository, '_get_session') as mock_session:
            mock_session.side_effect = Exception("Database unavailable")
            
            with pytest.raises(DatabaseOperationError):
                await self.repository.get_email_statistics()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        # This would test the TransactionManager rollback functionality
        # Implementation depends on actual transaction handling
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])