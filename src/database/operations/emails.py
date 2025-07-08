"""
Email-related database operations.

This module provides specialized repository operations for email management,
analytics, and thread tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.emails import Email, EmailThread, EmailAnalytics, EmailAttachment, EmailStatus, EmailPriority
from .base import BaseRepository, DatabaseOperationError, QueryBuilder


class EmailRepository(BaseRepository):
    """Repository for email operations."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize email repository."""
        super().__init__(Email, logger)
    
    async def get_by_gmail_id(self, gmail_id: str) -> Optional[Email]:
        """Get email by Gmail ID."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(Email).where(Email.gmail_id == gmail_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting email by Gmail ID {gmail_id}: {e}")
            raise DatabaseOperationError(f"Failed to get email by Gmail ID: {e}")
    
    async def get_by_message_id(self, message_id: str) -> Optional[Email]:
        """Get email by message ID."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(Email).where(Email.message_id == message_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting email by message ID {message_id}: {e}")
            raise DatabaseOperationError(f"Failed to get email by message ID: {e}")
    
    async def get_unread_emails(self, limit: Optional[int] = None) -> List[Email]:
        """Get unread emails."""
        try:
            async with (await self._get_session()) as session:
                query = select(Email).where(
                    Email.status == EmailStatus.UNREAD
                ).order_by(desc(Email.received_at))
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting unread emails: {e}")
            raise DatabaseOperationError(f"Failed to get unread emails: {e}")
    
    async def get_high_priority_emails(self, limit: Optional[int] = None) -> List[Email]:
        """Get high priority emails."""
        try:
            async with (await self._get_session()) as session:
                query = select(Email).where(
                    Email.priority.in_([EmailPriority.HIGH, EmailPriority.URGENT])
                ).order_by(
                    desc(Email.priority),
                    desc(Email.received_at)
                )
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting high priority emails: {e}")
            raise DatabaseOperationError(f"Failed to get high priority emails: {e}")
    
    async def get_emails_by_sender(self, sender_email: str, limit: Optional[int] = None) -> List[Email]:
        """Get emails from a specific sender."""
        try:
            async with (await self._get_session()) as session:
                query = select(Email).where(
                    Email.sender_email == sender_email.lower()
                ).order_by(desc(Email.received_at))
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting emails by sender {sender_email}: {e}")
            raise DatabaseOperationError(f"Failed to get emails by sender: {e}")
    
    async def get_emails_by_thread(self, thread_id: UUID) -> List[Email]:
        """Get emails in a specific thread."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(Email).where(
                        Email.thread_id == thread_id
                    ).order_by(Email.sent_at)
                )
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting emails by thread {thread_id}: {e}")
            raise DatabaseOperationError(f"Failed to get emails by thread: {e}")
    
    async def get_emails_in_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> List[Email]:
        """Get emails within a date range."""
        try:
            async with (await self._get_session()) as session:
                query = select(Email).where(
                    and_(
                        Email.received_at >= start_date,
                        Email.received_at <= end_date
                    )
                ).order_by(desc(Email.received_at))
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting emails in date range: {e}")
            raise DatabaseOperationError(f"Failed to get emails in date range: {e}")
    
    async def search_emails(
        self,
        query: str,
        sender: Optional[str] = None,
        status: Optional[EmailStatus] = None,
        priority: Optional[EmailPriority] = None,
        limit: Optional[int] = None
    ) -> List[Email]:
        """Search emails by content and filters."""
        try:
            async with (await self._get_session()) as session:
                search_query = select(Email)
                
                # Text search in subject and body
                if query:
                    search_conditions = [
                        Email.subject.ilike(f"%{query}%"),
                        Email.body_text.ilike(f"%{query}%"),
                        Email.snippet.ilike(f"%{query}%")
                    ]
                    search_query = search_query.where(or_(*search_conditions))
                
                # Sender filter
                if sender:
                    search_query = search_query.where(
                        Email.sender_email == sender.lower()
                    )
                
                # Status filter
                if status:
                    search_query = search_query.where(Email.status == status)
                
                # Priority filter
                if priority:
                    search_query = search_query.where(Email.priority == priority)
                
                # Order by relevance and date
                search_query = search_query.order_by(desc(Email.received_at))
                
                if limit:
                    search_query = search_query.limit(limit)
                
                result = await session.execute(search_query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error searching emails: {e}")
            raise DatabaseOperationError(f"Failed to search emails: {e}")
    
    async def mark_as_read(self, email_id: UUID) -> bool:
        """Mark an email as read."""
        try:
            result = await self.update(email_id, status=EmailStatus.READ)
            return result is not None
        except Exception as e:
            self.logger.error(f"Error marking email {email_id} as read: {e}")
            raise DatabaseOperationError(f"Failed to mark email as read: {e}")
    
    async def mark_as_processed(self, email_id: UUID, processed_by: UUID) -> bool:
        """Mark an email as processed."""
        try:
            result = await self.update(
                email_id,
                status=EmailStatus.PROCESSED,
                processed_at=datetime.utcnow(),
                processed_by=processed_by
            )
            return result is not None
        except Exception as e:
            self.logger.error(f"Error marking email {email_id} as processed: {e}")
            raise DatabaseOperationError(f"Failed to mark email as processed: {e}")
    
    async def update_ai_scores(
        self,
        email_id: UUID,
        sentiment_score: Optional[float] = None,
        urgency_score: Optional[float] = None,
        importance_score: Optional[float] = None
    ) -> bool:
        """Update AI analysis scores for an email."""
        try:
            updates = {}
            if sentiment_score is not None:
                updates["sentiment_score"] = sentiment_score
            if urgency_score is not None:
                updates["urgency_score"] = urgency_score
            if importance_score is not None:
                updates["importance_score"] = importance_score
            
            if updates:
                result = await self.update(email_id, **updates)
                return result is not None
            return False
        except Exception as e:
            self.logger.error(f"Error updating AI scores for email {email_id}: {e}")
            raise DatabaseOperationError(f"Failed to update AI scores: {e}")
    
    async def get_email_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get email statistics for the last N days."""
        try:
            async with (await self._get_session()) as session:
                start_date = datetime.utcnow() - timedelta(days=days)
                
                # Total emails
                total_query = select(func.count(Email.id)).where(
                    Email.received_at >= start_date
                )
                total_result = await session.execute(total_query)
                total_emails = total_result.scalar()
                
                # Unread emails
                unread_query = select(func.count(Email.id)).where(
                    and_(
                        Email.received_at >= start_date,
                        Email.status == EmailStatus.UNREAD
                    )
                )
                unread_result = await session.execute(unread_query)
                unread_emails = unread_result.scalar()
                
                # High priority emails
                high_priority_query = select(func.count(Email.id)).where(
                    and_(
                        Email.received_at >= start_date,
                        Email.priority.in_([EmailPriority.HIGH, EmailPriority.URGENT])
                    )
                )
                high_priority_result = await session.execute(high_priority_query)
                high_priority_emails = high_priority_result.scalar()
                
                # Processed emails
                processed_query = select(func.count(Email.id)).where(
                    and_(
                        Email.received_at >= start_date,
                        Email.status == EmailStatus.PROCESSED
                    )
                )
                processed_result = await session.execute(processed_query)
                processed_emails = processed_result.scalar()
                
                # Average processing time
                avg_processing_query = select(
                    func.avg(Email.processing_duration)
                ).where(
                    and_(
                        Email.received_at >= start_date,
                        Email.processing_duration.is_not(None)
                    )
                )
                avg_processing_result = await session.execute(avg_processing_query)
                avg_processing_time = avg_processing_result.scalar() or 0
                
                return {
                    "total_emails": total_emails,
                    "unread_emails": unread_emails,
                    "high_priority_emails": high_priority_emails,
                    "processed_emails": processed_emails,
                    "processing_rate": processed_emails / total_emails if total_emails > 0 else 0,
                    "average_processing_time": float(avg_processing_time),
                    "period_days": days
                }
        except Exception as e:
            self.logger.error(f"Error getting email statistics: {e}")
            raise DatabaseOperationError(f"Failed to get email statistics: {e}")
    
    async def get_top_senders(self, limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
        """Get top email senders by volume."""
        try:
            async with (await self._get_session()) as session:
                start_date = datetime.utcnow() - timedelta(days=days)
                
                query = select(
                    Email.sender_email,
                    Email.sender_name,
                    func.count(Email.id).label("email_count")
                ).where(
                    Email.received_at >= start_date
                ).group_by(
                    Email.sender_email,
                    Email.sender_name
                ).order_by(
                    desc("email_count")
                ).limit(limit)
                
                result = await session.execute(query)
                return [
                    {
                        "sender_email": row.sender_email,
                        "sender_name": row.sender_name,
                        "email_count": row.email_count
                    }
                    for row in result
                ]
        except Exception as e:
            self.logger.error(f"Error getting top senders: {e}")
            raise DatabaseOperationError(f"Failed to get top senders: {e}")
    
    async def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository-specific statistics."""
        try:
            async with (await self._get_session()) as session:
                # Total emails
                total_query = select(func.count(Email.id))
                total_result = await session.execute(total_query)
                total_emails = total_result.scalar()
                
                # Status distribution
                status_query = select(
                    Email.status,
                    func.count(Email.id).label("count")
                ).group_by(Email.status)
                status_result = await session.execute(status_query)
                status_distribution = {
                    row.status: row.count for row in status_result
                }
                
                # Priority distribution
                priority_query = select(
                    Email.priority,
                    func.count(Email.id).label("count")
                ).group_by(Email.priority)
                priority_result = await session.execute(priority_query)
                priority_distribution = {
                    row.priority: row.count for row in priority_result
                }
                
                return {
                    "total_emails": total_emails,
                    "status_distribution": status_distribution,
                    "priority_distribution": priority_distribution,
                    "repository_type": "EmailRepository"
                }
        except Exception as e:
            self.logger.error(f"Error getting repository stats: {e}")
            raise DatabaseOperationError(f"Failed to get repository stats: {e}")


class EmailThreadRepository(BaseRepository):
    """Repository for email thread operations."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize email thread repository."""
        super().__init__(EmailThread, logger)
    
    async def get_by_gmail_thread_id(self, gmail_thread_id: str) -> Optional[EmailThread]:
        """Get thread by Gmail thread ID."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(EmailThread).where(
                        EmailThread.gmail_thread_id == gmail_thread_id
                    )
                )
                return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting thread by Gmail ID {gmail_thread_id}: {e}")
            raise DatabaseOperationError(f"Failed to get thread by Gmail ID: {e}")
    
    async def get_active_threads(self, limit: Optional[int] = None) -> List[EmailThread]:
        """Get active email threads."""
        try:
            async with (await self._get_session()) as session:
                query = select(EmailThread).where(
                    EmailThread.is_active == True
                ).order_by(desc(EmailThread.last_message_at))
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting active threads: {e}")
            raise DatabaseOperationError(f"Failed to get active threads: {e}")
    
    async def get_threads_by_participant(self, participant_email: str) -> List[EmailThread]:
        """Get threads involving a specific participant."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(EmailThread).where(
                        EmailThread.participants.contains([participant_email.lower()])
                    ).order_by(desc(EmailThread.last_message_at))
                )
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting threads by participant {participant_email}: {e}")
            raise DatabaseOperationError(f"Failed to get threads by participant: {e}")
    
    async def update_thread_summary(self, thread_id: UUID, summary: str, key_points: List[str]) -> bool:
        """Update thread summary and key points."""
        try:
            result = await self.update(
                thread_id,
                summary=summary,
                key_points=key_points
            )
            return result is not None
        except Exception as e:
            self.logger.error(f"Error updating thread summary {thread_id}: {e}")
            raise DatabaseOperationError(f"Failed to update thread summary: {e}")
    
    async def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository-specific statistics."""
        try:
            async with (await self._get_session()) as session:
                # Total threads
                total_query = select(func.count(EmailThread.id))
                total_result = await session.execute(total_query)
                total_threads = total_result.scalar()
                
                # Active threads
                active_query = select(func.count(EmailThread.id)).where(
                    EmailThread.is_active == True
                )
                active_result = await session.execute(active_query)
                active_threads = active_result.scalar()
                
                # Average messages per thread
                avg_messages_query = select(
                    func.avg(EmailThread.message_count)
                )
                avg_messages_result = await session.execute(avg_messages_query)
                avg_messages = avg_messages_result.scalar() or 0
                
                return {
                    "total_threads": total_threads,
                    "active_threads": active_threads,
                    "average_messages_per_thread": float(avg_messages),
                    "repository_type": "EmailThreadRepository"
                }
        except Exception as e:
            self.logger.error(f"Error getting repository stats: {e}")
            raise DatabaseOperationError(f"Failed to get repository stats: {e}")


class EmailAnalyticsRepository(BaseRepository):
    """Repository for email analytics operations."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize email analytics repository."""
        super().__init__(EmailAnalytics, logger)
    
    async def record_metric(
        self,
        email_id: UUID,
        metric_name: str,
        metric_value: float,
        analysis_type: str,
        analysis_version: str,
        details: Optional[Dict[str, Any]] = None
    ) -> EmailAnalytics:
        """Record an analytics metric for an email."""
        try:
            return await self.create(
                email_id=email_id,
                metric_name=metric_name,
                metric_value=metric_value,
                analysis_type=analysis_type,
                analysis_version=analysis_version,
                details=details or {}
            )
        except Exception as e:
            self.logger.error(f"Error recording metric {metric_name} for email {email_id}: {e}")
            raise DatabaseOperationError(f"Failed to record metric: {e}")
    
    async def get_email_metrics(self, email_id: UUID) -> List[EmailAnalytics]:
        """Get all metrics for an email."""
        try:
            async with (await self._get_session()) as session:
                result = await session.execute(
                    select(EmailAnalytics).where(
                        EmailAnalytics.email_id == email_id
                    ).order_by(EmailAnalytics.created_at)
                )
                return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting metrics for email {email_id}: {e}")
            raise DatabaseOperationError(f"Failed to get email metrics: {e}")
    
    async def get_metric_trend(
        self,
        metric_name: str,
        days: int = 30,
        analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get trend data for a specific metric."""
        try:
            async with (await self._get_session()) as session:
                start_date = datetime.utcnow() - timedelta(days=days)
                
                query = select(
                    func.date_trunc('day', EmailAnalytics.created_at).label('date'),
                    func.avg(EmailAnalytics.metric_value).label('avg_value'),
                    func.count(EmailAnalytics.id).label('count')
                ).where(
                    and_(
                        EmailAnalytics.metric_name == metric_name,
                        EmailAnalytics.created_at >= start_date
                    )
                )
                
                if analysis_type:
                    query = query.where(EmailAnalytics.analysis_type == analysis_type)
                
                query = query.group_by(
                    func.date_trunc('day', EmailAnalytics.created_at)
                ).order_by('date')
                
                result = await session.execute(query)
                return [
                    {
                        "date": row.date,
                        "average_value": float(row.avg_value),
                        "count": row.count
                    }
                    for row in result
                ]
        except Exception as e:
            self.logger.error(f"Error getting metric trend for {metric_name}: {e}")
            raise DatabaseOperationError(f"Failed to get metric trend: {e}")
    
    async def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository-specific statistics."""
        try:
            async with (await self._get_session()) as session:
                # Total metrics
                total_query = select(func.count(EmailAnalytics.id))
                total_result = await session.execute(total_query)
                total_metrics = total_result.scalar()
                
                # Unique metrics
                unique_query = select(func.count(func.distinct(EmailAnalytics.metric_name)))
                unique_result = await session.execute(unique_query)
                unique_metrics = unique_result.scalar()
                
                # Most common metrics
                common_query = select(
                    EmailAnalytics.metric_name,
                    func.count(EmailAnalytics.id).label("count")
                ).group_by(
                    EmailAnalytics.metric_name
                ).order_by(desc("count")).limit(5)
                
                common_result = await session.execute(common_query)
                common_metrics = [
                    {"metric": row.metric_name, "count": row.count}
                    for row in common_result
                ]
                
                return {
                    "total_metrics": total_metrics,
                    "unique_metrics": unique_metrics,
                    "common_metrics": common_metrics,
                    "repository_type": "EmailAnalyticsRepository"
                }
        except Exception as e:
            self.logger.error(f"Error getting repository stats: {e}")
            raise DatabaseOperationError(f"Failed to get repository stats: {e}")