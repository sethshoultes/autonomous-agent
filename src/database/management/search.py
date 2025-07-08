"""
Advanced search capabilities for the database.

This module provides comprehensive search functionality across all data types
with support for full-text search, filtering, faceted search, and analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import and_, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..connection import get_database_manager
from ..models.emails import Email, EmailThread, EmailStatus, EmailPriority, EmailCategory
from ..models.research import ResearchQuery, ResearchResult, KnowledgeBase
from ..models.agents import Agent, AgentTask, TaskStatus
from ..models.users import User, UserPreference
from ...core.exceptions import CoreError


class SearchError(CoreError):
    """Base exception for search operations."""
    pass


class SearchResult:
    """Represents a search result with metadata."""
    
    def __init__(
        self,
        item: Any,
        item_type: str,
        score: float = 0.0,
        highlights: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize search result.
        
        Args:
            item: The actual database item
            item_type: Type of the item (email, research, etc.)
            score: Relevance score (0-1)
            highlights: List of highlighted text snippets
            metadata: Additional metadata about the result
        """
        self.item = item
        self.item_type = item_type
        self.score = score
        self.highlights = highlights or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "item": self.item.to_dict() if hasattr(self.item, 'to_dict') else str(self.item),
            "item_type": self.item_type,
            "score": self.score,
            "highlights": self.highlights,
            "metadata": self.metadata
        }


class SearchFacet:
    """Represents a search facet for filtering."""
    
    def __init__(self, name: str, values: List[Tuple[str, int]]):
        """
        Initialize search facet.
        
        Args:
            name: Facet name
            values: List of (value, count) tuples
        """
        self.name = name
        self.values = values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "values": [{"value": value, "count": count} for value, count in self.values]
        }


class SearchManager:
    """
    Advanced search manager for database operations.
    
    Provides comprehensive search capabilities including:
    - Full-text search across multiple data types
    - Faceted search with filters
    - Search analytics and suggestions
    - Saved searches and search history
    - Performance optimization
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize search manager.
        
        Args:
            logger: Logger instance for search operations
        """
        self.logger = logger
        self._db_manager = None
        
        # Search configuration
        self.search_config = {
            "max_results": 100,
            "default_limit": 20,
            "min_score": 0.1,
            "highlight_length": 200,
            "facet_limit": 10
        }
        
        # Supported search types
        self.searchable_types = {
            "emails": Email,
            "threads": EmailThread,
            "research": ResearchQuery,
            "knowledge": KnowledgeBase,
            "agents": Agent,
            "tasks": AgentTask,
            "users": User
        }
    
    async def _get_db_manager(self):
        """Get database manager instance."""
        if self._db_manager is None:
            self._db_manager = await get_database_manager()
        return self._db_manager
    
    async def search_all(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "relevance",
        include_facets: bool = False
    ) -> Dict[str, Any]:
        """
        Search across all data types.
        
        Args:
            query: Search query string
            filters: Optional filters to apply
            limit: Maximum number of results to return
            offset: Offset for pagination
            sort_by: Sort order (relevance, date, etc.)
            include_facets: Whether to include faceted search results
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            self.logger.info(f"Performing global search for: {query}")
            
            # Initialize results
            results = []
            facets = []
            total_count = 0
            
            # Search each data type
            for search_type, model_class in self.searchable_types.items():
                try:
                    type_results = await self._search_by_type(
                        model_class,
                        search_type,
                        query,
                        filters,
                        limit // len(self.searchable_types)
                    )
                    results.extend(type_results)
                    
                except Exception as e:
                    self.logger.error(f"Error searching {search_type}: {e}")
                    continue
            
            # Sort results by relevance/score
            if sort_by == "relevance":
                results.sort(key=lambda x: x.score, reverse=True)
            elif sort_by == "date":
                results.sort(key=lambda x: getattr(x.item, 'created_at', datetime.min), reverse=True)
            
            # Apply pagination
            paginated_results = results[offset:offset + limit]
            
            # Generate facets if requested
            if include_facets:
                facets = await self._generate_facets(results)
            
            # Record search analytics
            await self._record_search_analytics(query, len(results), filters)
            
            return {
                "results": [result.to_dict() for result in paginated_results],
                "total_count": len(results),
                "facets": [facet.to_dict() for facet in facets],
                "query": query,
                "filters": filters,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < len(results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Global search failed: {e}")
            raise SearchError(f"Search failed: {e}")
    
    async def search_emails(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Search emails with advanced filtering.
        
        Args:
            query: Search query string
            filters: Email-specific filters
            limit: Maximum results to return
            offset: Offset for pagination
            
        Returns:
            List of search results
        """
        try:
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_session() as session:
                # Build base query
                search_query = select(Email)
                
                # Add text search conditions
                if query:
                    search_conditions = [
                        Email.subject.ilike(f"%{query}%"),
                        Email.body_text.ilike(f"%{query}%"),
                        Email.sender_name.ilike(f"%{query}%"),
                        Email.sender_email.ilike(f"%{query}%")
                    ]
                    search_query = search_query.where(or_(*search_conditions))
                
                # Apply filters
                if filters:
                    search_query = await self._apply_email_filters(search_query, filters)
                
                # Add ordering
                search_query = search_query.order_by(desc(Email.received_at))
                
                # Apply pagination
                search_query = search_query.limit(limit).offset(offset)
                
                # Execute query
                result = await session.execute(search_query)
                emails = result.scalars().all()
                
                # Create search results with scoring
                search_results = []
                for email in emails:
                    score = await self._calculate_email_score(email, query)
                    highlights = await self._generate_email_highlights(email, query)
                    
                    search_results.append(SearchResult(
                        item=email,
                        item_type="email",
                        score=score,
                        highlights=highlights,
                        metadata={
                            "thread_id": email.thread_id,
                            "has_attachments": email.attachment_count > 0,
                            "priority": email.priority.value,
                            "category": email.category.value
                        }
                    ))
                
                return search_results
                
        except Exception as e:
            self.logger.error(f"Email search failed: {e}")
            raise SearchError(f"Email search failed: {e}")
    
    async def search_research(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Search research queries and results.
        
        Args:
            query: Search query string
            filters: Research-specific filters
            limit: Maximum results to return
            offset: Offset for pagination
            
        Returns:
            List of search results
        """
        try:
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_session() as session:
                # Search research queries
                query_search = select(ResearchQuery)
                
                if query:
                    query_conditions = [
                        ResearchQuery.query_text.ilike(f"%{query}%"),
                        ResearchQuery.refined_query.ilike(f"%{query}%"),
                        ResearchQuery.summary.ilike(f"%{query}%")
                    ]
                    query_search = query_search.where(or_(*query_conditions))
                
                # Apply filters
                if filters:
                    query_search = await self._apply_research_filters(query_search, filters)
                
                query_search = query_search.order_by(desc(ResearchQuery.created_at))
                query_search = query_search.limit(limit).offset(offset)
                
                result = await session.execute(query_search)
                research_queries = result.scalars().all()
                
                # Create search results
                search_results = []
                for research_query in research_queries:
                    score = await self._calculate_research_score(research_query, query)
                    highlights = await self._generate_research_highlights(research_query, query)
                    
                    search_results.append(SearchResult(
                        item=research_query,
                        item_type="research",
                        score=score,
                        highlights=highlights,
                        metadata={
                            "research_type": research_query.research_type.value,
                            "status": research_query.status.value,
                            "total_results": research_query.total_results,
                            "completion_rate": research_query.total_results / research_query.max_results if research_query.max_results > 0 else 0
                        }
                    ))
                
                return search_results
                
        except Exception as e:
            self.logger.error(f"Research search failed: {e}")
            raise SearchError(f"Research search failed: {e}")
    
    async def search_knowledge(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[SearchResult]:
        """
        Search knowledge base entries.
        
        Args:
            query: Search query string
            filters: Knowledge-specific filters
            limit: Maximum results to return
            offset: Offset for pagination
            
        Returns:
            List of search results
        """
        try:
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_session() as session:
                search_query = select(KnowledgeBase)
                
                if query:
                    search_conditions = [
                        KnowledgeBase.title.ilike(f"%{query}%"),
                        KnowledgeBase.content.ilike(f"%{query}%"),
                        KnowledgeBase.summary.ilike(f"%{query}%")
                    ]
                    search_query = search_query.where(or_(*search_conditions))
                
                # Apply filters
                if filters:
                    search_query = await self._apply_knowledge_filters(search_query, filters)
                
                search_query = search_query.order_by(desc(KnowledgeBase.confidence_score))
                search_query = search_query.limit(limit).offset(offset)
                
                result = await session.execute(search_query)
                knowledge_entries = result.scalars().all()
                
                # Create search results
                search_results = []
                for knowledge in knowledge_entries:
                    score = await self._calculate_knowledge_score(knowledge, query)
                    highlights = await self._generate_knowledge_highlights(knowledge, query)
                    
                    search_results.append(SearchResult(
                        item=knowledge,
                        item_type="knowledge",
                        score=score,
                        highlights=highlights,
                        metadata={
                            "category": knowledge.category,
                            "confidence": knowledge.confidence_score,
                            "validation_status": knowledge.validation_status,
                            "access_count": knowledge.access_count
                        }
                    ))
                
                return search_results
                
        except Exception as e:
            self.logger.error(f"Knowledge search failed: {e}")
            raise SearchError(f"Knowledge search failed: {e}")
    
    async def suggest_searches(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Generate search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum suggestions to return
            
        Returns:
            List of suggested search queries
        """
        try:
            suggestions = []
            
            # Get suggestions from different sources
            email_suggestions = await self._get_email_suggestions(partial_query, limit // 2)
            research_suggestions = await self._get_research_suggestions(partial_query, limit // 2)
            
            suggestions.extend(email_suggestions)
            suggestions.extend(research_suggestions)
            
            # Remove duplicates and sort by relevance
            unique_suggestions = list(set(suggestions))
            unique_suggestions.sort(key=lambda x: len(x))
            
            return unique_suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Search suggestions failed: {e}")
            return []
    
    async def get_search_history(self, user_id: UUID, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get search history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum history items to return
            
        Returns:
            List of search history items
        """
        try:
            # This would query a search_history table
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get search history: {e}")
            return []
    
    async def save_search(
        self,
        user_id: UUID,
        name: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a search for later use.
        
        Args:
            user_id: User ID
            name: Saved search name
            query: Search query
            filters: Search filters
            
        Returns:
            True if saved successfully
        """
        try:
            # This would save to a saved_searches table
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save search: {e}")
            return False
    
    async def get_saved_searches(self, user_id: UUID) -> List[Dict[str, Any]]:
        """
        Get saved searches for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of saved searches
        """
        try:
            # This would query a saved_searches table
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get saved searches: {e}")
            return []
    
    # Private helper methods
    
    async def _search_by_type(
        self,
        model_class: type,
        search_type: str,
        query: str,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[SearchResult]:
        """Search within a specific data type."""
        if search_type == "emails":
            return await self.search_emails(query, filters, limit)
        elif search_type == "research":
            return await self.search_research(query, filters, limit)
        elif search_type == "knowledge":
            return await self.search_knowledge(query, filters, limit)
        else:
            return []
    
    async def _apply_email_filters(self, query, filters: Dict[str, Any]):
        """Apply email-specific filters to query."""
        if "status" in filters:
            query = query.where(Email.status == filters["status"])
        
        if "priority" in filters:
            query = query.where(Email.priority == filters["priority"])
        
        if "category" in filters:
            query = query.where(Email.category == filters["category"])
        
        if "sender" in filters:
            query = query.where(Email.sender_email.ilike(f"%{filters['sender']}%"))
        
        if "date_from" in filters:
            query = query.where(Email.received_at >= filters["date_from"])
        
        if "date_to" in filters:
            query = query.where(Email.received_at <= filters["date_to"])
        
        return query
    
    async def _apply_research_filters(self, query, filters: Dict[str, Any]):
        """Apply research-specific filters to query."""
        if "research_type" in filters:
            query = query.where(ResearchQuery.research_type == filters["research_type"])
        
        if "status" in filters:
            query = query.where(ResearchQuery.status == filters["status"])
        
        if "priority" in filters:
            query = query.where(ResearchQuery.priority == filters["priority"])
        
        return query
    
    async def _apply_knowledge_filters(self, query, filters: Dict[str, Any]):
        """Apply knowledge-specific filters to query."""
        if "category" in filters:
            query = query.where(KnowledgeBase.category == filters["category"])
        
        if "min_confidence" in filters:
            query = query.where(KnowledgeBase.confidence_score >= filters["min_confidence"])
        
        if "validation_status" in filters:
            query = query.where(KnowledgeBase.validation_status == filters["validation_status"])
        
        return query
    
    async def _calculate_email_score(self, email: Email, query: str) -> float:
        """Calculate relevance score for email."""
        score = 0.0
        query_lower = query.lower()
        
        # Subject match
        if query_lower in email.subject.lower():
            score += 0.4
        
        # Body match
        if email.body_text and query_lower in email.body_text.lower():
            score += 0.3
        
        # Sender match
        if query_lower in email.sender_email.lower() or query_lower in email.sender_name.lower():
            score += 0.2
        
        # Priority boost
        if email.priority == EmailPriority.HIGH:
            score += 0.1
        elif email.priority == EmailPriority.URGENT:
            score += 0.2
        
        # Recent emails get slight boost
        days_old = (datetime.now() - email.received_at).days
        if days_old < 7:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _calculate_research_score(self, research: ResearchQuery, query: str) -> float:
        """Calculate relevance score for research."""
        score = 0.0
        query_lower = query.lower()
        
        # Query text match
        if query_lower in research.query_text.lower():
            score += 0.5
        
        # Summary match
        if research.summary and query_lower in research.summary.lower():
            score += 0.3
        
        # Relevance boost
        if research.average_relevance:
            score += research.average_relevance * 0.2
        
        return min(score, 1.0)
    
    async def _calculate_knowledge_score(self, knowledge: KnowledgeBase, query: str) -> float:
        """Calculate relevance score for knowledge."""
        score = 0.0
        query_lower = query.lower()
        
        # Title match
        if query_lower in knowledge.title.lower():
            score += 0.4
        
        # Content match
        if query_lower in knowledge.content.lower():
            score += 0.3
        
        # Confidence boost
        score += knowledge.confidence_score * 0.3
        
        return min(score, 1.0)
    
    async def _generate_email_highlights(self, email: Email, query: str) -> List[str]:
        """Generate highlighted text snippets for email."""
        highlights = []
        query_lower = query.lower()
        
        # Subject highlight
        if query_lower in email.subject.lower():
            highlights.append(f"Subject: ...{email.subject}...")
        
        # Body highlight
        if email.body_text and query_lower in email.body_text.lower():
            # Find query position and extract surrounding text
            body_lower = email.body_text.lower()
            pos = body_lower.find(query_lower)
            if pos != -1:
                start = max(0, pos - 100)
                end = min(len(email.body_text), pos + 100)
                snippet = email.body_text[start:end]
                highlights.append(f"Body: ...{snippet}...")
        
        return highlights
    
    async def _generate_research_highlights(self, research: ResearchQuery, query: str) -> List[str]:
        """Generate highlighted text snippets for research."""
        highlights = []
        query_lower = query.lower()
        
        if query_lower in research.query_text.lower():
            highlights.append(f"Query: {research.query_text}")
        
        if research.summary and query_lower in research.summary.lower():
            highlights.append(f"Summary: {research.summary[:200]}...")
        
        return highlights
    
    async def _generate_knowledge_highlights(self, knowledge: KnowledgeBase, query: str) -> List[str]:
        """Generate highlighted text snippets for knowledge."""
        highlights = []
        query_lower = query.lower()
        
        if query_lower in knowledge.title.lower():
            highlights.append(f"Title: {knowledge.title}")
        
        if query_lower in knowledge.content.lower():
            # Extract snippet around query
            content_lower = knowledge.content.lower()
            pos = content_lower.find(query_lower)
            if pos != -1:
                start = max(0, pos - 100)
                end = min(len(knowledge.content), pos + 100)
                snippet = knowledge.content[start:end]
                highlights.append(f"Content: ...{snippet}...")
        
        return highlights
    
    async def _generate_facets(self, results: List[SearchResult]) -> List[SearchFacet]:
        """Generate search facets from results."""
        facets = []
        
        # Type facet
        type_counts = {}
        for result in results:
            type_counts[result.item_type] = type_counts.get(result.item_type, 0) + 1
        
        if type_counts:
            facets.append(SearchFacet("type", list(type_counts.items())))
        
        # Add more facets based on result types
        # This would be expanded based on specific needs
        
        return facets
    
    async def _get_email_suggestions(self, partial_query: str, limit: int) -> List[str]:
        """Get email-based search suggestions."""
        try:
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_session() as session:
                # Get common subjects and senders
                subject_query = select(
                    Email.subject,
                    func.count(Email.id).label("count")
                ).where(
                    Email.subject.ilike(f"%{partial_query}%")
                ).group_by(Email.subject).order_by(desc("count")).limit(limit)
                
                result = await session.execute(subject_query)
                subjects = [row.subject for row in result]
                
                return subjects
                
        except Exception as e:
            self.logger.error(f"Failed to get email suggestions: {e}")
            return []
    
    async def _get_research_suggestions(self, partial_query: str, limit: int) -> List[str]:
        """Get research-based search suggestions."""
        try:
            db_manager = await self._get_db_manager()
            
            async with db_manager.get_session() as session:
                # Get common research queries
                query_query = select(
                    ResearchQuery.query_text,
                    func.count(ResearchQuery.id).label("count")
                ).where(
                    ResearchQuery.query_text.ilike(f"%{partial_query}%")
                ).group_by(ResearchQuery.query_text).order_by(desc("count")).limit(limit)
                
                result = await session.execute(query_query)
                queries = [row.query_text for row in result]
                
                return queries
                
        except Exception as e:
            self.logger.error(f"Failed to get research suggestions: {e}")
            return []
    
    async def _record_search_analytics(
        self,
        query: str,
        result_count: int,
        filters: Optional[Dict[str, Any]]
    ) -> None:
        """Record search analytics."""
        try:
            # This would record to a search_analytics table
            self.logger.info(f"Search analytics: query='{query}', results={result_count}")
            
        except Exception as e:
            self.logger.error(f"Failed to record search analytics: {e}")
    
    async def get_search_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get search analytics for the specified period."""
        try:
            # This would query search_analytics table
            return {
                "total_searches": 0,
                "average_results": 0,
                "top_queries": [],
                "search_volume_trend": []
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get search analytics: {e}")
            return {}