"""
Comprehensive test suite for the ResearchAgent class.
Tests all research functionality following TDD principles.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.agents.base import AgentMessage, AgentState
from src.agents.research import (
    ResearchAgent,
    ResearchTask,
    ResearchResult,
    ContentItem,
    ResearchReport,
    ResearchQuery,
    FeedMonitor,
    ContentExtractor,
    ContentScorer,
    ResearchCache,
    RobotsTxtChecker,
    ResearchException,
    RateLimitError,
    ContentExtractionError,
    CacheError,
)


# Module-level fixtures
@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return MagicMock()


@pytest.fixture
def mock_message_broker():
    """Mock message broker for testing."""
    mock_broker = MagicMock()
    mock_broker.publish = AsyncMock()
    mock_broker.disconnect = AsyncMock()
    return mock_broker


@pytest.fixture
def research_config():
    """Configuration for research agent."""
    return {
        "max_concurrent_requests": 5,
        "request_timeout": 30,
        "rate_limit_delay": 1.0,
        "cache_ttl": 3600,
        "user_agent": "AutonomousAgent/1.0 Research Bot",
        "respect_robots_txt": True,
        "max_content_length": 1024 * 1024,  # 1MB
        "content_extraction_timeout": 10,
        "relevance_threshold": 0.6,
        "deduplication_threshold": 0.8,
        "max_feed_items": 100,
        "feed_check_interval": 300,  # 5 minutes
        "research_topics": ["technology", "ai", "automation"],
        "blocked_domains": ["example.com", "spam.com"],
        "allowed_file_types": ["text/html", "application/rss+xml", "application/xml"],
    }


@pytest.fixture
def research_agent(research_config, mock_logger, mock_message_broker):
    """Create a ResearchAgent instance for testing."""
    return ResearchAgent(
        agent_id="research_agent_001",
        config=research_config,
        logger=mock_logger,
        message_broker=mock_message_broker,
    )


@pytest.fixture
def sample_research_task():
    """Sample research task for testing."""
    return ResearchTask(
        id="task_001",
        query="artificial intelligence trends",
        sources=["https://example.com/ai-news", "https://feeds.example.com/ai.rss"],
        max_results=50,
        priority=1,
        deadline=datetime.now(timezone.utc).timestamp() + 3600,
    )


@pytest.fixture
def sample_content_item():
    """Sample content item for testing."""
    return ContentItem(
        id="content_001",
        url="https://example.com/article",
        title="AI Trends in 2024",
        content="This article discusses the latest trends in artificial intelligence...",
        author="John Doe",
        published_date=datetime.now(timezone.utc),
        source_type="web",
        tags=["ai", "technology", "trends"],
        relevance_score=0.85,
        summary="Article about AI trends in 2024",
    )


class TestResearchAgent:
    """Test suite for ResearchAgent class."""

    @pytest.mark.asyncio
    async def test_research_agent_initialization(self, research_agent, research_config):
        """Test ResearchAgent initialization."""
        assert research_agent.agent_id == "research_agent_001"
        assert research_agent.config == research_config
        assert research_agent.state == AgentState.INACTIVE
        assert research_agent.max_concurrent_requests == 5
        assert research_agent.rate_limit_delay == 1.0
        assert research_agent.cache_ttl == 3600
        assert research_agent.respect_robots_txt is True

    @pytest.mark.asyncio
    async def test_research_agent_start_stop(self, research_agent):
        """Test starting and stopping the research agent."""
        # Test start
        await research_agent.start()
        assert research_agent.state == AgentState.ACTIVE
        assert research_agent.start_time is not None
        assert research_agent._feed_monitor is not None
        assert research_agent._content_extractor is not None
        assert research_agent._content_scorer is not None
        assert research_agent._research_cache is not None
        assert research_agent._robots_checker is not None

        # Test stop
        await research_agent.stop()
        assert research_agent.state == AgentState.INACTIVE

    @pytest.mark.asyncio
    async def test_research_agent_health_check(self, research_agent):
        """Test health check functionality."""
        await research_agent.start()
        
        # Mock successful health check
        with patch.object(research_agent._research_cache, 'health_check', return_value=True):
            health_status = await research_agent.health_check()
            assert health_status is True

        # Mock failed health check
        with patch.object(research_agent._research_cache, 'health_check', return_value=False):
            health_status = await research_agent.health_check()
            assert health_status is False

        await research_agent.stop()

    @pytest.mark.asyncio
    async def test_research_task_execution(self, research_agent, sample_research_task):
        """Test executing a research task."""
        await research_agent.start()

        # Mock successful research execution
        mock_results = [
            {
                "id": "result_001",
                "url": "https://example.com/article1",
                "title": "AI Trends 2024",
                "content": "Article content...",
                "relevance_score": 0.9,
            }
        ]

        with patch.object(research_agent, '_execute_research_task', return_value=mock_results):
            result = await research_agent.execute_task(sample_research_task.to_dict())
            
            assert result["status"] == "completed"
            assert result["results_count"] == 1
            assert len(result["results"]) == 1
            assert result["results"][0]["relevance_score"] == 0.9

        await research_agent.stop()

    @pytest.mark.asyncio
    async def test_message_handling(self, research_agent):
        """Test message handling functionality."""
        await research_agent.start()

        # Test research request message
        research_message = AgentMessage(
            id="msg_001",
            sender="test_sender",
            recipient="research_agent_001",
            message_type="research_request",
            payload={
                "query": "machine learning",
                "sources": ["https://example.com/ml"],
                "max_results": 10,
            },
        )

        with patch.object(research_agent, '_handle_research_request', return_value=None) as mock_handler:
            await research_agent.handle_message(research_message)
            mock_handler.assert_called_once()

        await research_agent.stop()

    @pytest.mark.asyncio
    async def test_metrics_collection(self, research_agent):
        """Test metrics collection."""
        await research_agent.start()

        # Simulate some research activity
        research_agent.metrics["research_tasks_completed"] = 5
        research_agent.metrics["content_items_processed"] = 100
        research_agent.metrics["cache_hits"] = 25
        research_agent.metrics["cache_misses"] = 10

        metrics = research_agent.get_metrics()
        
        assert metrics["research_tasks_completed"] == 5
        assert metrics["content_items_processed"] == 100
        assert metrics["cache_hits"] == 25
        assert metrics["cache_misses"] == 10
        assert "uptime" in metrics
        assert metrics["state"] == AgentState.ACTIVE.value

        await research_agent.stop()


class TestResearchTask:
    """Test suite for ResearchTask class."""

    def test_research_task_creation(self):
        """Test creating a research task."""
        task = ResearchTask(
            id="task_001",
            query="python programming",
            sources=["https://example.com/python"],
            max_results=20,
            priority=2,
            deadline=datetime.now(timezone.utc).timestamp() + 1800,
        )
        
        assert task.id == "task_001"
        assert task.query == "python programming"
        assert len(task.sources) == 1
        assert task.max_results == 20
        assert task.priority == 2
        assert task.deadline is not None

    def test_research_task_serialization(self):
        """Test serializing and deserializing research tasks."""
        task = ResearchTask(
            id="task_002",
            query="web scraping",
            sources=["https://example.com/scraping"],
            max_results=30,
        )
        
        task_dict = task.to_dict()
        assert task_dict["id"] == "task_002"
        assert task_dict["query"] == "web scraping"
        assert isinstance(task_dict["sources"], list)
        
        restored_task = ResearchTask.from_dict(task_dict)
        assert restored_task.id == task.id
        assert restored_task.query == task.query
        assert restored_task.sources == task.sources

    def test_research_task_validation(self):
        """Test research task validation."""
        # Test invalid task (empty query)
        with pytest.raises(ValueError):
            ResearchTask(
                id="task_003",
                query="",
                sources=["https://example.com"],
                max_results=10,
            )
        
        # Test invalid task (no sources)
        with pytest.raises(ValueError):
            ResearchTask(
                id="task_004",
                query="valid query",
                sources=[],
                max_results=10,
            )


class TestContentItem:
    """Test suite for ContentItem class."""

    def test_content_item_creation(self, sample_content_item):
        """Test creating a content item."""
        assert sample_content_item.id == "content_001"
        assert sample_content_item.url == "https://example.com/article"
        assert sample_content_item.title == "AI Trends in 2024"
        assert sample_content_item.relevance_score == 0.85
        assert "ai" in sample_content_item.tags

    def test_content_item_serialization(self, sample_content_item):
        """Test serializing and deserializing content items."""
        item_dict = sample_content_item.to_dict()
        assert item_dict["id"] == "content_001"
        assert item_dict["url"] == "https://example.com/article"
        assert item_dict["relevance_score"] == 0.85
        
        restored_item = ContentItem.from_dict(item_dict)
        assert restored_item.id == sample_content_item.id
        assert restored_item.url == sample_content_item.url
        assert restored_item.relevance_score == sample_content_item.relevance_score

    def test_content_item_duplicate_detection(self, sample_content_item):
        """Test duplicate detection logic."""
        # Create similar content item
        similar_item = ContentItem(
            id="content_002",
            url="https://example.com/article-duplicate",
            title="AI Trends in 2024 - Updated",
            content="This article discusses the latest trends in artificial intelligence...",
            author="Jane Smith",
            published_date=datetime.now(timezone.utc),
            source_type="web",
            tags=["ai", "technology", "trends"],
            relevance_score=0.80,
        )
        
        # Test similarity calculation
        similarity = sample_content_item.calculate_similarity(similar_item)
        assert similarity > 0.7  # Should be similar due to content overlap
        
        # Test duplicate detection
        is_duplicate = sample_content_item.is_duplicate(similar_item, threshold=0.8)
        assert is_duplicate is True


class TestContentExtractor:
    """Test suite for ContentExtractor class."""

    @pytest.fixture
    def content_extractor(self):
        """Create a ContentExtractor instance."""
        config = {
            "timeout": 10,
            "max_content_length": 1024 * 1024,
            "user_agent": "TestBot/1.0",
        }
        return ContentExtractor(config)

    @pytest.mark.asyncio
    async def test_web_content_extraction(self, content_extractor):
        """Test extracting content from web pages."""
        mock_html = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <h1>Test Article Title</h1>
                <p>This is the main content of the article.</p>
                <p>Additional content paragraph.</p>
            </body>
        </html>
        """
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_html
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_get.return_value = mock_response
            
            content = await content_extractor.extract_web_content("https://example.com/article")
            
            assert content.title == "Test Article Title"
            assert "main content" in content.content
            assert content.url == "https://example.com/article"
            assert content.source_type == "web"

    @pytest.mark.asyncio
    async def test_rss_content_extraction(self, content_extractor):
        """Test extracting content from RSS feeds."""
        mock_rss = """
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                    <description>This is the article description.</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_rss
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'application/rss+xml'}
            mock_get.return_value = mock_response
            
            items = await content_extractor.extract_rss_content("https://example.com/feed.rss")
            
            assert len(items) == 1
            assert items[0].title == "Test Article"
            assert items[0].url == "https://example.com/article"
            assert items[0].source_type == "rss"

    @pytest.mark.asyncio
    async def test_content_extraction_error_handling(self, content_extractor):
        """Test error handling in content extraction."""
        # Test timeout error
        with patch('httpx.AsyncClient.get', side_effect=asyncio.TimeoutError()):
            with pytest.raises(ContentExtractionError):
                await content_extractor.extract_web_content("https://example.com/timeout")
        
        # Test HTTP error
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response
            
            with pytest.raises(ContentExtractionError):
                await content_extractor.extract_web_content("https://example.com/notfound")


class TestContentScorer:
    """Test suite for ContentScorer class."""

    @pytest.fixture
    def content_scorer(self):
        """Create a ContentScorer instance."""
        config = {
            "relevance_threshold": 0.6,
            "scoring_algorithm": "tfidf",
        }
        return ContentScorer(config)

    def test_relevance_scoring(self, content_scorer, sample_content_item):
        """Test relevance scoring functionality."""
        query = "artificial intelligence machine learning"
        
        # Test scoring with relevant content
        score = content_scorer.calculate_relevance_score(sample_content_item, query)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be relevant

        # Test scoring with irrelevant content
        irrelevant_item = ContentItem(
            id="content_002",
            url="https://example.com/cooking",
            title="Best Cooking Recipes",
            content="This article discusses cooking techniques and recipes...",
            author="Chef Smith",
            published_date=datetime.now(timezone.utc),
            source_type="web",
            tags=["cooking", "recipes", "food"],
            relevance_score=0.0,
        )
        
        score = content_scorer.calculate_relevance_score(irrelevant_item, query)
        assert score < 0.3  # Should be irrelevant

    def test_content_categorization(self, content_scorer, sample_content_item):
        """Test content categorization."""
        categories = content_scorer.categorize_content(sample_content_item)
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "technology" in categories or "ai" in categories

    def test_keyword_extraction(self, content_scorer, sample_content_item):
        """Test keyword extraction from content."""
        keywords = content_scorer.extract_keywords(sample_content_item)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert any("ai" in keyword.lower() or "artificial" in keyword.lower() for keyword in keywords)


class TestResearchCache:
    """Test suite for ResearchCache class."""

    @pytest.fixture
    def research_cache(self):
        """Create a ResearchCache instance."""
        config = {
            "cache_type": "memory",
            "ttl": 3600,
            "max_size": 1000,
        }
        return ResearchCache(config)

    @pytest.mark.asyncio
    async def test_cache_operations(self, research_cache, sample_content_item):
        """Test basic cache operations."""
        # Test set and get
        await research_cache.set("key1", sample_content_item)
        cached_item = await research_cache.get("key1")
        
        assert cached_item is not None
        assert cached_item.id == sample_content_item.id
        assert cached_item.title == sample_content_item.title

        # Test cache miss
        missing_item = await research_cache.get("nonexistent_key")
        assert missing_item is None

    @pytest.mark.asyncio
    async def test_cache_expiration(self, research_cache, sample_content_item):
        """Test cache expiration functionality."""
        # Set item with short TTL
        await research_cache.set("expiring_key", sample_content_item, ttl=1)
        
        # Should be available immediately
        cached_item = await research_cache.get("expiring_key")
        assert cached_item is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be expired
        expired_item = await research_cache.get("expiring_key")
        assert expired_item is None

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, research_cache, sample_content_item):
        """Test cache invalidation."""
        # Set item
        await research_cache.set("key_to_invalidate", sample_content_item)
        
        # Verify it's cached
        cached_item = await research_cache.get("key_to_invalidate")
        assert cached_item is not None
        
        # Invalidate
        await research_cache.invalidate("key_to_invalidate")
        
        # Should be gone
        invalidated_item = await research_cache.get("key_to_invalidate")
        assert invalidated_item is None

    @pytest.mark.asyncio
    async def test_cache_statistics(self, research_cache, sample_content_item):
        """Test cache statistics tracking."""
        # Generate some cache activity
        await research_cache.get("miss_key")  # Cache miss
        await research_cache.set("hit_key", sample_content_item)
        await research_cache.get("hit_key")  # Cache hit
        
        stats = await research_cache.get_statistics()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["total_items"] >= 1


class TestRobotsTxtChecker:
    """Test suite for RobotsTxtChecker class."""

    @pytest.fixture
    def robots_checker(self):
        """Create a RobotsTxtChecker instance."""
        config = {
            "user_agent": "TestBot/1.0",
            "cache_ttl": 3600,
        }
        return RobotsTxtChecker(config)

    @pytest.mark.asyncio
    async def test_robots_txt_compliance(self, robots_checker):
        """Test robots.txt compliance checking."""
        mock_robots_txt = """
        User-agent: *
        Disallow: /private/
        Disallow: /admin/
        Allow: /public/
        
        User-agent: TestBot
        Disallow: /restricted/
        """
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_robots_txt
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Test allowed URL
            is_allowed = await robots_checker.is_allowed("https://example.com/public/page")
            assert is_allowed is True
            
            # Test disallowed URL
            is_allowed = await robots_checker.is_allowed("https://example.com/private/page")
            assert is_allowed is False
            
            # Test bot-specific restriction
            is_allowed = await robots_checker.is_allowed("https://example.com/restricted/page")
            assert is_allowed is False

    @pytest.mark.asyncio
    async def test_robots_txt_error_handling(self, robots_checker):
        """Test error handling for robots.txt."""
        # Test when robots.txt is not found (should allow)
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response
            
            is_allowed = await robots_checker.is_allowed("https://example.com/any/page")
            assert is_allowed is True  # Default to allow when robots.txt not found

    @pytest.mark.asyncio
    async def test_robots_txt_caching(self, robots_checker):
        """Test robots.txt caching."""
        mock_robots_txt = "User-agent: *\nDisallow: /private/"
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_robots_txt
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # First call should fetch robots.txt
            await robots_checker.is_allowed("https://example.com/public/page")
            
            # Second call should use cache
            await robots_checker.is_allowed("https://example.com/another/page")
            
            # Should only call HTTP get once due to caching
            assert mock_get.call_count == 1


class TestFeedMonitor:
    """Test suite for FeedMonitor class."""

    @pytest.fixture
    def feed_monitor(self):
        """Create a FeedMonitor instance."""
        config = {
            "check_interval": 300,
            "max_items": 100,
            "concurrent_feeds": 10,
        }
        return FeedMonitor(config)

    @pytest.mark.asyncio
    async def test_feed_monitoring(self, feed_monitor):
        """Test RSS feed monitoring."""
        mock_rss = """
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>New Article</title>
                    <link>https://example.com/new-article</link>
                    <description>New article content</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_rss
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Add feed to monitor
            await feed_monitor.add_feed("https://example.com/feed.rss")
            
            # Check for new items
            new_items = await feed_monitor.check_feeds()
            
            assert len(new_items) == 1
            assert new_items[0].title == "New Article"
            assert new_items[0].url == "https://example.com/new-article"

    @pytest.mark.asyncio
    async def test_feed_deduplication(self, feed_monitor):
        """Test feed item deduplication."""
        # Mock feed with duplicate items
        mock_rss = """
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Article 1</title>
                    <link>https://example.com/article1</link>
                    <description>Article 1 content</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Article 1</title>
                    <link>https://example.com/article1</link>
                    <description>Article 1 content</description>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_rss
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            await feed_monitor.add_feed("https://example.com/feed.rss")
            new_items = await feed_monitor.check_feeds()
            
            # Should deduplicate identical items
            assert len(new_items) == 1

    @pytest.mark.asyncio
    async def test_feed_error_handling(self, feed_monitor):
        """Test error handling in feed monitoring."""
        # Test feed fetch error
        with patch('httpx.AsyncClient.get', side_effect=asyncio.TimeoutError()):
            await feed_monitor.add_feed("https://example.com/timeout-feed.rss")
            
            # Should handle errors gracefully
            new_items = await feed_monitor.check_feeds()
            assert isinstance(new_items, list)  # Should return empty list on error


class TestResearchReport:
    """Test suite for ResearchReport class."""

    @pytest.fixture
    def sample_research_results(self, sample_content_item):
        """Sample research results for testing."""
        return [
            sample_content_item,
            ContentItem(
                id="content_002",
                url="https://example.com/article2",
                title="Machine Learning Advances",
                content="Article about recent ML advances...",
                author="Jane Smith",
                published_date=datetime.now(timezone.utc),
                source_type="web",
                tags=["ml", "technology", "research"],
                relevance_score=0.78,
            ),
        ]

    def test_research_report_generation(self, sample_research_results):
        """Test research report generation."""
        report = ResearchReport(
            id="report_001",
            query="artificial intelligence",
            results=sample_research_results,
            generated_at=datetime.now(timezone.utc),
            total_sources=5,
            processing_time=12.5,
        )
        
        assert report.id == "report_001"
        assert report.query == "artificial intelligence"
        assert len(report.results) == 2
        assert report.total_sources == 5
        assert report.processing_time == 12.5

    def test_report_summary_generation(self, sample_research_results):
        """Test generating report summary."""
        report = ResearchReport(
            id="report_002",
            query="machine learning",
            results=sample_research_results,
            generated_at=datetime.now(timezone.utc),
            total_sources=3,
            processing_time=8.2,
        )
        
        summary = report.generate_summary()
        
        assert isinstance(summary, dict)
        assert "total_results" in summary
        assert "average_relevance" in summary
        assert "top_keywords" in summary
        assert "source_distribution" in summary
        assert summary["total_results"] == 2

    def test_report_serialization(self, sample_research_results):
        """Test report serialization."""
        report = ResearchReport(
            id="report_003",
            query="ai research",
            results=sample_research_results,
            generated_at=datetime.now(timezone.utc),
            total_sources=4,
            processing_time=15.7,
        )
        
        report_dict = report.to_dict()
        
        assert report_dict["id"] == "report_003"
        assert report_dict["query"] == "ai research"
        assert len(report_dict["results"]) == 2
        assert report_dict["total_sources"] == 4
        
        # Test deserialization
        restored_report = ResearchReport.from_dict(report_dict)
        assert restored_report.id == report.id
        assert restored_report.query == report.query
        assert len(restored_report.results) == len(report.results)


class TestResearchQuery:
    """Test suite for ResearchQuery class."""

    def test_query_optimization(self):
        """Test query optimization functionality."""
        original_query = "machine learning artificial intelligence"
        
        query = ResearchQuery(original_query)
        optimized = query.optimize()
        
        assert isinstance(optimized, str)
        assert len(optimized) > 0
        # Should contain relevant terms
        assert "machine learning" in optimized.lower() or "ml" in optimized.lower()

    def test_query_expansion(self):
        """Test query expansion with synonyms."""
        query = ResearchQuery("AI")
        expanded = query.expand()
        
        assert isinstance(expanded, list)
        assert len(expanded) > 1
        # Should include synonyms
        assert any("artificial intelligence" in term.lower() for term in expanded)

    def test_query_topic_extraction(self):
        """Test topic extraction from queries."""
        query = ResearchQuery("deep learning neural networks computer vision")
        topics = query.extract_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        # Should identify relevant topics
        assert any("deep learning" in topic.lower() for topic in topics)


class TestResearchExceptions:
    """Test suite for Research-specific exceptions."""

    def test_research_exception(self):
        """Test ResearchException."""
        with pytest.raises(ResearchException) as exc_info:
            raise ResearchException("Test research error")
        
        assert "Test research error" in str(exc_info.value)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError("Rate limit exceeded", retry_after=60)
        
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_content_extraction_error(self):
        """Test ContentExtractionError."""
        with pytest.raises(ContentExtractionError) as exc_info:
            raise ContentExtractionError("Failed to extract content", url="https://example.com")
        
        assert "Failed to extract content" in str(exc_info.value)

    def test_cache_error(self):
        """Test CacheError."""
        with pytest.raises(CacheError) as exc_info:
            raise CacheError("Cache operation failed", operation="get")
        
        assert "Cache operation failed" in str(exc_info.value)


# Integration tests
class TestResearchAgentIntegration:
    """Integration tests for ResearchAgent."""

    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self, research_agent, sample_research_task):
        """Test complete research workflow from task to report."""
        await research_agent.start()
        
        # Mock all external dependencies
        with patch('httpx.AsyncClient.get') as mock_get:
            # Mock web content
            mock_response = MagicMock()
            mock_response.text = "<html><head><title>Test</title></head><body><p>Content</p></body></html>"
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_get.return_value = mock_response
            
            # Execute research task
            result = await research_agent.execute_task(sample_research_task.to_dict())
            
            assert result["status"] == "completed"
            assert "results" in result
            assert "report" in result

        await research_agent.stop()

    @pytest.mark.asyncio
    async def test_concurrent_research_tasks(self, research_agent):
        """Test handling multiple concurrent research tasks."""
        await research_agent.start()
        
        # Create multiple research tasks
        tasks = [
            ResearchTask(
                id=f"task_{i}",
                query=f"query {i}",
                sources=[f"https://example.com/source{i}"],
                max_results=10,
            )
            for i in range(3)
        ]
        
        # Mock responses
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = "<html><body><p>Test content</p></body></html>"
            mock_response.status_code = 200
            mock_response.headers = {'content-type': 'text/html'}
            mock_get.return_value = mock_response
            
            # Execute tasks concurrently
            results = await asyncio.gather(*[
                research_agent.execute_task(task.to_dict())
                for task in tasks
            ])
            
            assert len(results) == 3
            assert all(result["status"] == "completed" for result in results)

        await research_agent.stop()

    @pytest.mark.asyncio
    async def test_research_agent_communication(self, research_agent):
        """Test communication between research agent and other agents."""
        await research_agent.start()
        
        # Test receiving research request
        message = AgentMessage(
            id="msg_001",
            sender="coordinator",
            recipient="research_agent_001",
            message_type="research_request",
            payload={
                "query": "test query",
                "sources": ["https://example.com"],
                "max_results": 5,
            },
        )
        
        with patch.object(research_agent, 'send_message') as mock_send:
            await research_agent.handle_message(message)
            
            # Should send back results
            mock_send.assert_called()
            call_args = mock_send.call_args
            assert call_args[0][0] == "coordinator"  # recipient
            assert call_args[0][1] == "research_results"  # message_type

        await research_agent.stop()