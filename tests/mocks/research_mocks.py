"""
Comprehensive mocks for research agent testing.
Provides realistic mock responses for web requests, RSS feeds, and external APIs.
"""

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, AsyncMock
from urllib.parse import urlparse, urljoin


class MockHttpResponse:
    """Mock HTTP response for testing."""
    
    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
    ):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self._json_data = json_data
        self.content = content or text.encode('utf-8')
        self.url = "https://example.com"
        self.reason = "OK" if status_code == 200 else "Error"
    
    def json(self):
        """Return JSON data."""
        if self._json_data:
            return self._json_data
        return json.loads(self.text)
    
    def raise_for_status(self):
        """Raise exception for HTTP errors."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code} Error")


class MockWebScraper:
    """Mock web scraper for testing."""
    
    def __init__(self):
        self.request_count = 0
        self.rate_limit_calls = 0
        self.responses = {}
        self.default_response = self._create_default_html_response()
    
    def _create_default_html_response(self) -> str:
        """Create a default HTML response."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Default Test Page</title>
            <meta name="description" content="This is a test page for research agent testing.">
        </head>
        <body>
            <h1>Test Article Title</h1>
            <div class="content">
                <p>This is the main content of the test article. It contains relevant information about the topic.</p>
                <p>Additional paragraph with more details about the subject matter.</p>
            </div>
            <div class="metadata">
                <span class="author">By Test Author</span>
                <span class="date">2024-01-01</span>
            </div>
        </body>
        </html>
        """
    
    def add_response(self, url: str, response: MockHttpResponse):
        """Add a mock response for a specific URL."""
        self.responses[url] = response
    
    def add_html_response(self, url: str, html_content: str, status_code: int = 200):
        """Add an HTML response for a specific URL."""
        response = MockHttpResponse(
            status_code=status_code,
            text=html_content,
            headers={'content-type': 'text/html; charset=utf-8'}
        )
        self.add_response(url, response)
    
    def add_json_response(self, url: str, json_data: Dict[str, Any], status_code: int = 200):
        """Add a JSON response for a specific URL."""
        response = MockHttpResponse(
            status_code=status_code,
            text=json.dumps(json_data),
            headers={'content-type': 'application/json'},
            json_data=json_data
        )
        self.add_response(url, response)
    
    def add_error_response(self, url: str, status_code: int = 404, error_message: str = "Not Found"):
        """Add an error response for a specific URL."""
        response = MockHttpResponse(
            status_code=status_code,
            text=error_message,
            headers={'content-type': 'text/plain'}
        )
        self.add_response(url, response)
    
    async def get(self, url: str, **kwargs) -> MockHttpResponse:
        """Mock GET request."""
        self.request_count += 1
        
        # Simulate rate limiting
        if self.request_count > 100:
            self.rate_limit_calls += 1
            if self.rate_limit_calls > 5:
                raise Exception("Rate limit exceeded")
        
        # Simulate network delay
        await asyncio.sleep(0.01)
        
        # Return specific response if available
        if url in self.responses:
            return self.responses[url]
        
        # Return default response
        return MockHttpResponse(
            status_code=200,
            text=self.default_response,
            headers={'content-type': 'text/html; charset=utf-8'}
        )
    
    def reset(self):
        """Reset mock state."""
        self.request_count = 0
        self.rate_limit_calls = 0
        self.responses.clear()


class MockRSSFeed:
    """Mock RSS feed for testing."""
    
    def __init__(self, title: str = "Test Feed", description: str = "Test RSS Feed"):
        self.title = title
        self.description = description
        self.items = []
        self.last_updated = datetime.now(timezone.utc)
    
    def add_item(
        self,
        title: str,
        link: str,
        description: str,
        pub_date: Optional[datetime] = None,
        author: Optional[str] = None,
        category: Optional[str] = None,
        guid: Optional[str] = None,
    ):
        """Add an item to the RSS feed."""
        item = {
            'title': title,
            'link': link,
            'description': description,
            'pub_date': pub_date or datetime.now(timezone.utc),
            'author': author,
            'category': category,
            'guid': guid or link,
        }
        self.items.append(item)
    
    def generate_rss_xml(self) -> str:
        """Generate RSS XML content."""
        rss_items = []
        for item in self.items:
            pub_date_str = item['pub_date'].strftime('%a, %d %b %Y %H:%M:%S GMT')
            
            item_xml = f"""
            <item>
                <title><![CDATA[{item['title']}]]></title>
                <link>{item['link']}</link>
                <description><![CDATA[{item['description']}]]></description>
                <pubDate>{pub_date_str}</pubDate>
                <guid>{item['guid']}</guid>
            """
            
            if item['author']:
                item_xml += f"<author>{item['author']}</author>"
            
            if item['category']:
                item_xml += f"<category>{item['category']}</category>"
            
            item_xml += "</item>"
            rss_items.append(item_xml)
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title><![CDATA[{self.title}]]></title>
                <description><![CDATA[{self.description}]]></description>
                <link>https://example.com</link>
                <lastBuildDate>{self.last_updated.strftime('%a, %d %b %Y %H:%M:%S GMT')}</lastBuildDate>
                {''.join(rss_items)}
            </channel>
        </rss>
        """


class MockRSSFeedManager:
    """Mock RSS feed manager for testing."""
    
    def __init__(self):
        self.feeds = {}
        self.request_count = 0
    
    def add_feed(self, url: str, feed: MockRSSFeed):
        """Add a mock feed for a specific URL."""
        self.feeds[url] = feed
    
    def create_tech_news_feed(self, url: str):
        """Create a mock technology news feed."""
        feed = MockRSSFeed("Tech News", "Latest technology news and updates")
        
        # Add sample items
        feed.add_item(
            "AI Breakthrough in Natural Language Processing",
            "https://example.com/ai-breakthrough",
            "Researchers announce significant breakthrough in NLP technology.",
            datetime.now(timezone.utc),
            "Dr. Jane Smith",
            "Technology"
        )
        
        feed.add_item(
            "Machine Learning Advances in Healthcare",
            "https://example.com/ml-healthcare",
            "New ML algorithms showing promise in medical diagnosis.",
            datetime.now(timezone.utc),
            "Dr. John Doe",
            "Healthcare"
        )
        
        feed.add_item(
            "Quantum Computing Milestone Achieved",
            "https://example.com/quantum-computing",
            "Scientists achieve new quantum computing milestone.",
            datetime.now(timezone.utc),
            "Prof. Alice Johnson",
            "Quantum"
        )
        
        self.add_feed(url, feed)
    
    def create_research_feed(self, url: str):
        """Create a mock research feed."""
        feed = MockRSSFeed("Research Papers", "Latest research publications")
        
        feed.add_item(
            "Deep Learning Applications in Computer Vision",
            "https://example.com/research/dl-cv",
            "Comprehensive study on deep learning applications in computer vision.",
            datetime.now(timezone.utc),
            "Research Team",
            "Computer Vision"
        )
        
        feed.add_item(
            "Reinforcement Learning for Robotics",
            "https://example.com/research/rl-robotics",
            "Novel approaches to reinforcement learning in robotics applications.",
            datetime.now(timezone.utc),
            "Robotics Lab",
            "Robotics"
        )
        
        self.add_feed(url, feed)
    
    async def get_feed(self, url: str) -> MockHttpResponse:
        """Get RSS feed content."""
        self.request_count += 1
        
        # Simulate network delay
        await asyncio.sleep(0.01)
        
        if url in self.feeds:
            feed = self.feeds[url]
            return MockHttpResponse(
                status_code=200,
                text=feed.generate_rss_xml(),
                headers={'content-type': 'application/rss+xml; charset=utf-8'}
            )
        
        # Return 404 for unknown feeds
        return MockHttpResponse(
            status_code=404,
            text="Feed not found",
            headers={'content-type': 'text/plain'}
        )


class MockRobotsTxt:
    """Mock robots.txt handler for testing."""
    
    def __init__(self):
        self.robots_files = {}
        self.request_count = 0
    
    def add_robots_txt(self, domain: str, content: str):
        """Add robots.txt content for a domain."""
        self.robots_files[domain] = content
    
    def add_permissive_robots(self, domain: str):
        """Add permissive robots.txt for a domain."""
        content = """
        User-agent: *
        Allow: /
        
        Crawl-delay: 1
        """
        self.add_robots_txt(domain, content)
    
    def add_restrictive_robots(self, domain: str):
        """Add restrictive robots.txt for a domain."""
        content = """
        User-agent: *
        Disallow: /private/
        Disallow: /admin/
        Disallow: /api/
        
        User-agent: AutonomousAgent
        Disallow: /restricted/
        
        Crawl-delay: 5
        """
        self.add_robots_txt(domain, content)
    
    def add_blocked_robots(self, domain: str):
        """Add completely blocking robots.txt for a domain."""
        content = """
        User-agent: *
        Disallow: /
        """
        self.add_robots_txt(domain, content)
    
    async def get_robots_txt(self, url: str) -> MockHttpResponse:
        """Get robots.txt for a URL."""
        self.request_count += 1
        
        # Extract domain from URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Simulate network delay
        await asyncio.sleep(0.01)
        
        if domain in self.robots_files:
            return MockHttpResponse(
                status_code=200,
                text=self.robots_files[domain],
                headers={'content-type': 'text/plain'}
            )
        
        # Return 404 for unknown domains (which means allow all)
        return MockHttpResponse(
            status_code=404,
            text="Not found",
            headers={'content-type': 'text/plain'}
        )


class MockSearchAPI:
    """Mock search API for testing."""
    
    def __init__(self):
        self.search_results = {}
        self.request_count = 0
        self.rate_limit_calls = 0
    
    def add_search_results(self, query: str, results: List[Dict[str, Any]]):
        """Add mock search results for a query."""
        self.search_results[query.lower()] = results
    
    def create_default_results(self, query: str):
        """Create default search results for a query."""
        results = [
            {
                'title': f'Article about {query}',
                'url': f'https://example.com/article-{query.replace(" ", "-")}',
                'snippet': f'This article discusses {query} and related topics.',
                'date': '2024-01-01',
                'source': 'example.com'
            },
            {
                'title': f'Research on {query}',
                'url': f'https://research.example.com/{query.replace(" ", "-")}',
                'snippet': f'Comprehensive research study on {query}.',
                'date': '2024-01-02',
                'source': 'research.example.com'
            },
            {
                'title': f'{query} - Latest Developments',
                'url': f'https://news.example.com/{query.replace(" ", "-")}',
                'snippet': f'Latest news and developments in {query}.',
                'date': '2024-01-03',
                'source': 'news.example.com'
            }
        ]
        self.add_search_results(query, results)
    
    async def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform mock search."""
        self.request_count += 1
        
        # Simulate rate limiting
        if self.request_count > 1000:
            self.rate_limit_calls += 1
            if self.rate_limit_calls > 10:
                raise Exception("Search API rate limit exceeded")
        
        # Simulate network delay
        await asyncio.sleep(0.05)
        
        query_lower = query.lower()
        if query_lower in self.search_results:
            results = self.search_results[query_lower][:max_results]
        else:
            # Create default results if not found
            self.create_default_results(query)
            results = self.search_results[query_lower][:max_results]
        
        return {
            'query': query,
            'total_results': len(results),
            'results': results,
            'search_time': 0.05
        }


class MockContentDatabase:
    """Mock content database for testing."""
    
    def __init__(self):
        self.content_items = {}
        self.research_reports = {}
        self.cache_data = {}
        self.operation_count = 0
    
    async def store_content_item(self, item: Dict[str, Any]) -> str:
        """Store a content item."""
        self.operation_count += 1
        item_id = item.get('id', f"item_{self.operation_count}")
        self.content_items[item_id] = item
        return item_id
    
    async def get_content_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a content item by ID."""
        self.operation_count += 1
        return self.content_items.get(item_id)
    
    async def search_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search content items."""
        self.operation_count += 1
        
        # Simple text search simulation
        results = []
        for item in self.content_items.values():
            if query.lower() in item.get('title', '').lower() or \
               query.lower() in item.get('content', '').lower():
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    async def store_research_report(self, report: Dict[str, Any]) -> str:
        """Store a research report."""
        self.operation_count += 1
        report_id = report.get('id', f"report_{self.operation_count}")
        self.research_reports[report_id] = report
        return report_id
    
    async def get_research_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a research report by ID."""
        self.operation_count += 1
        return self.research_reports.get(report_id)
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value."""
        self.operation_count += 1
        self.cache_data[key] = {
            'value': value,
            'expires_at': datetime.now(timezone.utc).timestamp() + ttl
        }
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value."""
        self.operation_count += 1
        
        if key in self.cache_data:
            cache_entry = self.cache_data[key]
            if datetime.now(timezone.utc).timestamp() < cache_entry['expires_at']:
                return cache_entry['value']
            else:
                # Expired, remove from cache
                del self.cache_data[key]
        
        return None
    
    async def cache_delete(self, key: str):
        """Delete cache value."""
        self.operation_count += 1
        self.cache_data.pop(key, None)
    
    async def health_check(self) -> bool:
        """Perform health check."""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'content_items': len(self.content_items),
            'research_reports': len(self.research_reports),
            'cache_entries': len(self.cache_data),
            'total_operations': self.operation_count
        }


class MockResearchAgentFixtures:
    """Collection of mock fixtures for research agent testing."""
    
    def __init__(self):
        self.web_scraper = MockWebScraper()
        self.rss_manager = MockRSSFeedManager()
        self.robots_txt = MockRobotsTxt()
        self.search_api = MockSearchAPI()
        self.database = MockContentDatabase()
        self._setup_default_mocks()
    
    def _setup_default_mocks(self):
        """Setup default mock responses."""
        # Web scraper defaults
        self.web_scraper.add_html_response(
            "https://example.com/ai-article",
            self._create_ai_article_html()
        )
        
        self.web_scraper.add_html_response(
            "https://example.com/tech-news",
            self._create_tech_news_html()
        )
        
        # RSS feed defaults
        self.rss_manager.create_tech_news_feed("https://example.com/tech-feed.rss")
        self.rss_manager.create_research_feed("https://example.com/research-feed.rss")
        
        # Robots.txt defaults
        self.robots_txt.add_permissive_robots("example.com")
        self.robots_txt.add_restrictive_robots("restricted.com")
        
        # Search API defaults
        self.search_api.create_default_results("artificial intelligence")
        self.search_api.create_default_results("machine learning")
    
    def _create_ai_article_html(self) -> str:
        """Create mock AI article HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>The Future of Artificial Intelligence: Trends and Predictions</title>
            <meta name="description" content="Explore the latest trends in AI technology and future predictions.">
            <meta name="author" content="Dr. Sarah Johnson">
            <meta name="date" content="2024-01-15">
        </head>
        <body>
            <article>
                <h1>The Future of Artificial Intelligence: Trends and Predictions</h1>
                <div class="author">By Dr. Sarah Johnson</div>
                <div class="date">January 15, 2024</div>
                
                <p>Artificial intelligence is rapidly evolving and transforming various industries. 
                This comprehensive analysis explores the current trends and future predictions for AI technology.</p>
                
                <h2>Current AI Trends</h2>
                <p>Machine learning algorithms are becoming more sophisticated, with deep learning 
                models showing remarkable improvements in natural language processing and computer vision.</p>
                
                <h2>Future Predictions</h2>
                <p>Experts predict that AI will become more integrated into daily life, with 
                autonomous systems and intelligent assistants becoming commonplace.</p>
                
                <h2>Challenges and Opportunities</h2>
                <p>While AI presents numerous opportunities, challenges around ethics, privacy, 
                and job displacement must be carefully addressed.</p>
            </article>
        </body>
        </html>
        """
    
    def _create_tech_news_html(self) -> str:
        """Create mock tech news HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Latest Technology News and Updates</title>
            <meta name="description" content="Stay updated with the latest technology news and developments.">
        </head>
        <body>
            <main>
                <h1>Technology News</h1>
                
                <article>
                    <h2>Breakthrough in Quantum Computing</h2>
                    <p>Scientists have achieved a new milestone in quantum computing, 
                    demonstrating quantum supremacy in practical applications.</p>
                    <span class="date">January 20, 2024</span>
                </article>
                
                <article>
                    <h2>AI-Powered Medical Diagnosis</h2>
                    <p>New AI system shows 95% accuracy in medical diagnosis, 
                    potentially revolutionizing healthcare delivery.</p>
                    <span class="date">January 18, 2024</span>
                </article>
                
                <article>
                    <h2>Sustainable Tech Innovations</h2>
                    <p>Green technology initiatives are gaining momentum, 
                    with new sustainable solutions for energy and transportation.</p>
                    <span class="date">January 16, 2024</span>
                </article>
            </main>
        </body>
        </html>
        """
    
    def get_mock_http_client(self):
        """Get mock HTTP client with all responses configured."""
        mock_client = MagicMock()
        
        async def mock_get(url, **kwargs):
            # Check if it's a robots.txt request
            if url.endswith('/robots.txt'):
                return await self.robots_txt.get_robots_txt(url)
            
            # Check if it's an RSS feed request
            if 'feed' in url or url.endswith('.rss') or url.endswith('.xml'):
                return await self.rss_manager.get_feed(url)
            
            # Default to web scraper
            return await self.web_scraper.get(url, **kwargs)
        
        mock_client.get = mock_get
        return mock_client
    
    def reset_all_mocks(self):
        """Reset all mock states."""
        self.web_scraper.reset()
        self.rss_manager.request_count = 0
        self.robots_txt.request_count = 0
        self.search_api.request_count = 0
        self.search_api.rate_limit_calls = 0
        self.database.operation_count = 0
        self.database.content_items.clear()
        self.database.research_reports.clear()
        self.database.cache_data.clear()


# Utility functions for creating test fixtures
def create_sample_content_items(count: int = 5) -> List[Dict[str, Any]]:
    """Create sample content items for testing."""
    items = []
    topics = ["AI", "Machine Learning", "Quantum Computing", "Blockchain", "Cybersecurity"]
    
    for i in range(count):
        topic = topics[i % len(topics)]
        item = {
            'id': f'content_{i+1:03d}',
            'url': f'https://example.com/article-{i+1}',
            'title': f'{topic} Article {i+1}',
            'content': f'This is a comprehensive article about {topic} and its applications.',
            'author': f'Author {i+1}',
            'published_date': datetime.now(timezone.utc).isoformat(),
            'source_type': 'web',
            'tags': [topic.lower(), 'technology', 'research'],
            'relevance_score': 0.7 + (i * 0.05),
            'summary': f'Summary of {topic} article {i+1}',
        }
        items.append(item)
    
    return items


def create_sample_research_tasks(count: int = 3) -> List[Dict[str, Any]]:
    """Create sample research tasks for testing."""
    tasks = []
    queries = ["artificial intelligence", "machine learning algorithms", "quantum computing applications"]
    
    for i in range(count):
        query = queries[i % len(queries)]
        task = {
            'id': f'task_{i+1:03d}',
            'query': query,
            'sources': [
                f'https://example.com/source-{i+1}',
                f'https://research.example.com/feed-{i+1}.rss'
            ],
            'max_results': 20 + (i * 10),
            'priority': 1 + (i % 3),
            'deadline': (datetime.now(timezone.utc).timestamp() + 3600 + (i * 1800)),
            'filters': {
                'date_range': '30d',
                'content_type': ['article', 'research_paper'],
                'language': 'en'
            }
        }
        tasks.append(task)
    
    return tasks


# Error simulation utilities
class ErrorSimulator:
    """Utility class for simulating various error conditions."""
    
    @staticmethod
    def simulate_network_timeout():
        """Simulate network timeout error."""
        raise asyncio.TimeoutError("Network timeout")
    
    @staticmethod
    def simulate_rate_limit_error():
        """Simulate rate limiting error."""
        raise Exception("Rate limit exceeded. Please try again later.")
    
    @staticmethod
    def simulate_http_error(status_code: int = 500):
        """Simulate HTTP error."""
        return MockHttpResponse(
            status_code=status_code,
            text=f"HTTP {status_code} Error",
            headers={'content-type': 'text/plain'}
        )
    
    @staticmethod
    def simulate_malformed_content():
        """Simulate malformed content."""
        return MockHttpResponse(
            status_code=200,
            text="<html><body><p>Malformed HTML without closing tags<body></html>",
            headers={'content-type': 'text/html'}
        )
    
    @staticmethod
    def simulate_empty_response():
        """Simulate empty response."""
        return MockHttpResponse(
            status_code=200,
            text="",
            headers={'content-type': 'text/html'}
        )