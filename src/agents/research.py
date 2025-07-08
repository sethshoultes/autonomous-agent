"""
Research Agent for autonomous research automation.
Provides comprehensive research capabilities including web scraping, RSS monitoring,
content analysis, and report generation.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import uuid

import httpx
from bs4 import BeautifulSoup
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .base import BaseAgent, AgentMessage, AgentState
from .exceptions import AgentError


# Research-specific exceptions
class ResearchException(AgentError):
    """Base exception for research-related errors."""
    pass


class RateLimitError(ResearchException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ContentExtractionError(ResearchException):
    """Exception raised during content extraction."""
    
    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(message)
        self.url = url


class CacheError(ResearchException):
    """Exception raised during cache operations."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message)
        self.operation = operation


# Data models
@dataclass
class ContentItem:
    """Represents a single piece of research content."""
    id: str
    url: str
    title: str
    content: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    source_type: str = "web"  # web, rss, api
    tags: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "source_type": self.source_type,
            "tags": self.tags,
            "relevance_score": self.relevance_score,
            "summary": self.summary,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            url=data["url"],
            title=data["title"],
            content=data["content"],
            author=data.get("author"),
            published_date=datetime.fromisoformat(data["published_date"]) if data.get("published_date") else None,
            source_type=data.get("source_type", "web"),
            tags=data.get("tags", []),
            relevance_score=data.get("relevance_score", 0.0),
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
        )
    
    def calculate_similarity(self, other: "ContentItem") -> float:
        """Calculate similarity with another content item."""
        if not self.content or not other.content:
            return 0.0
        
        # Use TF-IDF for content similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform([self.content, other.content])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Fallback to simple text comparison
            return len(set(self.content.lower().split()) & set(other.content.lower().split())) / \
                   len(set(self.content.lower().split()) | set(other.content.lower().split()))
    
    def is_duplicate(self, other: "ContentItem", threshold: float = 0.8) -> bool:
        """Check if this item is a duplicate of another."""
        # Check URL first
        if self.url == other.url:
            return True
        
        # Check content similarity
        similarity = self.calculate_similarity(other)
        return similarity >= threshold


@dataclass
class ResearchTask:
    """Represents a research task."""
    id: str
    query: str
    sources: List[str]
    max_results: int = 50
    priority: int = 1
    deadline: Optional[float] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate task parameters."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.sources:
            raise ValueError("At least one source must be specified")
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "sources": self.sources,
            "max_results": self.max_results,
            "priority": self.priority,
            "deadline": self.deadline,
            "filters": self.filters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchTask":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            sources=data["sources"],
            max_results=data.get("max_results", 50),
            priority=data.get("priority", 1),
            deadline=data.get("deadline"),
            filters=data.get("filters", {}),
        )


@dataclass
class ResearchResult:
    """Represents research results."""
    task_id: str
    query: str
    items: List[ContentItem]
    total_sources: int
    processing_time: float
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "query": self.query,
            "items": [item.to_dict() for item in self.items],
            "total_sources": self.total_sources,
            "processing_time": self.processing_time,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class ResearchReport:
    """Represents a comprehensive research report."""
    id: str
    query: str
    results: List[ContentItem]
    generated_at: datetime
    total_sources: int
    processing_time: float
    summary: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "results": [item.to_dict() for item in self.results],
            "generated_at": self.generated_at.isoformat(),
            "total_sources": self.total_sources,
            "processing_time": self.processing_time,
            "summary": self.summary,
            "insights": self.insights,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchReport":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            results=[ContentItem.from_dict(item) for item in data["results"]],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            total_sources=data["total_sources"],
            processing_time=data["processing_time"],
            summary=data.get("summary"),
            insights=data.get("insights", []),
        )
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the research report."""
        if not self.results:
            return {
                "total_results": 0,
                "average_relevance": 0.0,
                "top_keywords": [],
                "source_distribution": {},
            }
        
        # Calculate statistics
        total_results = len(self.results)
        average_relevance = sum(item.relevance_score for item in self.results) / total_results
        
        # Extract top keywords from all content
        all_content = " ".join(item.content for item in self.results)
        keywords = self._extract_top_keywords(all_content)
        
        # Source distribution
        source_counts = {}
        for item in self.results:
            source = urlparse(item.url).netloc
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_results": total_results,
            "average_relevance": average_relevance,
            "top_keywords": keywords,
            "source_distribution": source_counts,
        }
    
    def _extract_top_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract top keywords from text."""
        # Simple keyword extraction using TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=max_keywords * 2,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:max_keywords]]
        except Exception:
            # Fallback to simple word frequency
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:max_keywords]


class ResearchQuery:
    """Handles query optimization and expansion."""
    
    def __init__(self, query: str):
        self.original_query = query
        self.optimized_query = query
        self.expanded_terms = []
    
    def optimize(self) -> str:
        """Optimize the query for better search results."""
        # Remove common stop words for search optimization
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = self.original_query.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        
        self.optimized_query = ' '.join(filtered_words)
        return self.optimized_query
    
    def expand(self) -> List[str]:
        """Expand query with synonyms and related terms."""
        # Simple expansion with common synonyms
        expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
            'ml': ['machine learning', 'artificial intelligence', 'data science'],
            'tech': ['technology', 'technical', 'innovation'],
            'research': ['study', 'analysis', 'investigation'],
        }
        
        expanded = [self.original_query]
        words = self.original_query.lower().split()
        
        for word in words:
            if word in expansions:
                for synonym in expansions[word]:
                    # Replace in a case-insensitive manner
                    import re
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    expanded_query = pattern.sub(synonym, self.original_query)
                    expanded.append(expanded_query)
        
        self.expanded_terms = expanded
        return expanded
    
    def extract_topics(self) -> List[str]:
        """Extract topics from the query."""
        # Simple topic extraction
        topics = []
        words = self.original_query.lower().split()
        
        # Multi-word topics
        for i in range(len(words) - 1):
            topic = f"{words[i]} {words[i+1]}"
            if len(topic) > 5:  # Reasonable topic length
                topics.append(topic)
        
        # Single word topics
        for word in words:
            if len(word) > 3:  # Skip short words
                topics.append(word)
        
        return topics


class ContentExtractor:
    """Handles content extraction from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get("timeout", 30)
        self.max_content_length = config.get("max_content_length", 1024 * 1024)
        self.user_agent = config.get("user_agent", "AutonomousAgent/1.0 Research Bot")
        
        # HTTP client configuration
        self.http_client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )
    
    async def extract_web_content(self, url: str) -> ContentItem:
        """Extract content from a web page."""
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            if len(response.content) > self.max_content_length:
                raise ContentExtractionError(f"Content too large: {len(response.content)} bytes", url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else urlparse(url).path
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract author
            author = self._extract_author(soup)
            
            # Extract published date
            published_date = self._extract_published_date(soup)
            
            # Generate content ID
            content_id = hashlib.md5(url.encode()).hexdigest()
            
            return ContentItem(
                id=content_id,
                url=url,
                title=title,
                content=content,
                author=author,
                published_date=published_date,
                source_type="web",
                tags=self._extract_tags(soup),
                summary=self._generate_summary(content),
            )
            
        except httpx.TimeoutException:
            raise ContentExtractionError(f"Timeout while fetching {url}", url)
        except httpx.HTTPStatusError as e:
            raise ContentExtractionError(f"HTTP error {e.response.status_code} for {url}", url)
        except Exception as e:
            raise ContentExtractionError(f"Error extracting content from {url}: {str(e)}", url)
    
    async def extract_rss_content(self, url: str) -> List[ContentItem]:
        """Extract content from RSS feed."""
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            # Parse RSS feed
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                return []
            
            items = []
            for entry in feed.entries:
                # Generate content ID
                content_id = hashlib.md5(entry.link.encode()).hexdigest()
                
                # Extract published date
                published_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                
                # Extract content
                content = entry.get('summary', '') or entry.get('description', '')
                
                item = ContentItem(
                    id=content_id,
                    url=entry.link,
                    title=entry.title,
                    content=content,
                    author=entry.get('author'),
                    published_date=published_date,
                    source_type="rss",
                    tags=self._extract_rss_tags(entry),
                    summary=self._generate_summary(content),
                )
                items.append(item)
            
            return items
            
        except Exception as e:
            raise ContentExtractionError(f"Error extracting RSS content from {url}: {str(e)}", url)
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        main_content = None
        for selector in ['main', 'article', '.content', '.post', '.entry']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text content
        text = main_content.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author from HTML."""
        # Try various author selectors
        author_selectors = [
            'meta[name="author"]',
            '.author',
            '.byline',
            '[rel="author"]',
            '.post-author',
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content')
                return element.get_text().strip()
        
        return None
    
    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract published date from HTML."""
        # Try various date selectors
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'time[datetime]',
            '.date',
            '.published',
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_text = None
                if element.name == 'meta':
                    date_text = element.get('content')
                elif element.name == 'time':
                    date_text = element.get('datetime')
                else:
                    date_text = element.get_text().strip()
                
                if date_text:
                    try:
                        # Try to parse ISO format first
                        return datetime.fromisoformat(date_text.replace('Z', '+00:00'))
                    except ValueError:
                        # Try other common formats
                        formats = [
                            '%Y-%m-%d',
                            '%Y-%m-%d %H:%M:%S',
                            '%B %d, %Y',
                            '%d %B %Y',
                        ]
                        for fmt in formats:
                            try:
                                return datetime.strptime(date_text, fmt)
                            except ValueError:
                                continue
        
        return None
    
    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract tags from HTML."""
        tags = []
        
        # Try meta keywords
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            keywords = keywords_meta.get('content', '')
            tags.extend([tag.strip() for tag in keywords.split(',') if tag.strip()])
        
        # Try to find tag elements
        for selector in ['.tags', '.categories', '.tag']:
            elements = soup.select(selector)
            for element in elements:
                tag_text = element.get_text().strip()
                if tag_text:
                    tags.append(tag_text)
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_rss_tags(self, entry) -> List[str]:
        """Extract tags from RSS entry."""
        tags = []
        
        # Try category field
        if hasattr(entry, 'tags'):
            tags.extend([tag.term for tag in entry.tags if hasattr(tag, 'term')])
        
        # Try categories
        if hasattr(entry, 'categories'):
            tags.extend(entry.categories)
        
        return tags
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a summary of content."""
        if len(content) <= max_length:
            return content
        
        # Simple extractive summary - take first sentence and most important sentences
        sentences = content.split('.')
        if not sentences:
            return content[:max_length]
        
        # Return first sentence as summary
        summary = sentences[0].strip()
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


class ContentScorer:
    """Handles content relevance scoring and categorization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.relevance_threshold = config.get("relevance_threshold", 0.6)
        self.scoring_algorithm = config.get("scoring_algorithm", "tfidf")
    
    def calculate_relevance_score(self, content: ContentItem, query: str) -> float:
        """Calculate relevance score for content against query."""
        if not content.content or not query:
            return 0.0
        
        try:
            # Use TF-IDF similarity
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([content.content, query])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Boost score based on title relevance
            title_score = 0.0
            if content.title:
                title_tfidf = vectorizer.transform([content.title])
                title_similarity = cosine_similarity(title_tfidf, tfidf_matrix[1:2])[0][0]
                title_score = title_similarity * 0.3  # 30% weight for title
            
            # Combine scores
            final_score = similarity * 0.7 + title_score
            
            return min(1.0, max(0.0, final_score))
            
        except Exception:
            # Fallback to simple word matching
            query_words = set(query.lower().split())
            content_words = set(content.content.lower().split())
            
            intersection = query_words & content_words
            union = query_words | content_words
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
    
    def categorize_content(self, content: ContentItem) -> List[str]:
        """Categorize content based on its content and metadata."""
        categories = []
        
        # Use existing tags if available
        if content.tags:
            categories.extend(content.tags)
        
        # Simple keyword-based categorization
        text = (content.title + " " + content.content).lower()
        
        category_keywords = {
            'technology': ['technology', 'tech', 'software', 'hardware', 'digital', 'computer'],
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai'],
            'science': ['research', 'study', 'experiment', 'analysis', 'scientific'],
            'business': ['business', 'company', 'market', 'industry', 'commercial'],
            'health': ['health', 'medical', 'medicine', 'healthcare', 'treatment'],
            'education': ['education', 'learning', 'university', 'school', 'academic'],
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return list(set(categories))  # Remove duplicates
    
    def extract_keywords(self, content: ContentItem, max_keywords: int = 10) -> List[str]:
        """Extract keywords from content."""
        text = content.title + " " + content.content
        
        try:
            # Use TF-IDF for keyword extraction
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=max_keywords * 2,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:max_keywords]]
            
        except Exception:
            # Fallback to simple frequency analysis
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:max_keywords]


class ResearchCache:
    """Handles caching of research data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_type = config.get("cache_type", "memory")
        self.ttl = config.get("ttl", 3600)  # 1 hour default
        self.max_size = config.get("max_size", 1000)
        
        # In-memory cache
        self._cache = {}
        self._access_times = {}
        self._hit_count = 0
        self._miss_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            # Check if expired
            if time.time() - self._access_times[key] > self.ttl:
                del self._cache[key]
                del self._access_times[key]
                self._miss_count += 1
                return None
            
            self._hit_count += 1
            return self._cache[key]
        
        self._miss_count += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache."""
        # Cleanup if cache is full
        if len(self._cache) >= self.max_size:
            await self._cleanup_cache()
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    async def invalidate(self, key: str):
        """Invalidate cache entry."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    async def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "total_items": len(self._cache),
            "max_size": self.max_size,
        }
    
    async def health_check(self) -> bool:
        """Perform health check."""
        return True  # Memory cache is always healthy
    
    async def _cleanup_cache(self):
        """Cleanup expired entries and enforce size limit."""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, access_time in self._access_times.items()
            if current_time - access_time > self.ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
        
        # If still over limit, remove oldest entries
        if len(self._cache) >= self.max_size:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
            keys_to_remove = sorted_keys[:len(self._cache) - self.max_size + 1]
            
            for key in keys_to_remove:
                del self._cache[key]
                del self._access_times[key]


class RobotsTxtChecker:
    """Handles robots.txt compliance checking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_agent = config.get("user_agent", "AutonomousAgent/1.0")
        self.cache_ttl = config.get("cache_ttl", 3600)
        
        # Cache for robots.txt files
        self._robots_cache = {}
        self._cache_times = {}
        
        # HTTP client
        self.http_client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=10,
        )
    
    async def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            # Check cache first
            if robots_url in self._robots_cache:
                cache_time = self._cache_times.get(robots_url, 0)
                if time.time() - cache_time < self.cache_ttl:
                    parser = self._robots_cache[robots_url]
                    return parser.can_fetch(self.user_agent, url)
            
            # Fetch robots.txt
            try:
                response = await self.http_client.get(robots_url)
                if response.status_code == 200:
                    # Parse robots.txt
                    parser = RobotFileParser()
                    parser.set_url(robots_url)
                    parser.feed(response.text)
                    
                    # Cache the parser
                    self._robots_cache[robots_url] = parser
                    self._cache_times[robots_url] = time.time()
                    
                    return parser.can_fetch(self.user_agent, url)
                else:
                    # If robots.txt not found, assume allowed
                    return True
            except Exception:
                # If can't fetch robots.txt, assume allowed
                return True
                
        except Exception:
            # If any error, assume allowed
            return True
    
    async def get_crawl_delay(self, url: str) -> Optional[int]:
        """Get crawl delay for URL."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            if robots_url in self._robots_cache:
                parser = self._robots_cache[robots_url]
                return parser.crawl_delay(self.user_agent)
            
            return None
            
        except Exception:
            return None
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


class FeedMonitor:
    """Handles RSS feed monitoring and processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.check_interval = config.get("check_interval", 300)  # 5 minutes
        self.max_items = config.get("max_items", 100)
        self.concurrent_feeds = config.get("concurrent_feeds", 10)
        
        # Monitored feeds
        self._feeds = set()
        self._last_check = {}
        self._seen_items = set()
        
        # Content extractor
        self.content_extractor = ContentExtractor(config)
    
    async def add_feed(self, url: str):
        """Add feed to monitor."""
        self._feeds.add(url)
        self._last_check[url] = 0
    
    async def remove_feed(self, url: str):
        """Remove feed from monitoring."""
        self._feeds.discard(url)
        self._last_check.pop(url, None)
    
    async def check_feeds(self) -> List[ContentItem]:
        """Check all feeds for new items."""
        if not self._feeds:
            return []
        
        # Create semaphore for concurrent processing
        semaphore = asyncio.Semaphore(self.concurrent_feeds)
        
        async def check_single_feed(feed_url: str) -> List[ContentItem]:
            async with semaphore:
                try:
                    return await self.content_extractor.extract_rss_content(feed_url)
                except Exception:
                    return []
        
        # Check all feeds concurrently
        tasks = [check_single_feed(feed_url) for feed_url in self._feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect new items
        new_items = []
        for feed_url, result in zip(self._feeds, results):
            if isinstance(result, list):
                # Filter out seen items
                for item in result:
                    item_key = f"{item.url}:{item.title}"
                    if item_key not in self._seen_items:
                        self._seen_items.add(item_key)
                        new_items.append(item)
                
                # Update last check time
                self._last_check[feed_url] = time.time()
        
        # Limit number of items
        if len(new_items) > self.max_items:
            new_items = new_items[:self.max_items]
        
        return new_items
    
    async def get_feed_status(self) -> Dict[str, Any]:
        """Get status of all monitored feeds."""
        return {
            "total_feeds": len(self._feeds),
            "last_checks": self._last_check,
            "total_seen_items": len(self._seen_items),
        }


class ResearchAgent(BaseAgent):
    """
    Research Agent for autonomous research automation.
    
    Provides comprehensive research capabilities including:
    - Multi-source web scraping with intelligent content extraction
    - RSS feed processing, monitoring, and aggregation
    - Content deduplication and relevance scoring
    - Research report generation with structured output
    - Data persistence and intelligent caching
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], logger: logging.Logger, message_broker: Any, service_manager: Optional[Any] = None):
        """Initialize the research agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
            logger: Logger instance
            message_broker: Message broker for communication
            service_manager: AI service manager for enhanced capabilities
        """
        super().__init__(agent_id, config, logger, message_broker)
        
        # Research-specific configuration
        self.max_concurrent_requests = config.get("max_concurrent_requests", 5)
        self.request_timeout = config.get("request_timeout", 30)
        self.rate_limit_delay = config.get("rate_limit_delay", 1.0)
        self.cache_ttl = config.get("cache_ttl", 3600)
        self.respect_robots_txt = config.get("respect_robots_txt", True)
        self.relevance_threshold = config.get("relevance_threshold", 0.6)
        self.deduplication_threshold = config.get("deduplication_threshold", 0.8)
        
        # AI service manager for enhanced capabilities
        self.service_manager = service_manager
        self.ai_enabled = service_manager is not None and getattr(service_manager, 'is_initialized', False)
        
        # AI-enhanced research features configuration
        self.ai_content_analysis_enabled = config.get("ai_content_analysis_enabled", True)
        self.ai_summarization_enabled = config.get("ai_summarization_enabled", True)
        self.ai_relevance_scoring_enabled = config.get("ai_relevance_scoring_enabled", True)
        self.ai_research_insights_enabled = config.get("ai_research_insights_enabled", True)
        self.ai_fact_checking_enabled = config.get("ai_fact_checking_enabled", False)
        
        # Research components
        self._content_extractor: Optional[ContentExtractor] = None
        self._content_scorer: Optional[ContentScorer] = None
        self._research_cache: Optional[ResearchCache] = None
        self._robots_checker: Optional[RobotsTxtChecker] = None
        self._feed_monitor: Optional[FeedMonitor] = None
        
        # Research state
        self._active_tasks = {}
        self._research_history = []
        
        # Add research-specific metrics
        self.metrics.update({
            "research_tasks_completed": 0,
            "content_items_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "robots_txt_checks": 0,
            "rate_limit_hits": 0,
            "ai_content_analyses": 0,
            "ai_summaries_generated": 0,
            "ai_insights_extracted": 0,
            "ai_relevance_scores": 0,
        })
    
    async def _initialize(self):
        """Initialize research agent components."""
        self.logger.info(f"Initializing research agent {self.agent_id}")
        
        # Initialize components
        self._content_extractor = ContentExtractor(self.config)
        self._content_scorer = ContentScorer(self.config)
        self._research_cache = ResearchCache(self.config)
        self._robots_checker = RobotsTxtChecker(self.config)
        self._feed_monitor = FeedMonitor(self.config)
        
        # Set up RSS feed monitoring if configured
        rss_feeds = self.config.get("rss_feeds", [])
        for feed_url in rss_feeds:
            await self._feed_monitor.add_feed(feed_url)
        
        self.logger.info(f"Research agent {self.agent_id} initialized with {len(rss_feeds)} RSS feeds")
    
    async def _cleanup(self):
        """Cleanup research agent resources."""
        self.logger.info(f"Cleaning up research agent {self.agent_id}")
        
        # Close HTTP clients
        if self._content_extractor:
            await self._content_extractor.close()
        if self._robots_checker:
            await self._robots_checker.close()
        
        # Clear cache
        if self._research_cache:
            await self._research_cache.clear()
    
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        if message.message_type == "research_request":
            return await self._handle_research_request(message)
        elif message.message_type == "feed_check":
            return await self._handle_feed_check(message)
        elif message.message_type == "cache_clear":
            return await self._handle_cache_clear(message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
            return None
    
    async def _handle_research_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle research request message."""
        try:
            # Extract task from message
            task_data = message.payload
            task = ResearchTask.from_dict(task_data)
            
            # Execute research task
            result = await self._execute_research_task(task)
            
            # Send response
            response = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="research_results",
                payload=result.to_dict(),
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling research request: {e}")
            
            # Send error response
            error_response = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="research_error",
                payload={"error": str(e)},
            )
            
            return error_response
    
    async def _handle_feed_check(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle feed check message."""
        try:
            new_items = await self._feed_monitor.check_feeds()
            
            if new_items:
                # Send new items to requester
                response = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender=self.agent_id,
                    recipient=message.sender,
                    message_type="feed_items",
                    payload={"items": [item.to_dict() for item in new_items]},
                )
                
                return response
            
        except Exception as e:
            self.logger.error(f"Error checking feeds: {e}")
        
        return None
    
    async def _handle_cache_clear(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle cache clear message."""
        try:
            await self._research_cache.clear()
            
            response = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="cache_cleared",
                payload={"status": "success"},
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            
            error_response = AgentMessage(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                recipient=message.sender,
                message_type="cache_error",
                payload={"error": str(e)},
            )
            
            return error_response
    
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research task."""
        try:
            # Convert task dict to ResearchTask
            research_task = ResearchTask.from_dict(task)
            
            # Execute research
            result = await self._execute_research_task(research_task)
            
            # Generate report
            report = await self._generate_research_report(result)
            
            return {
                "status": "completed",
                "task_id": research_task.id,
                "results_count": len(result.items),
                "processing_time": result.processing_time,
                "results": [item.to_dict() for item in result.items],
                "report": report.to_dict(),
            }
            
        except Exception as e:
            self.logger.error(f"Error executing research task: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    async def _execute_research_task(self, task: ResearchTask) -> ResearchResult:
        """Execute a research task and return results."""
        start_time = time.time()
        
        self.logger.info(f"Executing research task: {task.id}")
        
        # Store active task
        self._active_tasks[task.id] = task
        
        try:
            # Optimize query
            query_optimizer = ResearchQuery(task.query)
            optimized_query = query_optimizer.optimize()
            expanded_queries = query_optimizer.expand()
            
            # Collect content from all sources
            all_content = []
            
            # Process each source
            for source_url in task.sources:
                try:
                    # Check robots.txt if required
                    if self.respect_robots_txt:
                        if not await self._robots_checker.is_allowed(source_url):
                            self.logger.warning(f"Robots.txt disallows access to {source_url}")
                            continue
                        
                        # Get crawl delay
                        crawl_delay = await self._robots_checker.get_crawl_delay(source_url)
                        if crawl_delay:
                            await asyncio.sleep(crawl_delay)
                    
                    # Extract content based on source type
                    if source_url.endswith('.rss') or source_url.endswith('.xml') or 'feed' in source_url:
                        # RSS feed
                        content_items = await self._content_extractor.extract_rss_content(source_url)
                        all_content.extend(content_items)
                    else:
                        # Web page
                        content_item = await self._content_extractor.extract_web_content(source_url)
                        all_content.append(content_item)
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    self.logger.error(f"Error processing source {source_url}: {e}")
                    continue
            
            # Score content for relevance
            scored_content = []
            for content in all_content:
                relevance_score = self._content_scorer.calculate_relevance_score(content, optimized_query)
                content.relevance_score = relevance_score
                
                # Filter by relevance threshold
                if relevance_score >= self.relevance_threshold:
                    scored_content.append(content)
            
            # Deduplicate content
            deduplicated_content = self._deduplicate_content(scored_content)
            
            # Sort by relevance score
            deduplicated_content.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit results
            final_results = deduplicated_content[:task.max_results]
            
            # Update metrics
            self.metrics["research_tasks_completed"] += 1
            self.metrics["content_items_processed"] += len(all_content)
            
            # Cache results
            cache_key = f"research:{task.id}"
            await self._research_cache.set(cache_key, final_results)
            
            processing_time = time.time() - start_time
            
            result = ResearchResult(
                task_id=task.id,
                query=task.query,
                items=final_results,
                total_sources=len(task.sources),
                processing_time=processing_time,
                generated_at=datetime.now(timezone.utc),
            )
            
            # Store in history
            self._research_history.append(result)
            
            self.logger.info(f"Research task {task.id} completed in {processing_time:.2f}s with {len(final_results)} results")
            
            return result
            
        finally:
            # Remove from active tasks
            self._active_tasks.pop(task.id, None)
    
    def _deduplicate_content(self, content_items: List[ContentItem]) -> List[ContentItem]:
        """Remove duplicate content items."""
        unique_items = []
        seen_urls = set()
        
        for item in content_items:
            # Check URL duplicates first
            if item.url in seen_urls:
                continue
            
            # Check content similarity
            is_duplicate = False
            for existing_item in unique_items:
                if item.is_duplicate(existing_item, self.deduplication_threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_items.append(item)
                seen_urls.add(item.url)
        
        return unique_items
    
    async def _generate_research_report(self, result: ResearchResult) -> ResearchReport:
        """Generate a comprehensive research report."""
        report = ResearchReport(
            id=str(uuid.uuid4()),
            query=result.query,
            results=result.items,
            generated_at=result.generated_at,
            total_sources=result.total_sources,
            processing_time=result.processing_time,
        )
        
        # Generate summary
        if result.items:
            summary_stats = report.generate_summary()
            report.summary = f"Found {len(result.items)} relevant items from {result.total_sources} sources. " \
                           f"Average relevance score: {summary_stats['average_relevance']:.2f}. " \
                           f"Top keywords: {', '.join(summary_stats['top_keywords'][:5])}"
        
        # Generate insights
        insights = await self._generate_insights(result.items)
        report.insights = insights
        
        return report
    
    async def _generate_insights(self, items: List[ContentItem]) -> List[str]:
        """Generate insights from research results."""
        insights = []
        
        if not items:
            return insights
        
        # Source diversity analysis
        sources = set(urlparse(item.url).netloc for item in items)
        insights.append(f"Content sourced from {len(sources)} different domains")
        
        # Content type analysis
        content_types = {}
        for item in items:
            content_types[item.source_type] = content_types.get(item.source_type, 0) + 1
        
        type_analysis = ", ".join(f"{count} {type_}" for type_, count in content_types.items())
        insights.append(f"Content types: {type_analysis}")
        
        # Temporal analysis
        dated_items = [item for item in items if item.published_date]
        if dated_items:
            latest_date = max(item.published_date for item in dated_items)
            oldest_date = min(item.published_date for item in dated_items)
            insights.append(f"Content spans from {oldest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        
        # Quality analysis
        avg_relevance = sum(item.relevance_score for item in items) / len(items)
        high_quality_count = sum(1 for item in items if item.relevance_score > 0.8)
        insights.append(f"Average relevance: {avg_relevance:.2f}, {high_quality_count} high-quality items")
        
        return insights
    
    async def _health_check(self) -> bool:
        """Perform health check."""
        try:
            # Check if all components are initialized
            if not all([
                self._content_extractor,
                self._content_scorer,
                self._research_cache,
                self._robots_checker,
                self._feed_monitor,
            ]):
                return False
            
            # Check cache health
            cache_healthy = await self._research_cache.health_check()
            if not cache_healthy:
                return False
            
            # Check if we can perform basic operations
            test_query = ResearchQuery("test query")
            test_query.optimize()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get research agent metrics."""
        base_metrics = super().get_metrics()
        
        # Add research-specific metrics
        research_metrics = {
            "active_tasks": len(self._active_tasks),
            "research_history_size": len(self._research_history),
            "monitored_feeds": len(self._feed_monitor._feeds) if self._feed_monitor else 0,
        }
        
        return {**base_metrics, **research_metrics}
    
    async def get_research_history(self) -> List[Dict[str, Any]]:
        """Get research history."""
        return [result.to_dict() for result in self._research_history]
    
    async def search_research_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search research history."""
        if not self._research_history:
            return []
        
        # Simple search in research history
        matching_results = []
        query_lower = query.lower()
        
        for result in self._research_history:
            if query_lower in result.query.lower():
                matching_results.append(result.to_dict())
                if len(matching_results) >= limit:
                    break
        
        return matching_results
    
    # AI-Enhanced Research Methods
    
    async def analyze_content_with_ai(
        self,
        content: str,
        research_questions: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze research content using AI.
        
        Args:
            content: Content to analyze
            research_questions: Specific questions to address
            focus_areas: Areas to focus analysis on
            
        Returns:
            AI analysis results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_content_analysis_enabled:
                return {"error": "AI content analysis not available", "success": False}
            
            # Use AI service manager for research content analysis
            analysis = await self.service_manager.analyze_research_content(
                content=content,
                research_questions=research_questions,
                focus_areas=focus_areas
            )
            
            if analysis.get("success"):
                self.metrics["ai_content_analyses"] += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"AI content analysis failed: {e}")
            return {"error": str(e), "success": False}
    
    async def summarize_research_content_with_ai(
        self,
        content: str,
        summary_type: str = "general",
        max_length: int = 200,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Summarize research content using AI.
        
        Args:
            content: Content to summarize
            summary_type: Type of summary (general, bullet_points, executive)
            max_length: Maximum summary length
            focus_areas: Specific areas to focus on
            
        Returns:
            AI summarization results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_summarization_enabled:
                return {"error": "AI summarization not available", "success": False}
            
            # Generate AI summary
            summary_result = await self.service_manager.summarize_text(
                text=content,
                max_length=max_length,
                focus_areas=focus_areas,
                summary_type=summary_type
            )
            
            if summary_result.get("success"):
                self.metrics["ai_summaries_generated"] += 1
            
            return {
                "summary": summary_result.get("content", ""),
                "processing_time": summary_result.get("processing_time", 0),
                "success": summary_result.get("success", False),
                "error": summary_result.get("error")
            }
            
        except Exception as e:
            self.logger.error(f"AI research summarization failed: {e}")
            return {"error": str(e), "success": False}
    
    async def extract_research_insights_with_ai(
        self,
        content: str,
        insight_types: List[str] = None
    ) -> Dict[str, Any]:
        """Extract research insights using AI.
        
        Args:
            content: Content to extract insights from
            insight_types: Types of insights to extract
            
        Returns:
            AI insight extraction results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_research_insights_enabled:
                return {"error": "AI insight extraction not available", "success": False}
            
            insight_types = insight_types or [
                "key_findings", "methodology", "conclusions", "implications", "gaps"
            ]
            
            # Extract structured insights using AI
            insights = await self.service_manager.extract_structured_data(
                text=content,
                data_types=insight_types,
                output_format="json"
            )
            
            if insights.get("success"):
                self.metrics["ai_insights_extracted"] += 1
            
            return insights
            
        except Exception as e:
            self.logger.error(f"AI insight extraction failed: {e}")
            return {"error": str(e), "success": False}
    
    async def calculate_ai_relevance_score(
        self,
        content: str,
        research_query: str,
        research_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate content relevance using AI.
        
        Args:
            content: Content to score
            research_query: Research query to match against
            research_context: Additional context for relevance scoring
            
        Returns:
            AI relevance scoring results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_relevance_scoring_enabled:
                return {"error": "AI relevance scoring not available", "success": False}
            
            # Create a prompt for relevance scoring
            scoring_prompt = f"""
            Rate the relevance of the following content to the research query on a scale of 0-10.
            
            Research Query: {research_query}
            
            {f"Research Context: {research_context}" if research_context else ""}
            
            Content to Rate:
            {content[:1000]}...
            
            Provide your rating and reasoning in JSON format:
            {{
                "relevance_score": <score_0_to_10>,
                "reasoning": "<explanation>",
                "key_matches": ["<relevant_point_1>", "<relevant_point_2>"],
                "confidence": <confidence_0_to_1>
            }}
            """
            
            # Use AI chat to get relevance scoring
            response = await self.service_manager.chat_with_ai(
                message=scoring_prompt,
                system_prompt="You are an expert research analyst who evaluates content relevance."
            )
            
            if response.get("success"):
                self.metrics["ai_relevance_scores"] += 1
                
                # Try to parse JSON from response
                try:
                    import json
                    import re
                    
                    json_match = re.search(r'\{.*\}', response["content"], re.DOTALL)
                    if json_match:
                        scoring_data = json.loads(json_match.group())
                        return {
                            "relevance_score": scoring_data.get("relevance_score", 5),
                            "reasoning": scoring_data.get("reasoning", ""),
                            "key_matches": scoring_data.get("key_matches", []),
                            "confidence": scoring_data.get("confidence", 0.5),
                            "success": True
                        }
                except Exception:
                    pass
                
                # Fallback: extract score from text
                score_match = re.search(r'(\d+(?:\.\d+)?)/10|\b(\d+(?:\.\d+)?)\b', response["content"])
                if score_match:
                    score = float(score_match.group(1) or score_match.group(2))
                    return {
                        "relevance_score": min(score, 10),
                        "reasoning": response["content"],
                        "key_matches": [],
                        "confidence": 0.7,
                        "success": True
                    }
            
            return {"error": "Failed to parse AI relevance score", "success": False}
            
        except Exception as e:
            self.logger.error(f"AI relevance scoring failed: {e}")
            return {"error": str(e), "success": False}
    
    async def fact_check_content_with_ai(self, content: str) -> Dict[str, Any]:
        """Fact-check research content using AI.
        
        Args:
            content: Content to fact-check
            
        Returns:
            AI fact-checking results
        """
        try:
            if not self.ai_enabled or not self.service_manager or not self.ai_fact_checking_enabled:
                return {"error": "AI fact-checking not available", "success": False}
            
            fact_check_prompt = f"""
            Analyze the following research content for factual accuracy and credibility.
            Identify any claims that may need verification and provide your assessment.
            
            Content:
            {content[:1500]}...
            
            Provide your analysis in JSON format:
            {{
                "overall_credibility": "<high/medium/low>",
                "factual_claims": [
                    {{
                        "claim": "<claim_text>",
                        "credibility": "<high/medium/low>",
                        "verification_needed": <true/false>,
                        "notes": "<additional_notes>"
                    }}
                ],
                "red_flags": ["<flag_1>", "<flag_2>"],
                "recommendations": ["<recommendation_1>", "<recommendation_2>"]
            }}
            """
            
            response = await self.service_manager.chat_with_ai(
                message=fact_check_prompt,
                system_prompt="You are a fact-checking expert who evaluates content credibility."
            )
            
            if response.get("success"):
                # Parse fact-check results
                try:
                    import json
                    import re
                    
                    json_match = re.search(r'\{.*\}', response["content"], re.DOTALL)
                    if json_match:
                        fact_check_data = json.loads(json_match.group())
                        fact_check_data["success"] = True
                        return fact_check_data
                except Exception:
                    pass
                
                return {
                    "overall_credibility": "medium",
                    "factual_claims": [],
                    "red_flags": [],
                    "recommendations": ["Manual fact-checking recommended"],
                    "raw_response": response["content"],
                    "success": True
                }
            
            return {"error": "AI fact-checking failed", "success": False}
            
        except Exception as e:
            self.logger.error(f"AI fact-checking failed: {e}")
            return {"error": str(e), "success": False}
    
    async def generate_research_report_with_ai(
        self,
        research_results: List[Dict[str, Any]],
        research_query: str,
        report_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Generate a research report using AI.
        
        Args:
            research_results: List of research results
            research_query: Original research query
            report_type: Type of report (comprehensive, summary, executive)
            
        Returns:
            AI-generated research report
        """
        try:
            if not self.ai_enabled or not self.service_manager:
                return {"error": "AI report generation not available", "success": False}
            
            # Prepare research data for report generation
            content_summaries = []
            for result in research_results[:10]:  # Limit to top 10 results
                summary = f"Source: {result.get('url', 'Unknown')}\n"
                summary += f"Title: {result.get('title', 'No title')}\n"
                summary += f"Content: {result.get('content', '')[:500]}...\n"
                content_summaries.append(summary)
            
            combined_content = "\n\n---\n\n".join(content_summaries)
            
            # Generate report using AI
            if report_type == "executive":
                report_prompt = f"""
                Generate an executive summary report based on the research query and findings below.
                Focus on key insights, actionable recommendations, and strategic implications.
                
                Research Query: {research_query}
                
                Research Findings:
                {combined_content}
                
                Format as an executive summary with sections: Overview, Key Findings, Recommendations, Conclusion.
                """
            elif report_type == "summary":
                report_prompt = f"""
                Create a concise research summary based on the query and findings below.
                Focus on the most important points and conclusions.
                
                Research Query: {research_query}
                
                Research Findings:
                {combined_content}
                
                Format as a structured summary with clear bullet points.
                """
            else:  # comprehensive
                report_prompt = f"""
                Generate a comprehensive research report based on the query and findings below.
                Include detailed analysis, multiple perspectives, and thorough documentation.
                
                Research Query: {research_query}
                
                Research Findings:
                {combined_content}
                
                Format as a detailed report with sections: Introduction, Methodology, Findings, Analysis, Conclusions, References.
                """
            
            response = await self.service_manager.chat_with_ai(
                message=report_prompt,
                system_prompt="You are a research analyst who creates comprehensive, well-structured reports."
            )
            
            if response.get("success"):
                return {
                    "report": response["content"],
                    "report_type": report_type,
                    "research_query": research_query,
                    "sources_count": len(research_results),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "processing_time": response.get("processing_time", 0),
                    "success": True
                }
            
            return {"error": "AI report generation failed", "success": False}
            
        except Exception as e:
            self.logger.error(f"AI report generation failed: {e}")
            return {"error": str(e), "success": False}
    
    async def enhance_research_with_ai(
        self,
        research_results: List[Dict[str, Any]],
        research_query: str
    ) -> List[Dict[str, Any]]:
        """Enhance research results with AI analysis.
        
        Args:
            research_results: Original research results
            research_query: Research query for context
            
        Returns:
            Enhanced research results with AI insights
        """
        try:
            enhanced_results = []
            
            for result in research_results:
                enhanced_result = result.copy()
                
                # Add AI summarization
                if self.ai_summarization_enabled and result.get("content"):
                    summary = await self.summarize_research_content_with_ai(
                        result["content"],
                        summary_type="bullet_points",
                        max_length=150
                    )
                    if summary.get("success"):
                        enhanced_result["ai_summary"] = summary["summary"]
                
                # Add AI relevance scoring
                if self.ai_relevance_scoring_enabled and result.get("content"):
                    relevance = await self.calculate_ai_relevance_score(
                        result["content"],
                        research_query
                    )
                    if relevance.get("success"):
                        enhanced_result["ai_relevance_score"] = relevance["relevance_score"]
                        enhanced_result["ai_relevance_reasoning"] = relevance["reasoning"]
                
                # Add AI content analysis
                if self.ai_content_analysis_enabled and result.get("content"):
                    analysis = await self.analyze_content_with_ai(result["content"])
                    if analysis.get("success"):
                        enhanced_result["ai_analysis"] = {
                            "key_findings": analysis.get("key_findings", []),
                            "credibility_score": analysis.get("credibility_score", 5),
                            "methodology": analysis.get("methodology", ""),
                        }
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"AI research enhancement failed: {e}")
            return research_results  # Return original results on error