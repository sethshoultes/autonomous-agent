"""
AI processing cache for performance optimization.

This module provides caching capabilities for AI processing results
to reduce redundant API calls and improve response times.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import asyncio
import logging


class AICache:
    """
    In-memory cache for AI processing results with TTL support.
    
    Provides fast caching of AI responses to avoid redundant processing
    of similar content and improve overall system performance.
    """
    
    def __init__(
        self,
        default_ttl: int = 3600,  # 1 hour default TTL
        max_size: int = 1000,     # Maximum cache entries
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the AI cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of cache entries
            logger: Optional logger instance
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache storage: key -> (value, timestamp, ttl)
        self._cache: Dict[str, Tuple[Any, float, int]] = {}
        
        # Access tracking for LRU eviction
        self._access_times: Dict[str, float] = {}
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "total_requests": 0
        }
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running yet
            pass
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, timestamp, ttl) in self._cache.items():
            if current_time - timestamp > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _generate_cache_key(self, content: str, operation: str, **kwargs) -> str:
        """Generate cache key for content and operation.
        
        Args:
            content: Content to cache
            operation: AI operation type
            **kwargs: Additional parameters affecting the result
            
        Returns:
            Cache key string
        """
        # Include operation and relevant parameters in key
        key_data = {
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "operation": operation,
            "params": sorted(kwargs.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and access tracking."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        self.stats["size"] = len(self._cache)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_entry(lru_key)
        self.stats["evictions"] += 1
        
        self.logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed maximum size."""
        while len(self._cache) >= self.max_size:
            self._evict_lru()
    
    def get(self, content: str, operation: str, **kwargs) -> Optional[Any]:
        """Get cached result for content and operation.
        
        Args:
            content: Content that was processed
            operation: AI operation type
            **kwargs: Additional parameters that affect the result
            
        Returns:
            Cached result or None if not found/expired
        """
        self.stats["total_requests"] += 1
        
        cache_key = self._generate_cache_key(content, operation, **kwargs)
        current_time = time.time()
        
        if cache_key in self._cache:
            value, timestamp, ttl = self._cache[cache_key]
            
            # Check if entry has expired
            if current_time - timestamp <= ttl:
                # Update access time
                self._access_times[cache_key] = current_time
                self.stats["hits"] += 1
                
                self.logger.debug(f"Cache hit for {operation}: {cache_key}")
                return value
            else:
                # Remove expired entry
                self._remove_entry(cache_key)
        
        self.stats["misses"] += 1
        self.logger.debug(f"Cache miss for {operation}: {cache_key}")
        return None
    
    def set(
        self,
        content: str,
        operation: str,
        result: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """Cache result for content and operation.
        
        Args:
            content: Content that was processed
            operation: AI operation type
            result: Result to cache
            ttl: Time-to-live in seconds (uses default if None)
            **kwargs: Additional parameters that affect the result
        """
        cache_key = self._generate_cache_key(content, operation, **kwargs)
        current_time = time.time()
        ttl = ttl or self.default_ttl
        
        # Ensure we have capacity
        self._ensure_capacity()
        
        # Store the result
        self._cache[cache_key] = (result, current_time, ttl)
        self._access_times[cache_key] = current_time
        self.stats["size"] = len(self._cache)
        
        self.logger.debug(f"Cached result for {operation}: {cache_key} (TTL: {ttl}s)")
    
    def invalidate(self, content: str, operation: str, **kwargs) -> bool:
        """Invalidate cached result for specific content and operation.
        
        Args:
            content: Content to invalidate
            operation: AI operation type
            **kwargs: Additional parameters
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        cache_key = self._generate_cache_key(content, operation, **kwargs)
        
        if cache_key in self._cache:
            self._remove_entry(cache_key)
            self.logger.debug(f"Invalidated cache entry: {cache_key}")
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all cached entries."""
        entry_count = len(self._cache)
        self._cache.clear()
        self._access_times.clear()
        self.stats["size"] = 0
        
        self.logger.info(f"Cleared {entry_count} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = (self.stats["hits"] / self.stats["total_requests"]) if self.stats["total_requests"] > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }
    
    def cleanup(self) -> None:
        """Cleanup cache and stop background tasks."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        self.clear()
        self.logger.info("AI cache cleaned up")


class CachedAIProcessor:
    """
    Wrapper for AI processing with caching capabilities.
    
    Automatically caches AI processing results to improve performance
    and reduce redundant API calls.
    """
    
    def __init__(
        self,
        ai_processor: Any,
        cache: Optional[AICache] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize cached AI processor.
        
        Args:
            ai_processor: AI processor instance to wrap
            cache: Cache instance (creates new one if None)
            logger: Optional logger instance
        """
        self.ai_processor = ai_processor
        self.cache = cache or AICache()
        self.logger = logger or logging.getLogger(__name__)
    
    async def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Cached text summarization.
        
        Args:
            text: Text to summarize
            **kwargs: Additional parameters
            
        Returns:
            Summarization result (cached or fresh)
        """
        # Check cache first
        cached_result = self.cache.get(text, "summarize", **kwargs)
        if cached_result is not None:
            return cached_result
        
        # Get fresh result
        result = await self.ai_processor.summarize_text(text, **kwargs)
        
        # Cache successful results
        if result.get("success"):
            self.cache.set(text, "summarize", result, **kwargs)
        
        return result
    
    async def classify_content(self, content: str, categories: list, **kwargs) -> Dict[str, Any]:
        """Cached content classification.
        
        Args:
            content: Content to classify
            categories: List of categories
            **kwargs: Additional parameters
            
        Returns:
            Classification result (cached or fresh)
        """
        # Include categories in cache key
        cache_params = {**kwargs, "categories": tuple(sorted(categories))}
        
        # Check cache first
        cached_result = self.cache.get(content, "classify", **cache_params)
        if cached_result is not None:
            return cached_result
        
        # Get fresh result
        result = await self.ai_processor.classify_content(content, categories, **kwargs)
        
        # Cache successful results
        if result.get("success"):
            self.cache.set(content, "classify", result, **cache_params)
        
        return result
    
    async def analyze_email(self, email_content: Dict[str, Any]) -> Dict[str, Any]:
        """Cached email analysis.
        
        Args:
            email_content: Email data to analyze
            
        Returns:
            Analysis result (cached or fresh)
        """
        # Create content string for caching
        content_str = json.dumps(email_content, sort_keys=True)
        
        # Check cache first
        cached_result = self.cache.get(content_str, "analyze_email")
        if cached_result is not None:
            return cached_result
        
        # Get fresh result
        result = await self.ai_processor.analyze_email(email_content)
        
        # Cache successful results
        if result.get("success"):
            self.cache.set(content_str, "analyze_email", result)
        
        return result
    
    async def analyze_research_content(
        self,
        content: str,
        research_questions: Optional[list] = None,
        focus_areas: Optional[list] = None
    ) -> Dict[str, Any]:
        """Cached research content analysis.
        
        Args:
            content: Research content to analyze
            research_questions: Specific questions to address
            focus_areas: Areas to focus analysis on
            
        Returns:
            Analysis result (cached or fresh)
        """
        cache_params = {
            "research_questions": tuple(research_questions or []),
            "focus_areas": tuple(focus_areas or [])
        }
        
        # Check cache first
        cached_result = self.cache.get(content, "analyze_research", **cache_params)
        if cached_result is not None:
            return cached_result
        
        # Get fresh result
        result = await self.ai_processor.analyze_research_content(
            content, research_questions, focus_areas
        )
        
        # Cache successful results
        if result.get("success"):
            self.cache.set(content, "analyze_research", result, **cache_params)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
    
    def cleanup(self) -> None:
        """Cleanup cached processor."""
        self.cache.cleanup()