"""
Rate limiting functionality for the Autonomous Agent System.

This module provides rate limiting capabilities to prevent abuse and
ensure fair usage of the system resources.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import redis.asyncio as redis
from functools import wraps


logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests: int  # Number of requests
    window: int  # Time window in seconds
    burst: int = 0  # Burst allowance
    key_prefix: str = ""  # Key prefix for Redis


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None


class RateLimiter:
    """Rate limiter with multiple strategies."""
    
    def __init__(self, redis_url: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.redis_client = None
        self.local_cache = defaultdict(lambda: defaultdict(deque))
        self.local_cache_lock = asyncio.Lock()
        
        # Default rate limit rules
        self.default_rules = {
            'global': RateLimitRule(requests=1000, window=3600, burst=100),  # 1000/hour
            'api': RateLimitRule(requests=100, window=60, burst=10),  # 100/minute
            'auth': RateLimitRule(requests=5, window=300, burst=2),  # 5/5minutes
            'upload': RateLimitRule(requests=10, window=300, burst=1),  # 10/5minutes
        }
        
        # Custom rules from config
        if 'rules' in self.config:
            self.default_rules.update(self.config['rules'])
        
        # Initialize Redis client if URL provided
        if redis_url:
            self._init_redis(redis_url)
    
    def _init_redis(self, redis_url: str) -> None:
        """Initialize Redis client."""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            logger.info("Redis rate limiter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            logger.info("Falling back to in-memory rate limiting")
    
    async def check_rate_limit(self, 
                             identifier: str, 
                             endpoint: str, 
                             rule_name: str = 'api') -> bool:
        """Check if request is within rate limit."""
        try:
            rule = self.default_rules.get(rule_name)
            if not rule:
                logger.warning(f"Unknown rate limit rule: {rule_name}")
                return True
            
            # Use Redis if available, otherwise use local cache
            if self.redis_client:
                result = await self._check_redis_rate_limit(identifier, endpoint, rule)
            else:
                result = await self._check_local_rate_limit(identifier, endpoint, rule)
            
            return result.allowed
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return True
    
    async def get_rate_limit_status(self, 
                                   identifier: str, 
                                   endpoint: str, 
                                   rule_name: str = 'api') -> RateLimitResult:
        """Get detailed rate limit status."""
        try:
            rule = self.default_rules.get(rule_name)
            if not rule:
                return RateLimitResult(allowed=True, remaining=0, reset_time=0)
            
            if self.redis_client:
                return await self._check_redis_rate_limit(identifier, endpoint, rule)
            else:
                return await self._check_local_rate_limit(identifier, endpoint, rule)
                
        except Exception as e:
            logger.error(f"Rate limit status check failed: {e}")
            return RateLimitResult(allowed=True, remaining=0, reset_time=0)
    
    async def _check_redis_rate_limit(self, 
                                    identifier: str, 
                                    endpoint: str, 
                                    rule: RateLimitRule) -> RateLimitResult:
        """Check rate limit using Redis."""
        key = f"rate_limit:{rule.key_prefix}:{identifier}:{endpoint}"
        now = time.time()
        window_start = now - rule.window
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Execute pipeline
        results = await pipe.execute()
        current_count = results[1]
        
        # Check if limit exceeded
        if current_count >= rule.requests:
            # Get oldest entry to calculate reset time
            oldest_entries = await self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_entries:
                reset_time = oldest_entries[0][1] + rule.window
                retry_after = int(reset_time - now)
            else:
                reset_time = now + rule.window
                retry_after = rule.window
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after
            )
        
        # Add current request
        pipe = self.redis_client.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, rule.window)
        await pipe.execute()
        
        return RateLimitResult(
            allowed=True,
            remaining=rule.requests - current_count - 1,
            reset_time=now + rule.window
        )
    
    async def _check_local_rate_limit(self, 
                                    identifier: str, 
                                    endpoint: str, 
                                    rule: RateLimitRule) -> RateLimitResult:
        """Check rate limit using local cache."""
        async with self.local_cache_lock:
            key = f"{identifier}:{endpoint}"
            now = time.time()
            window_start = now - rule.window
            
            # Get or create request queue
            request_queue = self.local_cache[key]['requests']
            
            # Remove old entries
            while request_queue and request_queue[0] < window_start:
                request_queue.popleft()
            
            # Check if limit exceeded
            if len(request_queue) >= rule.requests:
                # Calculate reset time
                reset_time = request_queue[0] + rule.window
                retry_after = int(reset_time - now)
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after
                )
            
            # Add current request
            request_queue.append(now)
            
            return RateLimitResult(
                allowed=True,
                remaining=rule.requests - len(request_queue),
                reset_time=now + rule.window
            )
    
    async def clear_rate_limit(self, identifier: str, endpoint: str) -> None:
        """Clear rate limit for a specific identifier and endpoint."""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(f"rate_limit:*:{identifier}:{endpoint}")
                if keys:
                    await self.redis_client.delete(*keys)
            else:
                async with self.local_cache_lock:
                    key = f"{identifier}:{endpoint}"
                    if key in self.local_cache:
                        del self.local_cache[key]
                        
        except Exception as e:
            logger.error(f"Failed to clear rate limit: {e}")
    
    async def get_rate_limit_info(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit information for an identifier."""
        try:
            info = {}
            
            if self.redis_client:
                keys = await self.redis_client.keys(f"rate_limit:*:{identifier}:*")
                for key in keys:
                    count = await self.redis_client.zcard(key)
                    ttl = await self.redis_client.ttl(key)
                    info[key] = {'count': count, 'ttl': ttl}
            else:
                async with self.local_cache_lock:
                    for key, data in self.local_cache.items():
                        if key.startswith(f"{identifier}:"):
                            info[key] = {'count': len(data['requests'])}
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get rate limit info: {e}")
            return {}
    
    async def cleanup_expired(self) -> None:
        """Clean up expired rate limit entries."""
        try:
            if self.redis_client:
                # Redis handles expiration automatically
                pass
            else:
                async with self.local_cache_lock:
                    now = time.time()
                    expired_keys = []
                    
                    for key, data in self.local_cache.items():
                        request_queue = data['requests']
                        # Remove old entries
                        while request_queue and request_queue[0] < now - 3600:  # 1 hour
                            request_queue.popleft()
                        
                        # Mark empty queues for deletion
                        if not request_queue:
                            expired_keys.append(key)
                    
                    # Delete expired keys
                    for key in expired_keys:
                        del self.local_cache[key]
                        
        except Exception as e:
            logger.error(f"Failed to cleanup expired rate limits: {e}")


def rate_limit(rule_name: str = 'api', 
               identifier_func: Optional[callable] = None,
               endpoint_func: Optional[callable] = None):
    """Decorator for rate limiting functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get rate limiter instance
            rate_limiter = kwargs.get('rate_limiter')
            if not rate_limiter:
                # Try to get from global context or create new one
                rate_limiter = RateLimiter()
            
            # Determine identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = 'default'
            
            # Determine endpoint
            if endpoint_func:
                endpoint = endpoint_func(*args, **kwargs)
            else:
                endpoint = func.__name__
            
            # Check rate limit
            if not await rate_limiter.check_rate_limit(identifier, endpoint, rule_name):
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_windows = defaultdict(list)
        self.local_lock = asyncio.Lock()
    
    async def is_allowed(self, 
                        key: str, 
                        limit: int, 
                        window: int, 
                        current_time: Optional[float] = None) -> Tuple[bool, int]:
        """Check if request is allowed under sliding window."""
        if current_time is None:
            current_time = time.time()
        
        window_start = current_time - window
        
        if self.redis_client:
            return await self._redis_sliding_window(key, limit, window, current_time, window_start)
        else:
            return await self._local_sliding_window(key, limit, window, current_time, window_start)
    
    async def _redis_sliding_window(self, 
                                   key: str, 
                                   limit: int, 
                                   window: int, 
                                   current_time: float, 
                                   window_start: float) -> Tuple[bool, int]:
        """Redis-based sliding window implementation."""
        # Use Lua script for atomic operations
        lua_script = """
            local key = KEYS[1]
            local window_start = ARGV[1]
            local current_time = ARGV[2]
            local limit = tonumber(ARGV[3])
            
            -- Remove old entries
            redis.call('zremrangebyscore', key, 0, window_start)
            
            -- Count current requests
            local current_count = redis.call('zcard', key)
            
            if current_count < limit then
                -- Add current request
                redis.call('zadd', key, current_time, current_time)
                redis.call('expire', key, 3600)  -- 1 hour expiration
                return {1, limit - current_count - 1}
            else
                return {0, 0}
            end
        """
        
        result = await self.redis_client.eval(
            lua_script, 
            1, 
            key, 
            window_start, 
            current_time, 
            limit
        )
        
        return bool(result[0]), int(result[1])
    
    async def _local_sliding_window(self, 
                                   key: str, 
                                   limit: int, 
                                   window: int, 
                                   current_time: float, 
                                   window_start: float) -> Tuple[bool, int]:
        """Local memory-based sliding window implementation."""
        async with self.local_lock:
            # Get or create window
            window_requests = self.local_windows[key]
            
            # Remove old entries
            window_requests[:] = [t for t in window_requests if t > window_start]
            
            # Check if limit exceeded
            if len(window_requests) >= limit:
                return False, 0
            
            # Add current request
            window_requests.append(current_time)
            
            return True, limit - len(window_requests)


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_buckets = defaultdict(dict)
        self.local_lock = asyncio.Lock()
    
    async def is_allowed(self, 
                        key: str, 
                        capacity: int, 
                        refill_rate: float, 
                        tokens_requested: int = 1,
                        current_time: Optional[float] = None) -> Tuple[bool, int]:
        """Check if request is allowed under token bucket."""
        if current_time is None:
            current_time = time.time()
        
        if self.redis_client:
            return await self._redis_token_bucket(key, capacity, refill_rate, tokens_requested, current_time)
        else:
            return await self._local_token_bucket(key, capacity, refill_rate, tokens_requested, current_time)
    
    async def _redis_token_bucket(self, 
                                 key: str, 
                                 capacity: int, 
                                 refill_rate: float, 
                                 tokens_requested: int, 
                                 current_time: float) -> Tuple[bool, int]:
        """Redis-based token bucket implementation."""
        lua_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local tokens_requested = tonumber(ARGV[3])
            local current_time = tonumber(ARGV[4])
            
            -- Get current bucket state
            local bucket = redis.call('hmget', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or capacity
            local last_refill = tonumber(bucket[2]) or current_time
            
            -- Calculate tokens to add
            local time_passed = current_time - last_refill
            local tokens_to_add = math.floor(time_passed * refill_rate)
            tokens = math.min(capacity, tokens + tokens_to_add)
            
            if tokens >= tokens_requested then
                -- Consume tokens
                tokens = tokens - tokens_requested
                redis.call('hmset', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('expire', key, 3600)  -- 1 hour expiration
                return {1, tokens}
            else
                -- Update last refill time even if request is denied
                redis.call('hmset', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('expire', key, 3600)
                return {0, tokens}
            end
        """
        
        result = await self.redis_client.eval(
            lua_script, 
            1, 
            key, 
            capacity, 
            refill_rate, 
            tokens_requested, 
            current_time
        )
        
        return bool(result[0]), int(result[1])
    
    async def _local_token_bucket(self, 
                                 key: str, 
                                 capacity: int, 
                                 refill_rate: float, 
                                 tokens_requested: int, 
                                 current_time: float) -> Tuple[bool, int]:
        """Local memory-based token bucket implementation."""
        async with self.local_lock:
            # Get or create bucket
            bucket = self.local_buckets[key]
            if not bucket:
                bucket = {'tokens': capacity, 'last_refill': current_time}
                self.local_buckets[key] = bucket
            
            # Calculate tokens to add
            time_passed = current_time - bucket['last_refill']
            tokens_to_add = int(time_passed * refill_rate)
            bucket['tokens'] = min(capacity, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = current_time
            
            # Check if enough tokens
            if bucket['tokens'] >= tokens_requested:
                bucket['tokens'] -= tokens_requested
                return True, bucket['tokens']
            else:
                return False, bucket['tokens']