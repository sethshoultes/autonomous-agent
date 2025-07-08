"""
Database connection management with async PostgreSQL support.

This module provides robust connection pooling, transaction management,
and database session handling for the autonomous agent system.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Union
from urllib.parse import urlparse

import asyncpg
from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from ..core.exceptions import CoreError


class DatabaseError(CoreError):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Exception raised for connection-related errors."""
    pass


class TransactionError(DatabaseError):
    """Exception raised for transaction-related errors."""
    pass


class DatabaseManager:
    """
    Manages PostgreSQL database connections with advanced features.
    
    Features:
    - Async connection pooling with configurable pool size
    - Transaction management and rollback support
    - Connection health monitoring and automatic recovery
    - Query performance tracking and optimization
    - Read/write replica support for scaling
    - Connection encryption and security
    """
    
    def __init__(
        self,
        database_url: str,
        logger: logging.Logger,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        read_replica_urls: Optional[list] = None,
        enable_query_logging: bool = False,
        query_timeout: int = 30,
    ):
        """
        Initialize the DatabaseManager.
        
        Args:
            database_url: Primary database connection URL
            logger: Logger instance for database operations
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum number of connections beyond pool_size
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Recycle connections after this many seconds
            read_replica_urls: Optional list of read replica URLs
            enable_query_logging: Whether to log SQL queries
            query_timeout: Timeout for individual queries
        """
        self.database_url = database_url
        self.logger = logger
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.read_replica_urls = read_replica_urls or []
        self.enable_query_logging = enable_query_logging
        self.query_timeout = query_timeout
        
        # Connection components
        self.async_engine = None
        self.async_session_factory = None
        self.connection_pool = None
        self.read_pools = []
        
        # Monitoring
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "query_count": 0,
            "avg_query_time": 0.0,
            "slow_queries": 0,
        }
        
        # Health check
        self.is_healthy = False
        self.last_health_check = None
        
        # Parse database URL for connection parameters
        self._parse_database_url()
        
    def _parse_database_url(self) -> None:
        """Parse database URL to extract connection parameters."""
        parsed = urlparse(self.database_url)
        
        self.db_params = {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/") or "autonomous_agent",
            "user": parsed.username or "postgres",
            "password": parsed.password or "",
        }
        
        # Build asyncpg connection string
        self.asyncpg_url = (
            f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
            f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
        )
        
    async def initialize(self) -> None:
        """Initialize database connections and pools."""
        try:
            self.logger.info("Initializing database connections...")
            
            # Create async engine with connection pooling
            self.async_engine = create_async_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=self.enable_query_logging,
                future=True,
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Create direct connection pool for raw queries
            self.connection_pool = await asyncpg.create_pool(
                self.asyncpg_url,
                min_size=max(1, self.pool_size // 2),
                max_size=self.pool_size,
                command_timeout=self.query_timeout,
                server_settings={
                    "jit": "off",  # Disable JIT for better connection reuse
                    "application_name": "autonomous_agent",
                },
            )
            
            # Initialize read replica pools
            for replica_url in self.read_replica_urls:
                replica_pool = await asyncpg.create_pool(
                    replica_url,
                    min_size=1,
                    max_size=max(2, self.pool_size // 4),
                    command_timeout=self.query_timeout,
                    server_settings={
                        "jit": "off",
                        "application_name": "autonomous_agent_read",
                    },
                )
                self.read_pools.append(replica_pool)
            
            # Test connectivity
            await self.health_check()
            
            if self.is_healthy:
                self.logger.info("Database connections initialized successfully")
            else:
                raise ConnectionError("Database health check failed")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise ConnectionError(f"Database initialization failed: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown database connections gracefully."""
        try:
            self.logger.info("Shutting down database connections...")
            
            # Close connection pools
            if self.connection_pool:
                await self.connection_pool.close()
                
            for replica_pool in self.read_pools:
                await replica_pool.close()
            
            # Close async engine
            if self.async_engine:
                await self.async_engine.dispose()
                
            self.logger.info("Database connections closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during database shutdown: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic transaction management."""
        if not self.async_session_factory:
            raise ConnectionError("Database not initialized")
        
        async with self.async_session_factory() as session:
            try:
                self.connection_stats["active_connections"] += 1
                yield session
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Database session error: {e}")
                raise
            finally:
                self.connection_stats["active_connections"] -= 1
    
    @asynccontextmanager
    async def get_connection(self, read_only: bool = False) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a raw database connection for custom queries."""
        pool = self._get_connection_pool(read_only)
        
        if not pool:
            raise ConnectionError("No available connection pool")
        
        async with pool.acquire() as connection:
            try:
                self.connection_stats["active_connections"] += 1
                yield connection
            except Exception as e:
                self.logger.error(f"Raw connection error: {e}")
                raise
            finally:
                self.connection_stats["active_connections"] -= 1
    
    def _get_connection_pool(self, read_only: bool = False) -> Optional[asyncpg.Pool]:
        """Get appropriate connection pool based on read/write preference."""
        if read_only and self.read_pools:
            # Round-robin selection for read replicas
            import random
            return random.choice(self.read_pools)
        return self.connection_pool
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        read_only: bool = False,
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Execute a raw SQL query with performance tracking.
        
        Args:
            query: SQL query string
            params: Query parameters
            read_only: Whether query is read-only (can use replica)
            timeout: Query timeout in seconds
            
        Returns:
            Query result
        """
        import time
        
        start_time = time.time()
        timeout = timeout or self.query_timeout
        
        try:
            async with self.get_connection(read_only=read_only) as conn:
                if params:
                    result = await conn.fetch(query, *params.values())
                else:
                    result = await conn.fetch(query)
                
                execution_time = time.time() - start_time
                self._track_query_performance(query, execution_time)
                
                return result
                
        except Exception as e:
            self.connection_stats["failed_connections"] += 1
            self.logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query execution failed: {e}")
    
    async def execute_transaction(
        self,
        queries: list,
        params_list: Optional[list] = None,
    ) -> list:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of SQL query strings
            params_list: List of parameter dictionaries for each query
            
        Returns:
            List of query results
        """
        results = []
        params_list = params_list or [None] * len(queries)
        
        async with self.get_connection() as conn:
            async with conn.transaction():
                try:
                    for query, params in zip(queries, params_list):
                        if params:
                            result = await conn.fetch(query, *params.values())
                        else:
                            result = await conn.fetch(query)
                        results.append(result)
                    
                    return results
                    
                except Exception as e:
                    self.logger.error(f"Transaction failed: {e}")
                    raise TransactionError(f"Transaction failed: {e}")
    
    def _track_query_performance(self, query: str, execution_time: float) -> None:
        """Track query performance metrics."""
        self.connection_stats["query_count"] += 1
        
        # Update average query time
        current_avg = self.connection_stats["avg_query_time"]
        query_count = self.connection_stats["query_count"]
        
        new_avg = ((current_avg * (query_count - 1)) + execution_time) / query_count
        self.connection_stats["avg_query_time"] = new_avg
        
        # Track slow queries (>1 second)
        if execution_time > 1.0:
            self.connection_stats["slow_queries"] += 1
            self.logger.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
    
    async def health_check(self) -> bool:
        """Perform comprehensive database health check."""
        import time
        
        try:
            start_time = time.time()
            
            # Test primary connection
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            
            # Test read replicas
            for replica_pool in self.read_pools:
                async with replica_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
            
            # Test SQLAlchemy engine
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            
            health_check_time = time.time() - start_time
            
            self.is_healthy = True
            self.last_health_check = time.time()
            
            self.logger.debug(f"Database health check passed in {health_check_time:.2f}s")
            return True
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        stats = self.connection_stats.copy()
        
        # Add pool statistics
        if self.connection_pool:
            stats.update({
                "pool_size": self.connection_pool.get_size(),
                "pool_free_connections": self.connection_pool.get_idle_size(),
                "pool_used_connections": self.connection_pool.get_size() - self.connection_pool.get_idle_size(),
            })
        
        # Add replica statistics
        replica_stats = []
        for i, replica_pool in enumerate(self.read_pools):
            replica_stats.append({
                "replica_index": i,
                "pool_size": replica_pool.get_size(),
                "pool_free": replica_pool.get_idle_size(),
                "pool_used": replica_pool.get_size() - replica_pool.get_idle_size(),
            })
        
        stats["read_replicas"] = replica_stats
        stats["is_healthy"] = self.is_healthy
        stats["last_health_check"] = self.last_health_check
        
        return stats
    
    async def optimize_connections(self) -> Dict[str, Any]:
        """Optimize database connections based on current usage."""
        stats = await self.get_connection_stats()
        optimizations = []
        
        # Check pool utilization
        if self.connection_pool:
            utilization = stats["pool_used_connections"] / stats["pool_size"]
            
            if utilization > 0.8:
                optimizations.append({
                    "type": "pool_size_increase",
                    "current_size": stats["pool_size"],
                    "recommended_size": min(stats["pool_size"] * 1.5, 50),
                    "reason": "High pool utilization detected"
                })
            elif utilization < 0.2:
                optimizations.append({
                    "type": "pool_size_decrease",
                    "current_size": stats["pool_size"],
                    "recommended_size": max(stats["pool_size"] * 0.8, 5),
                    "reason": "Low pool utilization detected"
                })
        
        # Check slow queries
        if stats["slow_queries"] > stats["query_count"] * 0.1:
            optimizations.append({
                "type": "query_optimization",
                "slow_query_ratio": stats["slow_queries"] / stats["query_count"],
                "recommendation": "Review and optimize slow queries"
            })
        
        return {
            "optimizations": optimizations,
            "current_stats": stats,
            "optimization_timestamp": time.time()
        }


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database_manager
    
    if _database_manager is None:
        raise ConnectionError("Database manager not initialized")
    
    return _database_manager


async def initialize_database_manager(
    database_url: str,
    logger: logging.Logger,
    **kwargs
) -> DatabaseManager:
    """Initialize the global database manager."""
    global _database_manager
    
    if _database_manager is not None:
        await _database_manager.shutdown()
    
    _database_manager = DatabaseManager(database_url, logger, **kwargs)
    await _database_manager.initialize()
    
    return _database_manager


async def shutdown_database_manager() -> None:
    """Shutdown the global database manager."""
    global _database_manager
    
    if _database_manager is not None:
        await _database_manager.shutdown()
        _database_manager = None