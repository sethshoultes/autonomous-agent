"""
Health monitoring and checking system for the Autonomous Agent System.

This module provides comprehensive health monitoring, status checks,
and system diagnostics.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import psutil
import aiohttp
import asyncpg
import redis.asyncio as redis


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "duration": self.duration
        }


class HealthChecker:
    """Comprehensive health monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize health checker.
        
        Args:
            config: Health checker configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval = self.config.get('check_interval', 30)
        self.running = False
        
        # Register built-in checks
        self._register_builtin_checks()
    
    def _register_builtin_checks(self):
        """Register built-in health checks."""
        self.register_check("system_cpu", self._check_system_cpu)
        self.register_check("system_memory", self._check_system_memory)
        self.register_check("system_disk", self._check_system_disk)
        self.register_check("database", self._check_database)
        self.register_check("redis", self._check_redis)
        self.register_check("ollama", self._check_ollama)
        self.register_check("api_endpoints", self._check_api_endpoints)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.debug(f"Registered health check: {name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check function."""
        if name in self.checks:
            del self.checks[name]
            if name in self.last_results:
                del self.last_results[name]
            self.logger.debug(f"Unregistered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        try:
            result = await self.checks[name]()
            result.duration = time.time() - start_time
            self.last_results[name] = result
            return result
        except Exception as e:
            duration = time.time() - start_time
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration=duration,
                details={"error": str(e)}
            )
            self.last_results[name] = result
            self.logger.error(f"Health check '{name}' failed: {e}")
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name in self.checks:
            task = asyncio.create_task(self.run_check(name))
            tasks.append((name, task))
        
        # Wait for all checks to complete
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        return results
    
    async def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health status."""
        results = await self.run_all_checks()
        
        # Determine overall status
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
            message = "System has critical issues"
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
            message = "System has warnings"
        elif HealthStatus.HEALTHY in statuses:
            overall_status = HealthStatus.HEALTHY
            message = "System is healthy"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "System status unknown"
        
        # Calculate statistics
        total_checks = len(results)
        healthy_checks = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        warning_checks = sum(1 for r in results.values() if r.status == HealthStatus.WARNING)
        critical_checks = sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL)
        
        return HealthCheckResult(
            name="overall",
            status=overall_status,
            message=message,
            details={
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "warning_checks": warning_checks,
                "critical_checks": critical_checks,
                "checks": {name: result.to_dict() for name, result in results.items()}
            }
        )
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.running = True
        self.logger.info("Started health monitoring")
        
        while self.running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        self.logger.info("Stopped health monitoring")
    
    async def _check_system_cpu(self) -> HealthCheckResult:
        """Check system CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent}%"
            elif cpu_percent > 75:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent}%"
            
            return HealthCheckResult(
                name="system_cpu",
                status=status,
                message=message,
                details={"cpu_percent": cpu_percent}
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_cpu",
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check CPU: {str(e)}"
            )
    
    async def _check_system_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory.percent}%"
            elif memory.percent > 80:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory.percent}%"
            
            return HealthCheckResult(
                name="system_memory",
                status=status,
                message=message,
                details={
                    "memory_percent": memory.percent,
                    "memory_used": memory.used,
                    "memory_total": memory.total
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_memory",
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check memory: {str(e)}"
            )
    
    async def _check_system_disk(self) -> HealthCheckResult:
        """Check system disk usage."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent > 80:
                status = HealthStatus.WARNING
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_disk",
                status=status,
                message=message,
                details={
                    "disk_percent": disk_percent,
                    "disk_used": disk.used,
                    "disk_total": disk.total
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_disk",
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check disk: {str(e)}"
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            database_url = self.config.get('database_url', 'postgresql://agent:password@localhost:5432/autonomous_agent')
            
            conn = await asyncpg.connect(database_url)
            
            # Test query
            result = await conn.fetchval("SELECT 1")
            
            await conn.close()
            
            if result == 1:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection healthy",
                    details={"connection": "successful"}
                )
            else:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.CRITICAL,
                    message="Database query failed",
                    details={"connection": "failed"}
                )
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379/0')
            
            r = redis.from_url(redis_url)
            
            # Test ping
            result = await r.ping()
            
            await r.close()
            
            if result:
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    message="Redis connection healthy",
                    details={"connection": "successful"}
                )
            else:
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.CRITICAL,
                    message="Redis ping failed",
                    details={"connection": "failed"}
                )
        except Exception as e:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _check_ollama(self) -> HealthCheckResult:
        """Check Ollama service availability."""
        try:
            ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            name="ollama",
                            status=HealthStatus.HEALTHY,
                            message="Ollama service healthy",
                            details={"models": len(data.get('models', []))}
                        )
                    else:
                        return HealthCheckResult(
                            name="ollama",
                            status=HealthStatus.CRITICAL,
                            message=f"Ollama service returned status {response.status}",
                            details={"status_code": response.status}
                        )
        except Exception as e:
            return HealthCheckResult(
                name="ollama",
                status=HealthStatus.CRITICAL,
                message=f"Ollama service unavailable: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _check_api_endpoints(self) -> HealthCheckResult:
        """Check API endpoints availability."""
        try:
            base_url = self.config.get('api_base_url', 'http://localhost:8000')
            
            endpoints = [
                '/health',
                '/metrics',
                '/'
            ]
            
            results = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints:
                    try:
                        async with session.get(f"{base_url}{endpoint}", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            results[endpoint] = {
                                "status_code": response.status,
                                "available": response.status < 500
                            }
                    except Exception as e:
                        results[endpoint] = {
                            "status_code": None,
                            "available": False,
                            "error": str(e)
                        }
            
            # Check if all endpoints are available
            available_endpoints = sum(1 for r in results.values() if r["available"])
            total_endpoints = len(endpoints)
            
            if available_endpoints == total_endpoints:
                status = HealthStatus.HEALTHY
                message = "All API endpoints available"
            elif available_endpoints > 0:
                status = HealthStatus.WARNING
                message = f"{available_endpoints}/{total_endpoints} API endpoints available"
            else:
                status = HealthStatus.CRITICAL
                message = "No API endpoints available"
            
            return HealthCheckResult(
                name="api_endpoints",
                status=status,
                message=message,
                details={
                    "endpoints": results,
                    "available_count": available_endpoints,
                    "total_count": total_endpoints
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="api_endpoints",
                status=HealthStatus.CRITICAL,
                message=f"API endpoints check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of health check results."""
        if not self.last_results:
            return {
                "status": "unknown",
                "message": "No health checks have been run",
                "checks": {}
            }
        
        # Calculate overall status
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = "critical"
            message = "System has critical issues"
        elif HealthStatus.WARNING in statuses:
            overall_status = "warning"
            message = "System has warnings"
        elif HealthStatus.HEALTHY in statuses:
            overall_status = "healthy"
            message = "System is healthy"
        else:
            overall_status = "unknown"
            message = "System status unknown"
        
        return {
            "status": overall_status,
            "message": message,
            "last_check": max(r.timestamp for r in self.last_results.values()).isoformat(),
            "checks": {name: result.to_dict() for name, result in self.last_results.items()}
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def configure_health_checker(config: Dict[str, Any]):
    """Configure the global health checker."""
    global _health_checker
    _health_checker = HealthChecker(config)