"""
Metrics collection and monitoring for the Autonomous Agent System.

This module provides comprehensive metrics collection, storage, and export
functionality with support for Prometheus and custom metrics.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio

from prometheus_client import Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import REGISTRY


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None
    
    def __post_init__(self):
        # Validate metric name
        if not self.name or not self.name.replace('_', '').replace(':', '').isalnum():
            raise ValueError(f"Invalid metric name: {self.name}")
        
        # Add default buckets for histograms
        if self.type == MetricType.HISTOGRAM and self.buckets is None:
            self.buckets = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]


@dataclass
class MetricValue:
    """A metric value with timestamp and labels."""
    value: Union[float, int, str]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Prometheus registry to use
        """
        self.registry = registry or REGISTRY
        self.metrics: Dict[str, Any] = {}
        self.definitions: Dict[str, MetricDefinition] = {}
        self.values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
        
        # Built-in metrics
        self._register_builtin_metrics()
        
        # Background cleanup
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _register_builtin_metrics(self):
        """Register built-in system metrics."""
        # Application metrics
        self.register_metric(MetricDefinition(
            name="app_requests_total",
            type=MetricType.COUNTER,
            description="Total number of HTTP requests",
            labels=["method", "endpoint", "status_code"]
        ))
        
        self.register_metric(MetricDefinition(
            name="app_request_duration_seconds",
            type=MetricType.HISTOGRAM,
            description="HTTP request duration in seconds",
            labels=["method", "endpoint"],
            unit="seconds"
        ))
        
        self.register_metric(MetricDefinition(
            name="app_active_connections",
            type=MetricType.GAUGE,
            description="Number of active connections",
            unit="connections"
        ))
        
        # Agent metrics
        self.register_metric(MetricDefinition(
            name="agent_tasks_total",
            type=MetricType.COUNTER,
            description="Total number of agent tasks",
            labels=["agent_id", "task_type", "status"]
        ))
        
        self.register_metric(MetricDefinition(
            name="agent_task_duration_seconds",
            type=MetricType.HISTOGRAM,
            description="Agent task duration in seconds",
            labels=["agent_id", "task_type"],
            unit="seconds"
        ))
        
        self.register_metric(MetricDefinition(
            name="agent_active_tasks",
            type=MetricType.GAUGE,
            description="Number of active agent tasks",
            labels=["agent_id"],
            unit="tasks"
        ))
        
        # Database metrics
        self.register_metric(MetricDefinition(
            name="db_connections_active",
            type=MetricType.GAUGE,
            description="Number of active database connections",
            unit="connections"
        ))
        
        self.register_metric(MetricDefinition(
            name="db_query_duration_seconds",
            type=MetricType.HISTOGRAM,
            description="Database query duration in seconds",
            labels=["query_type"],
            unit="seconds"
        ))
        
        self.register_metric(MetricDefinition(
            name="db_errors_total",
            type=MetricType.COUNTER,
            description="Total number of database errors",
            labels=["error_type"]
        ))
        
        # System metrics
        self.register_metric(MetricDefinition(
            name="system_memory_usage_bytes",
            type=MetricType.GAUGE,
            description="System memory usage in bytes",
            unit="bytes"
        ))
        
        self.register_metric(MetricDefinition(
            name="system_cpu_usage_percent",
            type=MetricType.GAUGE,
            description="System CPU usage percentage",
            unit="percent"
        ))
        
        # Security metrics
        self.register_metric(MetricDefinition(
            name="security_auth_attempts_total",
            type=MetricType.COUNTER,
            description="Total authentication attempts",
            labels=["result", "method"]
        ))
        
        self.register_metric(MetricDefinition(
            name="security_rate_limit_hits_total",
            type=MetricType.COUNTER,
            description="Total rate limit hits",
            labels=["endpoint", "client_id"]
        ))
        
        # Authentication metrics
        self.register_metric(MetricDefinition(
            name="auth_requests_total",
            type=MetricType.COUNTER,
            description="Total authentication requests",
            labels=["method", "success", "role"]
        ))
        
        self.register_metric(MetricDefinition(
            name="auth_successful_logins_total",
            type=MetricType.COUNTER,
            description="Total successful login attempts",
            labels=["method", "role"]
        ))
        
        self.register_metric(MetricDefinition(
            name="auth_failed_logins_total",
            type=MetricType.COUNTER,
            description="Total failed login attempts",
            labels=["method", "reason"]
        ))
        
        self.register_metric(MetricDefinition(
            name="auth_registrations_total",
            type=MetricType.COUNTER,
            description="Total user registrations"
        ))
        
        self.register_metric(MetricDefinition(
            name="auth_password_changes_total",
            type=MetricType.COUNTER,
            description="Total password changes"
        ))
        
        self.register_metric(MetricDefinition(
            name="auth_mfa_enabled_total",
            type=MetricType.COUNTER,
            description="Total MFA activations",
            labels=["method"]
        ))
        
        self.register_metric(MetricDefinition(
            name="auth_oauth_logins_total",
            type=MetricType.COUNTER,
            description="Total OAuth login attempts",
            labels=["provider", "success"]
        ))
        
        self.register_metric(MetricDefinition(
            name="security_events_total",
            type=MetricType.COUNTER,
            description="Total security events",
            labels=["event_type", "severity"]
        ))
        
        self.register_metric(MetricDefinition(
            name="security_alerts_total",
            type=MetricType.COUNTER,
            description="Total security alerts",
            labels=["severity", "threat_level"]
        ))
        
        self.register_metric(MetricDefinition(
            name="security_blocked_requests_total",
            type=MetricType.COUNTER,
            description="Total blocked requests",
            labels=["reason"]
        ))
        
        self.register_metric(MetricDefinition(
            name="rate_limit_exceeded_total",
            type=MetricType.COUNTER,
            description="Total rate limit exceeded events",
            labels=["identifier_type", "endpoint"]
        ))
        
        self.register_metric(MetricDefinition(
            name="rate_limit_requests_total",
            type=MetricType.COUNTER,
            description="Total rate limit checks",
            labels=["identifier_type", "endpoint", "allowed"]
        ))
        
        self.register_metric(MetricDefinition(
            name="active_sessions_total",
            type=MetricType.GAUGE,
            description="Number of active user sessions"
        ))
        
        self.register_metric(MetricDefinition(
            name="users_total",
            type=MetricType.GAUGE,
            description="Total number of users"
        ))
        
        self.register_metric(MetricDefinition(
            name="users_active_total",
            type=MetricType.GAUGE,
            description="Number of active users"
        ))
        
        self.register_metric(MetricDefinition(
            name="user_profile_updates_total",
            type=MetricType.COUNTER,
            description="Total user profile updates"
        ))
        
        self.register_metric(MetricDefinition(
            name="user_preferences_updates_total",
            type=MetricType.COUNTER,
            description="Total user preferences updates"
        ))
        
        self.register_metric(MetricDefinition(
            name="user_deletions_total",
            type=MetricType.COUNTER,
            description="Total user deletions"
        ))
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric definition."""
        with self.lock:
            if definition.name in self.definitions:
                return  # Already registered
            
            self.definitions[definition.name] = definition
            
            # Create Prometheus metric
            if definition.type == MetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.type == MetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.type == MetricType.HISTOGRAM:
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets,
                    registry=self.registry
                )
            elif definition.type == MetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.type == MetricType.INFO:
                metric = Info(
                    definition.name,
                    definition.description,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unsupported metric type: {definition.type}")
            
            self.metrics[definition.name] = metric
    
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self._record_metric(name, value, labels, MetricType.COUNTER)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        self._record_metric(name, value, labels, MetricType.GAUGE)
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value for histogram or summary metrics."""
        definition = self.definitions.get(name)
        if definition and definition.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            self._record_metric(name, value, labels, definition.type)
    
    def time_function(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.observe(name, duration, labels)
            return wrapper
        return decorator
    
    def _record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]], metric_type: MetricType) -> None:
        """Record a metric value."""
        with self.lock:
            metric = self.metrics.get(name)
            if not metric:
                # Auto-register metric if not exists
                definition = MetricDefinition(
                    name=name,
                    type=metric_type,
                    description=f"Auto-generated {metric_type.value} metric",
                    labels=list(labels.keys()) if labels else []
                )
                self.register_metric(definition)
                metric = self.metrics[name]
            
            # Record to Prometheus
            if labels:
                if metric_type == MetricType.COUNTER:
                    metric.labels(**labels).inc(value)
                elif metric_type == MetricType.GAUGE:
                    metric.labels(**labels).set(value)
                elif metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                    metric.labels(**labels).observe(value)
            else:
                if metric_type == MetricType.COUNTER:
                    metric.inc(value)
                elif metric_type == MetricType.GAUGE:
                    metric.set(value)
                elif metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                    metric.observe(value)
            
            # Store historical data
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self.values[name].append(metric_value)
            
            # Cleanup old data periodically
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_data()
                self._last_cleanup = time.time()
    
    def _cleanup_old_data(self) -> None:
        """Clean up old metric data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for name, values in self.values.items():
            while values and values[0].timestamp < cutoff_time:
                values.popleft()
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current metric value."""
        with self.lock:
            metric = self.metrics.get(name)
            if not metric:
                return None
            
            # For Prometheus metrics, we need to access the internal value
            # This is a simplified implementation
            return None  # Prometheus metrics don't expose current values directly
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[MetricValue]:
        """Get metric history."""
        with self.lock:
            values = self.values.get(name, deque())
            return list(values)[-limit:]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics information."""
        with self.lock:
            return {
                "definitions": {name: {
                    "type": defn.type.value,
                    "description": defn.description,
                    "labels": defn.labels,
                    "unit": defn.unit
                } for name, defn in self.definitions.items()},
                "values": {name: [v.to_dict() for v in values] for name, values in self.values.items()}
            }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def export_json(self) -> str:
        """Export metrics in JSON format."""
        return json.dumps(self.get_all_metrics(), indent=2)


class PrometheusExporter:
    """Prometheus metrics exporter."""
    
    def __init__(self, collector: MetricsCollector, port: int = 9090):
        """
        Initialize Prometheus exporter.
        
        Args:
            collector: Metrics collector instance
            port: Port to serve metrics on
        """
        self.collector = collector
        self.port = port
        self.server = None
        self.running = False
    
    async def start(self):
        """Start the Prometheus metrics server."""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/metrics', self._metrics_handler)
        app.router.add_get('/health', self._health_handler)
        
        self.server = web.AppRunner(app)
        await self.server.setup()
        
        site = web.TCPSite(self.server, '0.0.0.0', self.port)
        await site.start()
        
        self.running = True
        print(f"Prometheus metrics server started on port {self.port}")
    
    async def stop(self):
        """Stop the Prometheus metrics server."""
        if self.server:
            await self.server.cleanup()
            self.running = False
            print("Prometheus metrics server stopped")
    
    async def _metrics_handler(self, request):
        """Handle metrics endpoint."""
        from aiohttp import web
        
        metrics_data = self.collector.export_prometheus()
        return web.Response(text=metrics_data, content_type=CONTENT_TYPE_LATEST)
    
    async def _health_handler(self, request):
        """Handle health check endpoint."""
        from aiohttp import web
        
        return web.json_response({"status": "healthy", "metrics_count": len(self.collector.definitions)})


class SystemMetricsCollector:
    """Collector for system-level metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize system metrics collector.
        
        Args:
            metrics_collector: Main metrics collector
        """
        self.metrics_collector = metrics_collector
        self.running = False
        self.collection_interval = 30  # seconds
    
    async def start(self):
        """Start collecting system metrics."""
        self.running = True
        while self.running:
            await self._collect_metrics()
            await asyncio.sleep(self.collection_interval)
    
    def stop(self):
        """Stop collecting system metrics."""
        self.running = False
    
    async def _collect_metrics(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_usage_bytes", memory.used)
            self.metrics_collector.set_gauge("system_memory_usage_percent", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_usage_bytes", disk.used)
            self.metrics_collector.set_gauge("system_disk_usage_percent", (disk.used / disk.total) * 100)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics_collector.set_gauge("system_network_bytes_sent", net_io.bytes_sent)
            self.metrics_collector.set_gauge("system_network_bytes_recv", net_io.bytes_recv)
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            print(f"Error collecting system metrics: {e}")


class ApplicationMetricsCollector:
    """Collector for application-specific metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize application metrics collector.
        
        Args:
            metrics_collector: Main metrics collector
        """
        self.metrics_collector = metrics_collector
        self.request_count = 0
        self.error_count = 0
        self.active_connections = 0
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.metrics_collector.increment(
            "app_requests_total",
            labels={"method": method, "endpoint": endpoint, "status_code": str(status_code)}
        )
        
        self.metrics_collector.observe(
            "app_request_duration_seconds",
            duration,
            labels={"method": method, "endpoint": endpoint}
        )
        
        self.request_count += 1
    
    def record_error(self, error_type: str, component: str):
        """Record application error."""
        self.metrics_collector.increment(
            "app_errors_total",
            labels={"error_type": error_type, "component": component}
        )
        
        self.error_count += 1
    
    def update_active_connections(self, count: int):
        """Update active connections count."""
        self.active_connections = count
        self.metrics_collector.set_gauge("app_active_connections", count)
    
    def record_agent_task(self, agent_id: str, task_type: str, status: str, duration: float):
        """Record agent task metrics."""
        self.metrics_collector.increment(
            "agent_tasks_total",
            labels={"agent_id": agent_id, "task_type": task_type, "status": status}
        )
        
        if status == "completed":
            self.metrics_collector.observe(
                "agent_task_duration_seconds",
                duration,
                labels={"agent_id": agent_id, "task_type": task_type}
            )
    
    def update_active_agent_tasks(self, agent_id: str, count: int):
        """Update active agent tasks count."""
        self.metrics_collector.set_gauge(
            "agent_active_tasks",
            count,
            labels={"agent_id": agent_id}
        )
    
    def record_database_query(self, query_type: str, duration: float, success: bool):
        """Record database query metrics."""
        self.metrics_collector.observe(
            "db_query_duration_seconds",
            duration,
            labels={"query_type": query_type}
        )
        
        if not success:
            self.metrics_collector.increment(
                "db_errors_total",
                labels={"error_type": "query_failed"}
            )
    
    def update_database_connections(self, count: int):
        """Update active database connections count."""
        self.metrics_collector.set_gauge("db_connections_active", count)
    
    def record_security_event(self, event_type: str, result: str, method: str):
        """Record security event metrics."""
        if event_type == "authentication":
            self.metrics_collector.increment(
                "security_auth_attempts_total",
                labels={"result": result, "method": method}
            )
        elif event_type == "rate_limit":
            self.metrics_collector.increment(
                "security_rate_limit_hits_total",
                labels={"endpoint": method, "client_id": result}
            )


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_application_metrics() -> ApplicationMetricsCollector:
    """Get application metrics collector."""
    return ApplicationMetricsCollector(get_metrics_collector())