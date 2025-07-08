"""
Lifecycle monitoring and health checking for agents.
"""

import asyncio
from collections import defaultdict, deque
import contextlib
import logging
import time
from typing import Any, Dict, List, Optional

from ..agents.base import BaseAgent

# Constants
MIN_METRICS_HISTORY_SIZE = 2
MAX_CONSECUTIVE_FAILURES_THRESHOLD = 3


class HealthCheck:
    """
    Health check implementation for agents.

    Provides configurable health checking with timeout
    and failure handling.
    """

    def __init__(self, interval: float = 30.0, timeout: float = 5.0):
        """
        Initialize the health check.

        Args:
            interval: Time between health checks in seconds
            timeout: Timeout for individual health checks
        """
        self.interval = interval
        self.timeout = timeout
        self.last_check_time: Optional[float] = None
        self.last_check_result: Optional[bool] = None
        self.consecutive_failures = 0
        self.total_checks = 0
        self.total_failures = 0

    async def check(self, agent: BaseAgent) -> bool:
        """
        Perform a health check on an agent.

        Args:
            agent: Agent to check

        Returns:
            True if healthy, False otherwise
        """
        self.total_checks += 1
        self.last_check_time = time.time()

        try:
            # Perform health check with timeout
            result = await asyncio.wait_for(
                agent.health_check(),
                timeout=self.timeout
            )

            if result:
                self.consecutive_failures = 0
                self.last_check_result = True
                return True
            else:
                self.consecutive_failures += 1
                self.total_failures += 1
                self.last_check_result = False
                return False

        except TimeoutError:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_check_result = False
            return False
        except Exception:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_check_result = False
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get health check statistics."""
        return {
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "last_check_time": self.last_check_time,
            "last_check_result": self.last_check_result,
            "success_rate": (self.total_checks - self.total_failures) / max(self.total_checks, 1)
        }


class PerformanceMonitor:
    """
    Performance monitoring for agents.

    Tracks metrics, performance trends, and threshold violations.
    """

    def __init__(self, interval: float = 60.0, history_size: int = 100):
        """
        Initialize the performance monitor.

        Args:
            interval: Time between performance checks in seconds
            history_size: Number of historical metrics to keep
        """
        self.interval = interval
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.thresholds: Dict[str, float] = {}
        self.last_check_time: Optional[float] = None

    async def collect_metrics(self, agent: BaseAgent) -> Dict[str, Any]:
        """
        Collect performance metrics from an agent.

        Args:
            agent: Agent to collect metrics from

        Returns:
            Dictionary of metrics
        """
        try:
            metrics = agent.get_metrics()
            metrics["timestamp"] = time.time()

            # Add to history
            self.metrics_history.append(metrics)
            self.last_check_time = time.time()

            return metrics

        except Exception as e:
            # Return error metrics
            return {
                "timestamp": time.time(),
                "error": str(e),
                "collection_failed": True
            }

    def set_threshold(self, metric_name: str, threshold_value: float) -> None:
        """
        Set a threshold for a metric.

        Args:
            metric_name: Name of the metric
            threshold_value: Threshold value
        """
        self.thresholds[metric_name] = threshold_value

    async def check_thresholds(self, agent: BaseAgent) -> List[Dict[str, Any]]:
        """
        Check metrics against configured thresholds.

        Args:
            agent: Agent to check

        Returns:
            List of threshold violations
        """
        violations = []

        try:
            current_metrics = await self.collect_metrics(agent)

            for metric_name, threshold in self.thresholds.items():
                if metric_name in current_metrics:
                    value = current_metrics[metric_name]

                    # Special handling for error rate
                    if metric_name == "error_rate":
                        # Calculate error rate from errors and total operations
                        errors = current_metrics.get("errors", 0)
                        total_ops = (
                            current_metrics.get("messages_processed", 0) +
                            current_metrics.get("tasks_completed", 0)
                        )
                        value = errors / total_ops if total_ops > 0 else 0

                    if isinstance(value, (int, float)) and value > threshold:
                        violations.append({
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold,
                            "timestamp": time.time()
                        })

        except Exception as e:
            violations.append({
                "metric": "collection_error",
                "value": str(e),
                "threshold": "N/A",
                "timestamp": time.time()
            })

        return violations

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical metrics.

        Args:
            limit: Optional limit on number of entries

        Returns:
            List of historical metrics
        """
        history = list(self.metrics_history)

        if limit:
            history = history[-limit:]

        return history

    def calculate_rates(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate rates from current and previous metrics.

        Args:
            current_metrics: Current metrics snapshot

        Returns:
            Dictionary of calculated rates
        """
        rates: Dict[str, float] = {}

        if len(self.metrics_history) < MIN_METRICS_HISTORY_SIZE:
            return rates

        previous_metrics = self.metrics_history[-2]
        time_diff = current_metrics.get("timestamp", 0) - previous_metrics.get("timestamp", 0)

        if time_diff <= 0:
            return rates

        # Calculate message processing rate
        current_messages = current_metrics.get("messages_processed", 0)
        previous_messages = previous_metrics.get("messages_processed", 0)
        message_diff = current_messages - previous_messages

        if message_diff >= 0:
            rates["messages_per_second"] = message_diff / time_diff

        # Calculate task completion rate
        current_tasks = current_metrics.get("tasks_completed", 0)
        previous_tasks = previous_metrics.get("tasks_completed", 0)
        task_diff = current_tasks - previous_tasks

        if task_diff >= 0:
            rates["tasks_per_second"] = task_diff / time_diff

        # Calculate error rate
        current_errors = current_metrics.get("errors", 0)
        previous_errors = previous_metrics.get("errors", 0)
        error_diff = current_errors - previous_errors

        if error_diff >= 0:
            rates["errors_per_second"] = error_diff / time_diff

        return rates

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.

        Returns:
            Performance summary dictionary
        """
        if not self.metrics_history:
            return {"status": "No metrics available"}

        latest_metrics = self.metrics_history[-1]

        # Calculate averages over history
        total_messages = sum(m.get("messages_processed", 0) for m in self.metrics_history)
        total_tasks = sum(m.get("tasks_completed", 0) for m in self.metrics_history)
        total_errors = sum(m.get("errors", 0) for m in self.metrics_history)

        avg_messages = total_messages / len(self.metrics_history)
        avg_tasks = total_tasks / len(self.metrics_history)
        avg_errors = total_errors / len(self.metrics_history)

        return {
            "latest_metrics": latest_metrics,
            "averages": {
                "messages_processed": avg_messages,
                "tasks_completed": avg_tasks,
                "errors": avg_errors
            },
            "history_size": len(self.metrics_history),
            "last_check_time": self.last_check_time,
            "configured_thresholds": dict(self.thresholds)
        }


class LifecycleMonitor:
    """
    Comprehensive lifecycle monitoring for agents.

    Provides health checking, performance monitoring, and
    status tracking for all agents in the system.
    """

    def __init__(
        self,
        logger: logging.Logger,
        health_check_interval: float = 30.0,
        performance_monitor_interval: float = 60.0
    ):
        """
        Initialize the lifecycle monitor.

        Args:
            logger: Logger instance
            health_check_interval: Time between health checks
            performance_monitor_interval: Time between performance checks
        """
        self.logger = logger
        self.health_check_interval = health_check_interval
        self.performance_monitor_interval = performance_monitor_interval

        # Monitored agents and their status
        self.monitored_agents: Dict[str, BaseAgent] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.performance_monitors: Dict[str, PerformanceMonitor] = {}

        # Monitoring state
        self.is_running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._performance_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Status tracking
        self.agent_status: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def start(self) -> None:
        """Start the lifecycle monitor."""
        if self.is_running:
            self.logger.warning("Lifecycle monitor is already running")
            return

        self.logger.info("Starting lifecycle monitor")
        self.is_running = True

        # Start monitoring tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())

        self.logger.info("Lifecycle monitor started successfully")

    async def stop(self) -> None:
        """Stop the lifecycle monitor."""
        if not self.is_running:
            self.logger.warning("Lifecycle monitor is not running")
            return

        self.logger.info("Stopping lifecycle monitor")
        self.is_running = False

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel monitoring tasks
        tasks = [self._health_check_task, self._performance_monitor_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self.logger.info("Lifecycle monitor stopped successfully")

    async def start_monitoring(self, agent: BaseAgent) -> None:
        """
        Start monitoring an agent.

        Args:
            agent: Agent to monitor
        """
        agent_id = agent.agent_id

        if agent_id in self.monitored_agents:
            self.logger.warning(f"Agent {agent_id} is already being monitored")
            return

        self.logger.info(f"Starting monitoring for agent {agent_id}")

        # Add agent to monitoring
        self.monitored_agents[agent_id] = agent
        self.health_checks[agent_id] = HealthCheck(interval=self.health_check_interval)
        self.performance_monitors[agent_id] = PerformanceMonitor(interval=self.performance_monitor_interval)

        # Initialize status
        self.agent_status[agent_id] = {
            "monitoring_started": time.time(),
            "last_health_check": None,
            "last_performance_check": None,
            "health_status": None,
            "performance_violations": []
        }

        self.logger.debug(f"Monitoring started for agent {agent_id}")

    async def stop_monitoring(self, agent_id: str) -> None:
        """
        Stop monitoring an agent.

        Args:
            agent_id: ID of agent to stop monitoring
        """
        if agent_id not in self.monitored_agents:
            self.logger.warning(f"Agent {agent_id} is not being monitored")
            return

        self.logger.info(f"Stopping monitoring for agent {agent_id}")

        # Remove agent from monitoring
        del self.monitored_agents[agent_id]
        del self.health_checks[agent_id]
        del self.performance_monitors[agent_id]
        del self.agent_status[agent_id]

        self.logger.debug(f"Monitoring stopped for agent {agent_id}")

    async def _health_check_loop(self) -> None:
        """Health check monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()

                # Wait for next interval or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.health_check_interval
                    )
                except TimeoutError:
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(1)

    async def _performance_monitor_loop(self) -> None:
        """Performance monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_performance_checks()

                # Wait for next interval or shutdown signal
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.performance_monitor_interval
                    )
                except TimeoutError:
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitor loop: {e}")
                await asyncio.sleep(1)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all monitored agents."""
        if not self.monitored_agents:
            return

        self.logger.debug(f"Performing health checks on {len(self.monitored_agents)} agents")

        for agent_id, agent in self.monitored_agents.items():
            try:
                health_checker = self.health_checks[agent_id]
                is_healthy = await health_checker.check(agent)

                # Update status
                self.agent_status[agent_id]["last_health_check"] = time.time()
                self.agent_status[agent_id]["health_status"] = is_healthy

                if not is_healthy:
                    self.logger.warning(f"Health check failed for agent {agent_id}")

                    # Log consecutive failures
                    if health_checker.consecutive_failures > MAX_CONSECUTIVE_FAILURES_THRESHOLD:
                        self.logger.error(
                            f"Agent {agent_id} has failed {health_checker.consecutive_failures} "
                            "consecutive health checks"
                        )

            except Exception as e:
                self.logger.error(f"Health check error for agent {agent_id}: {e}")
                self.agent_status[agent_id]["health_status"] = False

    async def _perform_performance_checks(self) -> None:
        """Perform performance checks on all monitored agents."""
        if not self.monitored_agents:
            return

        self.logger.debug(f"Performing performance checks on {len(self.monitored_agents)} agents")

        for agent_id, agent in self.monitored_agents.items():
            try:
                performance_monitor = self.performance_monitors[agent_id]
                violations = await performance_monitor.check_thresholds(agent)

                # Update status
                self.agent_status[agent_id]["last_performance_check"] = time.time()
                self.agent_status[agent_id]["performance_violations"] = violations

                if violations:
                    self.logger.warning(f"Performance violations for agent {agent_id}: {violations}")

            except Exception as e:
                self.logger.error(f"Performance check error for agent {agent_id}: {e}")

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get status for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent status dictionary
        """
        if agent_id not in self.monitored_agents:
            return {"error": f"Agent {agent_id} is not being monitored"}

        agent = self.monitored_agents[agent_id]
        status = dict(self.agent_status[agent_id])

        # Add current agent state
        status.update({
            "agent_id": agent_id,
            "state": agent.state.value,
            "is_monitored": True
        })

        # Add health check stats
        if agent_id in self.health_checks:
            status["health_check_stats"] = self.health_checks[agent_id].get_stats()

        # Add performance summary
        if agent_id in self.performance_monitors:
            status["performance_summary"] = self.performance_monitors[agent_id].get_performance_summary()

        return status

    def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all monitored agents.

        Returns:
            Dictionary mapping agent IDs to their status
        """
        all_status = {}

        for agent_id in self.monitored_agents:
            all_status[agent_id] = self.get_agent_status(agent_id)

        return all_status

    def get_monitor_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the monitor status.

        Returns:
            Monitor summary dictionary
        """
        healthy_agents = sum(
            1 for status in self.agent_status.values()
            if status.get("health_status") is True
        )

        unhealthy_agents = sum(
            1 for status in self.agent_status.values()
            if status.get("health_status") is False
        )

        agents_with_violations = sum(
            1 for status in self.agent_status.values()
            if status.get("performance_violations")
        )

        return {
            "is_running": self.is_running,
            "monitored_agents": len(self.monitored_agents),
            "healthy_agents": healthy_agents,
            "unhealthy_agents": unhealthy_agents,
            "agents_with_performance_violations": agents_with_violations,
            "health_check_interval": self.health_check_interval,
            "performance_monitor_interval": self.performance_monitor_interval
        }
